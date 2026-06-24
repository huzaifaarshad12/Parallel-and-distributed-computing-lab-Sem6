// ---------------------------------------------------------------------------
// server.js — backend API for the PDC pipeline web UI
// ---------------------------------------------------------------------------
// Runs the REAL compiled pipeline binaries and serves their output as JSON:
//   * Sequential / OpenMP  -> Windows binaries (build/*.exe), executed directly.
//   * MPI / Hybrid         -> Linux binaries (build/mpi_pipeline, hybrid_pipeline)
//                             executed inside WSL via `wsl.exe ... mpirun ...`.
// Every run is checked byte-for-byte (SHA-256) against the same-environment
// sequential reference, and timed against it for a live speedup. The benchmark
// CSVs (all 4 versions) are aggregated for the charts.
// ---------------------------------------------------------------------------
import express from "express";
import cors from "cors";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..", "..");
const BUILD = path.join(ROOT, "build");
const DATA = path.join(ROOT, "data", "agnews");
const RESULTS = path.join(ROOT, "results");
const CLIENT_DIST = path.join(__dirname, "..", "client", "dist");
const PORT = process.env.PORT || 3001;
const WSL_DISTRO = process.env.WSL_DISTRO || "Ubuntu";

// project root as a WSL path:  D:\a\b  ->  /mnt/d/a/b
function toWslPath(winPath) {
  const m = winPath.match(/^([A-Za-z]):[\\/](.*)$/);
  return m ? `/mnt/${m[1].toLowerCase()}/${m[2].replace(/\\/g, "/")}` : winPath;
}
const WSL_ROOT = toWslPath(ROOT);

const app = express();
app.use(cors());
app.use(express.json());

// ---- datasets -------------------------------------------------------------
const DATASETS = [
  { id: "1k", file: "agnews_1k.txt", label: "AG News 1K" },
  { id: "10k", file: "agnews_10k.txt", label: "AG News 10K" },
  { id: "50k", file: "agnews_50k.txt", label: "AG News 50K" },
  { id: "full", file: "agnews_full.txt", label: "AG News Full (~120K)" },
];
const lineCountCache = new Map();
function countLines(file) {
  if (lineCountCache.has(file)) return lineCountCache.get(file);
  const buf = fs.readFileSync(file);
  let n = 0;
  for (let i = 0; i < buf.length; i++) if (buf[i] === 10) n++;
  if (buf.length && buf[buf.length - 1] !== 10) n++;
  lineCountCache.set(file, n);
  return n;
}

// ---- output parsing -------------------------------------------------------
function parsePipelineOutput(stdout) {
  const num = (re) => {
    const m = stdout.match(re);
    return m ? Number(m[1]) : null;
  };
  return {
    counters: {
      total: num(/Total documents\s*:\s*(\d+)/),
      removed: num(/Removed \(low quality\)\s*:\s*(\d+)/),
      duplicates: num(/Duplicates removed\s*:\s*(\d+)/),
      kept: num(/Kept documents\s*:\s*(\d+)/),
      tokens: num(/Total tokens\s*:\s*(\d+)/),
    },
    timings: {
      load: num(/load\s*:\s*([\d.]+)/),
      clean: num(/clean\s*:\s*([\d.]+)/),
      filter: num(/filter\s*:\s*([\d.]+)/),
      dedup: num(/dedup\s*:\s*([\d.]+)/),
      tokenize: num(/tokenize\s*:\s*([\d.]+)/),
      total: num(/TOTAL\s*:\s*([\d.]+)/),
    },
  };
}

// ---- process runners ------------------------------------------------------
function runProcess(cmd, args, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd: ROOT, windowsHide: true });
    let stdout = "", stderr = "";
    const timer = setTimeout(() => { child.kill(); reject(new Error("Run timed out.")); }, timeoutMs);
    child.stdout.on("data", (d) => (stdout += d.toString("utf8")));
    child.stderr.on("data", (d) => (stderr += d.toString("utf8")));
    child.on("error", (e) => { clearTimeout(timer); reject(e); });
    child.on("close", (code) => {
      clearTimeout(timer);
      // strip stray NULs that wsl.exe can interleave
      stdout = stdout.split(String.fromCharCode(0)).join("");
      if (code !== 0) return reject(new Error((stderr || stdout || `exit ${code}`).trim().slice(0, 400)));
      resolve(stdout);
    });
  });
}
const runWin = (exe, args) => {
  const p = path.join(BUILD, exe);
  if (!fs.existsSync(p)) return Promise.reject(new Error(`Binary not found: build/${exe}. Run .\\build.ps1 first.`));
  return runProcess(p, args);
};
const runWsl = (bashCmd) => runProcess("wsl.exe", ["-d", WSL_DISTRO, "--", "bash", "-lc", bashCmd]);

function sha256(file) { return createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }
function previewLines(file, n = 12) {
  const buf = fs.readFileSync(file);
  return buf.toString("utf8", 0, Math.min(buf.length, 200000)).split("\n").slice(0, n).filter((l) => l.length);
}

// ---- sequential reference (per environment) -------------------------------
// Returns {hash, totalMs}; cached. env = "windows" | "wsl".
const refCache = new Map();
async function seqReference(env, ds, mw) {
  const key = `${env}:${ds.file}:${mw}`;
  if (refCache.has(key)) return refCache.get(key);
  let out, stdout;
  if (env === "wsl") {
    out = path.join(RESULTS, `_web_ref_wsl_${ds.id}_${mw}.txt`);
    stdout = await runWsl(`cd '${WSL_ROOT}' && build/seq_pipeline 'data/agnews/${ds.file}' 'results/${path.basename(out)}' ${mw}`);
  } else {
    out = path.join(RESULTS, `_web_ref_win_${ds.id}_${mw}.txt`);
    stdout = await runWin("seq_pipeline.exe", [path.join(DATA, ds.file), out, String(mw)]);
  }
  const parsed = parsePipelineOutput(stdout);
  const ref = { hash: sha256(out), totalMs: parsed.timings.total };
  refCache.set(key, ref);
  return ref;
}

// ---- CSV aggregation (benchmarks) -----------------------------------------
function readCsv(file) {
  if (!fs.existsSync(file)) return [];
  const text = fs.readFileSync(file, "utf8").replace(/^﻿/, "");
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  const header = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const c = line.split(",");
    const row = {};
    header.forEach((h, i) => (row[h] = c[i]));
    return row;
  });
}
const minBy = (rows, keyFn, valFn) => {
  const m = new Map();
  for (const r of rows) {
    const k = keyFn(r), v = valFn(r);
    if (!m.has(k) || v < m.get(k)) m.set(k, v);
  }
  return m;
};
function openmpBenchmarks() {
  const rows = readCsv(path.join(RESULTS, "benchmark_openmp.csv"));
  if (!rows.length) return null;
  const datasets = ["10k", "50k", "full"];
  const threads = [...new Set(rows.map((r) => Number(r.threads)))].sort((a, b) => a - b);
  const best = minBy(rows, (r) => `${r.dataset}|${r.threads}`, (r) => Number(r.total_ms));
  const time = [], speedup = [], efficiency = [];
  for (const t of threads) {
    const rT = { threads: t }, rS = { threads: t }, rE = { threads: t };
    for (const d of datasets) {
      const tp = best.get(`${d}|${t}`), t1 = best.get(`${d}|${threads[0]}`);
      rT[d] = tp != null ? +tp.toFixed(1) : null;
      const s = tp != null && t1 != null ? t1 / tp : null;
      rS[d] = s != null ? +s.toFixed(2) : null;
      rE[d] = s != null ? +((s / t) * 100).toFixed(0) : null;
    }
    time.push(rT); speedup.push(rS); efficiency.push(rE);
  }
  return { datasets, threads, time, speedup, efficiency };
}
function mpiBenchmarks() {
  const rows = readCsv(path.join(RESULTS, "benchmark_mpi.csv"));
  if (!rows.length) return null;
  const datasets = ["10k", "50k", "full"];
  const mpiRows = rows.filter((r) => r.version === "mpi");
  const hybRows = rows.filter((r) => r.version === "hybrid");
  const procs = [...new Set(mpiRows.map((r) => Number(r.procs)))].sort((a, b) => a - b);
  const bestMpi = minBy(mpiRows, (r) => `${r.dataset}|${r.procs}`, (r) => Number(r.total_ms));
  const base = {};
  for (const d of datasets) base[d] = bestMpi.get(`${d}|1`);
  const mpiTime = [], mpiSpeed = [];
  for (const p of procs) {
    const rT = { procs: p }, rS = { procs: p };
    for (const d of datasets) {
      const tp = bestMpi.get(`${d}|${p}`);
      rT[d] = tp != null ? +tp.toFixed(0) : null;
      rS[d] = tp != null && base[d] != null ? +(base[d] / tp).toFixed(2) : null;
    }
    mpiTime.push(rT); mpiSpeed.push(rS);
  }
  const bestHyb = minBy(hybRows, (r) => `${r.dataset}|${r.procs}x${r.threads}`, (r) => Number(r.total_ms));
  const configs = [...new Set(hybRows.map((r) => `${r.procs}x${r.threads}`))];
  const hybSpeed = configs.map((c) => {
    const row = { config: c };
    for (const d of datasets) {
      const tp = bestHyb.get(`${d}|${c}`);
      row[d] = tp != null && base[d] != null ? +(base[d] / tp).toFixed(2) : null;
    }
    return row;
  });
  return { datasets, mpi: { procs, time: mpiTime, speedup: mpiSpeed }, hybrid: { configs, speedup: hybSpeed } };
}

// ---- routes ---------------------------------------------------------------
app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    binaries: {
      seq: fs.existsSync(path.join(BUILD, "seq_pipeline.exe")),
      omp: fs.existsSync(path.join(BUILD, "omp_pipeline.exe")),
      mpi: fs.existsSync(path.join(BUILD, "mpi_pipeline")),
      hybrid: fs.existsSync(path.join(BUILD, "hybrid_pipeline")),
    },
  });
});

app.get("/api/datasets", (_req, res) => {
  res.json(
    DATASETS.map((d) => {
      const file = path.join(DATA, d.file);
      const exists = fs.existsSync(file);
      return { id: d.id, label: d.label, file: d.file, exists, docs: exists ? countLines(file) : 0, bytes: exists ? fs.statSync(file).size : 0 };
    })
  );
});

app.post("/api/run", async (req, res) => {
  try {
    const { dataset, version, minWords = 5, threads = 8, procs = 4 } = req.body || {};
    const ds = DATASETS.find((d) => d.id === dataset);
    if (!ds) return res.status(400).json({ error: "Unknown dataset." });
    if (!["sequential", "openmp", "mpi", "hybrid"].includes(version))
      return res.status(400).json({ error: "Invalid version." });
    if (!fs.existsSync(path.join(DATA, ds.file))) return res.status(400).json({ error: "Dataset file missing." });

    const mw = Math.max(0, Math.min(1000, parseInt(minWords, 10) || 0));
    const th = Math.max(1, Math.min(64, parseInt(threads, 10) || 1));
    const np = Math.max(1, Math.min(16, parseInt(procs, 10) || 1));

    const inWin = path.join(DATA, ds.file);
    const outFile = path.join(RESULTS, `_web_${version}_${ds.id}.txt`);
    const outBase = path.basename(outFile);
    const isWsl = version === "mpi" || version === "hybrid";

    let stdout, engineLabel, commandText, environment, workers;
    const t0 = Date.now();

    if (version === "sequential") {
      stdout = await runWin("seq_pipeline.exe", [inWin, outFile, String(mw)]);
      engineLabel = "seq_pipeline.exe"; environment = "Windows · MinGW g++";
      commandText = `build\\seq_pipeline.exe ${ds.file} ${outBase} ${mw}`;
      workers = "1 thread";
    } else if (version === "openmp") {
      stdout = await runWin("omp_pipeline.exe", [inWin, outFile, String(mw), String(th)]);
      engineLabel = "omp_pipeline.exe"; environment = "Windows · MinGW g++ · OpenMP";
      commandText = `build\\omp_pipeline.exe ${ds.file} ${outBase} ${mw} ${th}`;
      workers = `${th} threads`;
    } else if (version === "mpi") {
      if (!fs.existsSync(path.join(BUILD, "mpi_pipeline")))
        return res.status(400).json({ error: "build/mpi_pipeline not found. Build it in WSL: ./build.sh" });
      commandText = `mpirun --oversubscribe -np ${np} build/mpi_pipeline ${ds.file} ${outBase} ${mw}`;
      stdout = await runWsl(`cd '${WSL_ROOT}' && mpirun --allow-run-as-root --oversubscribe -np ${np} build/mpi_pipeline 'data/agnews/${ds.file}' 'results/${outBase}' ${mw}`);
      engineLabel = "mpirun · mpi_pipeline"; environment = "WSL Ubuntu · OpenMPI";
      workers = `${np} processes`;
    } else {
      if (!fs.existsSync(path.join(BUILD, "hybrid_pipeline")))
        return res.status(400).json({ error: "build/hybrid_pipeline not found. Build it in WSL: ./build.sh" });
      commandText = `mpirun --oversubscribe -np ${np} build/hybrid_pipeline ${ds.file} ${outBase} ${mw} ${th}`;
      stdout = await runWsl(`cd '${WSL_ROOT}' && mpirun --allow-run-as-root --oversubscribe -np ${np} build/hybrid_pipeline 'data/agnews/${ds.file}' 'results/${outBase}' ${mw} ${th}`);
      engineLabel = "mpirun · hybrid_pipeline"; environment = "WSL Ubuntu · OpenMPI + OpenMP";
      workers = `${np} procs × ${th} threads`;
    }
    const serverWallMs = Date.now() - t0;

    const parsed = parsePipelineOutput(stdout);
    const hash = sha256(outFile);
    const ref = await seqReference(isWsl ? "wsl" : "windows", ds, mw);
    const speedup = ref.totalMs && parsed.timings.total ? +(ref.totalMs / parsed.timings.total).toFixed(2) : null;

    res.json({
      version, dataset: ds.id, datasetLabel: ds.label,
      minWords: mw, threads: th, procs: np, workers,
      environment, engineLabel, commandText,
      ...parsed,
      hash, referenceHash: ref.hash, matchesSequential: hash === ref.hash,
      baselineMs: ref.totalMs, speedup,
      serverWallMs, preview: previewLines(outFile), rawOutput: stdout.trim(),
    });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.get("/api/benchmarks/openmp", (_req, res) => {
  const d = openmpBenchmarks();
  d ? res.json(d) : res.status(404).json({ error: "benchmark_openmp.csv not found." });
});
app.get("/api/benchmarks/mpi", (_req, res) => {
  const d = mpiBenchmarks();
  d ? res.json(d) : res.status(404).json({ error: "benchmark_mpi.csv not found." });
});

if (fs.existsSync(CLIENT_DIST)) {
  app.use(express.static(CLIENT_DIST));
  app.get("*", (_req, res) => res.sendFile(path.join(CLIENT_DIST, "index.html")));
}

app.listen(PORT, () => {
  console.log(`PDC pipeline API + UI on http://localhost:${PORT}`);
  console.log(`Project root: ${ROOT}  (WSL: ${WSL_ROOT})`);
});
