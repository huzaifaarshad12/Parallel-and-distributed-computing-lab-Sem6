import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid,
} from "recharts";
import { getDatasets, getHealth, runPipeline } from "../api.js";

const fmt = (n) => (n == null ? "—" : n.toLocaleString());
const STAGE_COLORS = {
  load: "#948a78", clean: "#cc6a48", filter: "#3d6b94", dedup: "#7a9a5b", tokenize: "#c79a3e",
};

const MODES = [
  { id: "sequential", label: "Sequential", phase: "Phase 1", env: "Windows", needs: [] },
  { id: "openmp", label: "OpenMP", phase: "Phase 2", env: "Windows", needs: ["threads"] },
  { id: "mpi", label: "MPI", phase: "Phase 3", env: "WSL", needs: ["procs"] },
  { id: "hybrid", label: "Hybrid", phase: "Phase 4", env: "WSL", needs: ["procs", "threads"] },
];

export default function RunConsole() {
  const [datasets, setDatasets] = useState([]);
  const [health, setHealth] = useState(null);
  const [dataset, setDataset] = useState("10k");
  const [version, setVersion] = useState("openmp");
  const [minWords, setMinWords] = useState(5);
  const [threads, setThreads] = useState(8);
  const [procs, setProcs] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    getDatasets().then(setDatasets).catch((e) => setError(e.message));
    getHealth().then(setHealth).catch(() => {});
  }, []);

  const mode = MODES.find((m) => m.id === version);
  const needs = (k) => mode.needs.includes(k);
  const binaryReady =
    !health ? true :
    version === "sequential" ? health.binaries.seq :
    version === "openmp" ? health.binaries.omp :
    version === "mpi" ? health.binaries.mpi :
    health.binaries.hybrid;

  const run = async () => {
    setLoading(true); setError("");
    try {
      setResult(await runPipeline({ dataset, version, minWords, threads, procs }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const timingData =
    result?.timings
      ? ["load", "clean", "filter", "dedup", "tokenize"].map((s) => ({ stage: s, ms: result.timings[s] ?? 0 }))
      : [];

  return (
    <>
      <section className="hero">
        <h1>Run the pipeline <span className="accent">live</span>.</h1>
        <p>
          Pick a dataset and a mode, then run the real pipeline live — every run is
          verified <b>byte-identical</b> to sequential and timed for speedup.
        </p>
        <div className="pillrow">
          {MODES.map((m) => (
            <span key={m.id} className="pill">
              {m.label}: <b>{m.phase}</b> · {m.env}
            </span>
          ))}
        </div>
      </section>

      <div className="grid" style={{ marginTop: 22 }}>
        <div className="card">
          <h2>Configuration</h2>
          <p className="hint">
            All four versions run live and produce a <b>byte-identical</b> cleaned corpus —
            only the speed differs.
          </p>

          <div className="field" style={{ marginBottom: 18 }}>
            <label>Parallelism mode</label>
            <div className="segmented seg-4" role="tablist">
              {MODES.map((m) => (
                <button key={m.id} className={version === m.id ? "on" : ""} onClick={() => setVersion(m.id)}>
                  {m.label}
                  <span className="seg-sub">{m.phase}</span>
                </button>
              ))}
            </div>
            <span className="desc">
              {mode.env === "WSL"
                ? "↪ Runs in WSL (Ubuntu · OpenMPI). Make sure WSL is available."
                : "Runs natively on Windows (MinGW g++)."}
            </span>
          </div>

          <div className="controls">
            <div className="field">
              <label>Dataset</label>
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id} disabled={!d.exists}>
                    {d.label} {d.exists ? `· ${fmt(d.docs)} docs` : "· missing"}
                  </option>
                ))}
              </select>
              <span className="desc">One news article per line.</span>
            </div>

            <div className="field">
              <label>Minimum words <span className="desc">(quality filter)</span></label>
              <input type="number" min="0" max="1000" value={minWords} onChange={(e) => setMinWords(e.target.value)} />
              <span className="desc">Raise (e.g. 30) to see documents dropped.</span>
            </div>

            <div className="field" style={{ opacity: needs("procs") ? 1 : 0.45 }}>
              <label>MPI processes {!needs("procs") && <span className="desc">(MPI / Hybrid only)</span>}</label>
              <input type="number" min="1" max="16" value={procs} disabled={!needs("procs")} onChange={(e) => setProcs(e.target.value)} />
              <span className="desc">Distributed-memory ranks (uses --oversubscribe).</span>
            </div>

            <div className="field" style={{ opacity: needs("threads") ? 1 : 0.45 }}>
              <label>OpenMP threads {!needs("threads") && <span className="desc">(OpenMP / Hybrid only)</span>}</label>
              <input type="number" min="1" max="64" value={threads} disabled={!needs("threads")} onChange={(e) => setThreads(e.target.value)} />
              <span className="desc">Shared-memory worker threads.</span>
            </div>
          </div>

          {!binaryReady && (
            <div className="error" style={{ marginTop: 16 }}>
              ⚠ The {mode.label} binary isn’t built yet.{" "}
              {mode.env === "WSL" ? "Build it in WSL: ./build.sh" : "Build it on Windows: .\\build.ps1"}
            </div>
          )}

          <div className="btn-row">
            <button className="btn" onClick={run} disabled={loading || !binaryReady}>
              {loading ? <><span className="spinner" />&nbsp; Running {mode.env === "WSL" ? "in WSL" : ""}…</> : "▶  Run pipeline"}
            </button>
            {result && !loading && <span className="desc">Server wall time: {result.serverWallMs} ms</span>}
          </div>

          {error && <div className="error" style={{ marginTop: 16 }}>⚠ {error}</div>}
        </div>

        {!result && !loading && (
          <div className="card empty-card">
            <div className="empty-emoji">⚙️</div>
            <h2>No run yet</h2>
            <p className="hint" style={{ marginBottom: 0 }}>
              Choose a mode above and hit <b>Run pipeline</b>. You’ll see the live
              counters, a speedup vs. sequential, the per-stage timing chart, and a
              cleaned-output preview — all from the real binary.
            </p>
          </div>
        )}

        {result && (
          <>
            <div className="card">
              <div className="result-head">
                <div>
                  <h2 style={{ marginBottom: 2 }}>{mode.label} — {result.workers}</h2>
                  <p className="hint" style={{ margin: 0 }}>{result.datasetLabel} · min_words = {result.minWords} · {result.environment}</p>
                </div>
                <div className="speedup-hero">
                  <div className="su-val">{result.speedup ? `${result.speedup}×` : "—"}</div>
                  <div className="su-lbl">vs sequential</div>
                </div>
              </div>

              <div className="stats" style={{ marginTop: 18 }}>
                <div className="stat"><div className="k">Total documents</div><div className="v">{fmt(result.counters.total)}</div></div>
                <div className="stat"><div className="k">Removed</div><div className="v">{fmt(result.counters.removed)}</div></div>
                <div className="stat"><div className="k">Duplicates</div><div className="v">{fmt(result.counters.duplicates)}</div></div>
                <div className="stat accent"><div className="k">Kept</div><div className="v">{fmt(result.counters.kept)}</div></div>
                <div className="stat accent"><div className="k">Total tokens</div><div className="v">{fmt(result.counters.tokens)}</div></div>
              </div>

              <div className="badge-row">
                {result.matchesSequential
                  ? <span className="badge ok"><span className="dot" /> Byte-identical to sequential ✓</span>
                  : <span className="badge warn"><span className="dot" /> Differs from sequential</span>}
                <span className="badge soft">Total: {result.timings.total} ms</span>
                <span className="badge soft">Baseline (seq): {result.baselineMs} ms</span>
              </div>

              <div className="cmd" title="The exact command executed">$ {result.commandText}</div>
              <div className="hashline">SHA-256: {result.hash}</div>
            </div>

            <div className="card">
              <h2>Per-stage timings</h2>
              <p className="hint">Where the work goes — file load is serial I/O; clean / filter / tokenize parallelize well.</p>
              <div className="chart-wrap">
                <ResponsiveContainer>
                  <BarChart data={timingData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ece2cf" vertical={false} />
                    <XAxis dataKey="stage" tick={{ fontSize: 12, fill: "#6f6657" }} />
                    <YAxis tick={{ fontSize: 12, fill: "#6f6657" }} unit=" ms" width={64} />
                    <Tooltip formatter={(v) => [`${v} ms`, "time"]} cursor={{ fill: "rgba(204,106,72,0.06)" }} />
                    <Bar dataKey="ms" radius={[6, 6, 0, 0]}>
                      {timingData.map((d) => <Cell key={d.stage} fill={STAGE_COLORS[d.stage]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <h2>Cleaned output preview</h2>
              <p className="hint">First lines of the cleaned corpus written by this run.</p>
              <div className="mono">{result.preview.join("\n")}</div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
