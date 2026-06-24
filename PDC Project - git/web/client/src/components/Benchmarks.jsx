import React, { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import { getOpenmpBenchmarks, getMpiBenchmarks } from "../api.js";

const DS_COLORS = { "10k": "#3d6b94", "50k": "#7a9a5b", full: "#cc6a48" };
const grid = "#ece2cf";
const axisTick = { fontSize: 12, fill: "#6f6657" };

function MultiLine({ data, xKey, datasets, unit, yLabel }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis dataKey={xKey} tick={axisTick} label={{ value: xKey, position: "insideBottom", offset: -2, fontSize: 11, fill: "#948a78" }} />
          <YAxis tick={axisTick} unit={unit} width={56} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {datasets.map((d) => (
            <Line key={d} type="monotone" dataKey={d} stroke={DS_COLORS[d]} strokeWidth={2.5} dot={{ r: 3 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function GroupedBar({ data, xKey, datasets, unit }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
          <XAxis dataKey={xKey} tick={axisTick} />
          <YAxis tick={axisTick} unit={unit} width={56} />
          <Tooltip cursor={{ fill: "rgba(204,106,72,0.06)" }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {datasets.map((d) => (
            <Bar key={d} dataKey={d} fill={DS_COLORS[d]} radius={[4, 4, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Benchmarks() {
  const [omp, setOmp] = useState(null);
  const [mpi, setMpi] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getOpenmpBenchmarks().then(setOmp).catch((e) => setError(e.message));
    getMpiBenchmarks().then(setMpi).catch((e) => setError(e.message));
  }, []);

  return (
    <>
      <section className="hero">
        <h1>Performance <span className="accent">benchmarks</span>.</h1>
        <p>
          Measured on a 12-core laptop over real AG News, best of several runs.
          Speedup is <code>S = T(1)/T(p)</code>; efficiency is <code>E = S/p</code>.
          The headline lesson: OpenMP wins on a single node, while MPI's payoff is
          across machines.
        </p>
      </section>

      {error && <div className="error" style={{ marginTop: 16 }}>⚠ {error}</div>}

      <h3 className="section-title">OpenMP — shared memory</h3>
      <p className="section-sub">Per-document work spread across threads on one machine.</p>
      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="card">
          <h2>Speedup vs threads</h2>
          {omp ? <MultiLine data={omp.speedup} xKey="threads" datasets={omp.datasets} unit="×" /> : <div className="placeholder">Loading…</div>}
          <p className="legendnote">Peak ≈ 2.94× (10K, 12 threads).</p>
        </div>
        <div className="card">
          <h2>Efficiency vs threads</h2>
          {omp ? <MultiLine data={omp.efficiency} xKey="threads" datasets={omp.datasets} unit="%" /> : <div className="placeholder">Loading…</div>}
          <p className="legendnote">High at 2 threads (~91%), falls as the memory bus saturates.</p>
        </div>
      </div>

      <h3 className="section-title">MPI — distributed memory</h3>
      <p className="section-sub">Master–worker scatter/gather across processes (baseline = 1 process).</p>
      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="card">
          <h2>MPI speedup vs processes</h2>
          {mpi ? <MultiLine data={mpi.mpi.speedup} xKey="procs" datasets={mpi.datasets} unit="×" /> : <div className="placeholder">Loading…</div>}
          <p className="legendnote">Lower than OpenMP — single node shares one bus + copies data.</p>
        </div>
        <div className="card">
          <h2>Hybrid speedup by config</h2>
          {mpi ? <GroupedBar data={mpi.hybrid.speedup} xKey="config" datasets={mpi.datasets} unit="×" /> : <div className="placeholder">Loading…</div>}
          <p className="legendnote">procs × threads; peak ≈ 1.80× (50K, 2×2).</p>
        </div>
      </div>

      <hr className="divider" />
      <div className="card explainer">
        <h2>Why does MPI scale less than OpenMP on one node?</h2>
        <p className="lead">
          It’s expected — and it’s the key lesson of this project. On a <b>single
          machine</b>, three things hold MPI back:
        </p>
        <ul className="why-list">
          <li>
            <b>Same hardware ceiling.</b> Every MPI process shares the <b>one</b>
            memory bus and <b>one</b> disk — the exact limit OpenMP already hit.
          </li>
          <li>
            <b>Extra copying.</b> MPI copies every document between processes;
            shared-memory OpenMP shares the data and copies nothing.
          </li>
          <li>
            <b>A serial file read.</b> Only one process reads the dataset, which
            caps the whole run no matter how many processes you add (Amdahl’s law).
          </li>
        </ul>
      </div>
    </>
  );
}
