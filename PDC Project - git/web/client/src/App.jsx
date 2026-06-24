import React, { useState } from "react";
import RunConsole from "./components/RunConsole.jsx";
import Benchmarks from "./components/Benchmarks.jsx";
import About from "./components/About.jsx";

const TABS = [
  { id: "run", label: "Run Console" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "about", label: "About" },
];

export default function App() {
  const [tab, setTab] = useState("run");

  return (
    <div className="app">
      <header className="site-header">
        <div className="container">
          <div className="brand">
            <div className="logo">F</div>
            <div>
              <div className="title">Foundry</div>
              <div className="subtitle">
                Parallel text preprocessing for LLM fine-tuning
                <span className="brand-tags">OpenMP · MPI · Hybrid</span>
              </div>
            </div>
          </div>
          <nav className="nav">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={tab === t.id ? "active" : ""}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main>
        <div className="container">
          {tab === "run" && <RunConsole />}
          {tab === "benchmarks" && <Benchmarks />}
          {tab === "about" && <About />}
        </div>
      </main>
    </div>
  );
}
