import React from "react";

const STAGES = [
  { n: 1, t: "Text Cleaning", d: "Lowercase, strip special characters, collapse whitespace — one cache-friendly pass.", c: "#cc6a48" },
  { n: 2, t: "Quality Filter", d: "Drop documents below a minimum word count (default 5).", c: "#3d6b94" },
  { n: 3, t: "Deduplication", d: "64-bit FNV-1a fingerprint per document; keep the first occurrence only.", c: "#7a9a5b" },
  { n: 4, t: "Tokenization", d: "Whitespace tokenization; sum tokens across survivors (a reduction).", c: "#c79a3e" },
];

const IMPLS = [
  { t: "Sequential", d: "The correctness reference baseline.", tag: "Phase 1" },
  { t: "OpenMP", d: "Shared-memory threads, up to 2.94× speedup.", tag: "Phase 2" },
  { t: "MPI", d: "Distributed processes, scatter / gather.", tag: "Phase 3" },
  { t: "Hybrid", d: "MPI across nodes × OpenMP within each.", tag: "Phase 4" },
];

export default function About() {
  return (
    <div className="about">
      <section className="hero about-hero">
        <h1>About <span className="accent">Foundry</span>.</h1>
        <p className="lead-big">
          A data-preprocessing pipeline that turns raw text into a clean corpus
          ready for LLM fine-tuning — written in C++ in four progressively parallel
          versions that all produce a <b>byte-identical</b> result.
        </p>
      </section>

      <h2 className="section-title">The pipeline</h2>
      <p className="section-sub">Four independent, per-document stages run in a fixed order.</p>
      <div className="flow big-flow">
        <span className="step">Clean</span><span className="arrow">→</span>
        <span className="step">Quality filter</span><span className="arrow">→</span>
        <span className="step">Deduplicate</span><span className="arrow">→</span>
        <span className="step">Tokenize / count</span>
      </div>

      <div className="stage-grid">
        {STAGES.map((s) => (
          <div className="stage-card lift" key={s.n}>
            <div className="stage-badge" style={{ background: s.c }}>{s.n}</div>
            <h3>{s.t}</h3>
            <p>{s.d}</p>
          </div>
        ))}
      </div>

      <div className="about-cols">
        <div className="card lift">
          <span className="tag">DESIGN</span>
          <h2>One source of truth</h2>
          <p className="about-p">
            All stage logic lives in <code>src/common/</code> as pure functions.
            Each version only changes <i>how it loops</i> over documents — so a
            parallel run can differ in speed, <b>never in output</b>. The
            order-dependent steps (dedup decision, write order) stay serial and
            id-ordered, guaranteeing byte-identical results by construction.
          </p>
        </div>

        <div className="card lift">
          <span className="tag">VERSIONS</span>
          <h2>Four implementations</h2>
          <div className="impl-grid">
            {IMPLS.map((m) => (
              <div className="impl-card" key={m.t}>
                <div className="impl-head"><b>{m.t}</b><span>{m.tag}</span></div>
                <p>{m.d}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="endcard lift">
        <div className="endcard-text">
          <span className="tag light">END TO END</span>
          <h2>From messy data to a fine-tuned model</h2>
          <p>
            The cleaned corpus fine-tuned <b>DistilGPT-2</b> on a Colab T4 GPU,
            producing coherent, news-style text. Raw AG News → parallel cleaning
            (identical from every version) → a real language model trained on the
            output. Dataset: AG News (~120,000 articles) from the Hugging Face Hub.
          </p>
        </div>
        <div className="endcard-stat">
          <div className="big-num">≈ 95.3</div>
          <div className="big-lbl">validation perplexity</div>
        </div>
      </div>
    </div>
  );
}
