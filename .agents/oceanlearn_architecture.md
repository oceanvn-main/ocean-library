# OceanLearn — Web Platform Architecture
## A Universal Pedagogical Platform for AI/ML Fundamentals

> **Status: PLANNED — Not yet implemented.**  
> This document describes the future OceanLearn web application.  
> Current focus is on MATLAB experiments. OceanLearn will be built on top of those.  
> Read `architecture.md` for the current MATLAB library structure.

---

## 1. Vision & Core Insight

### The Problem
Students in AI/ML study domains in silos. A student of Signal Processing learns "convolution." A student of Computer Vision learns "spatial filtering." A student of NLP learns "attention." A student of Time Series learns "autoregression." A student of ML learns "kernel methods."

**These are all the same operation in different clothing.** No existing platform shows this.

### The Core Insight — Three Tiers of Knowledge

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1: DOMAIN CONCEPTS (what you study in your textbook)          │
│                                                                     │
│  Signal     Computer     Time        NLP /        Traditional       │
│  Processing  Vision      Series      Language      ML               │
│  ─────────  ─────────   ─────────   ─────────    ─────────         │
│  Sampling   Aliasing    Stationarity Tokenization  Feature Scaling  │
│  Filtering  Edge Det.   AR/MA Model  Word Embed.   Regularization   │
│  STFT       Pyramids    Spectral     Attention      Loss Functions   │
│  Wavelets   Warping     Covariance   Seq2Seq        Kernel Methods   │
│                                                                     │
│                         linked via CrossBridges                    │
│                                                                     │
│  TIER 2: ALGORITHMIC CONCEPTS (how it is computed)                  │
│                                                                     │
│  Convolution · Projection · Basis Decomposition · Sampling         │
│  Optimization · Distance Metric · Graph Laplacian                  │
│                                                                     │
│                         instantiated from                          │
│                                                                     │
│  TIER 3: MATHEMATICAL ATOMS (why it works — domain-agnostic)        │
│                                                                     │
│  Regularization · Loss Functions · Linear Algebra                  │
│  Probability/Bayes · Fourier Analysis · Information Theory         │
│  Manifolds · Optimization Theory · Measure Theory                  │
└─────────────────────────────────────────────────────────────────────┘
```

**OceanLearn's job**: let students navigate vertically (drill into why) and horizontally (see the same atom in a different domain).

---

## 2. Example — The Same Atom Across Domains

The concept **"Regularization"** (Tier 3 Atom) surfaces as:

| Domain | Domain-Specific Name | What it Actually Does |
| :--- | :--- | :--- |
| Signal Processing | Tikhonov smoothing | Penalizes high-frequency noise |
| Computer Vision | Total Variation (TV) prior | Penalizes image roughness |
| Time Series | Ridge on AR coefficients | Prevents overfitting in AR models |
| NLP | Weight decay in transformers | Prevents overfitting on sparse tokens |
| Traditional ML | L1/L2 regularization | Shrinks or zeros out model parameters |

---

## 3. Every Concept Follows the Same Pedagogical Chain

```
Problem Motivation  (WHY does this problem exist?)
      ↓
Knowns / Unknowns / Assumptions
      ↓
Mathematical Model  (derivation step by step)
      ↓
Algorithm / Pipeline  ← Click any stage to expand
      ↓
Live Demo  (linked to Ocean Library MATLAB experiment)
      ↓
Cross-Domain Bridges  (where does this same atom appear?)
```

### 3.1 The 3-Pillar Component Contract: $\mathcal{O} = (\phi, f, \mathcal{D})$

Every component or concept node in OceanLearn MUST fulfill three mandatory representations:

1. **Pillar 1: Symbolic Representation ($\phi$)**:
   - Math model, LaTeX equations, KaTeX rendering, and textbook references (Hamilton, Oppenheim, Cover & Thomas).
   - Exposed via `getMetadata().SymbolicLatex` in MATLAB or `"math_model"` in ConceptNode JSON.

2. **Pillar 2: Computational Representation ($f$)**:
   - High-performance numeric forward execution pass (`compute()`, `fit()`, `simulate()`).
   - Live numerical evaluation, parameter sliders, residual calculations, and error diagnostics.

3. **Pillar 3: Diagram & Architectural Specification ($\mathcal{D}$)**:
   - Self-describing port signatures (`PortSpec` / `getPipelineSpec()`) specifying input/output ports, shapes, and domain data types.
   - Enables dynamic SVG canvas rendering, cross-domain pipeline composition, and programmatic Simulink block generation (`add_block`, `simulink`).

---


## 4. Domain Coverage

| Domain | Status | Linked Experiments |
| :--- | :--- | :--- |
| Signal Processing | First (active) | exp_sampling_aliasing_visualizer.m, ... |
| Computer Vision | Active | exp_pocs_robustness_noisy.m, ... |
| Time Series | Active | exp_ar_fitting_forecast.m, ... |
| Traditional ML | Active | exp_spectral_clustering.m, ... |
| Reinforcement Learning | Active | exp_jack_car_rental.m, ... |
| Information Theory | Active | exp_hidden_cycles_spectral.m, ... |
| NLP / Language | Planned | (no experiments yet) |

---

## 5. Content Schema (ConceptNode JSON)

```json
{
  "id": "sampling-aliasing",
  "tier": 1,
  "domain": "signal_processing",
  "chapter": "Oppenheim Ch.4 §4.1-4.2",
  "title": "Sampling & Aliasing",
  "status": "active",
  "pedagogy": {
    "motivation": "WHY string",
    "problem_setup": {
      "knowns": ["list"],
      "unknowns": ["list"],
      "assumptions": ["list"]
    },
    "math_model": [
      { "step": 1, "label": "Impulse Train", "intuition": "...", "equation": "..." }
    ],
    "pipeline": [
      { "id": "sampler", "label": "Periodic Sampler", "bypassable": false }
    ]
  },
  "cross_bridges": [
    { "target_id": "image-pyramids-gaussian", "relationship": "dual_domain", "description": "..." }
  ],
  "atom_links": ["fourier_analysis"],
  "experiments": [
    { "id": "exp-sampling-aliasing", "path": "Experiments/Scripts/Signal Processing/exp_sampling_aliasing_visualizer.m" }
  ],
  "metadata": {
    "prerequisites": [],
    "difficulty": 2,
    "tags": ["sampling", "aliasing", "nyquist", "fourier"]
  }
}
```

---

## 6. Frontend (When Built)

- **Technology**: Plain HTML + Vanilla CSS + Vanilla JS (no build tools)
- **Location**: `C:\Ocean Library\Matlab code practice\OceanLearn\`
- **Math rendering**: KaTeX (self-hosted)
- **Pipeline diagrams**: SVG rendered from JSON pipeline array
- **Works offline**: open `index.html` directly as `file://`

---

## 7. Relation to the MATLAB Library

OceanLearn is a presentation layer **on top of** the Ocean Library:
- Every experiment (`exp_*.m`) links to one ConceptNode
- Every library class links to the pipeline stage it implements via `+meta/*.meta.json` sidecars
- Content is written first as MATLAB experiments → then surfaced in OceanLearn

**Build order**: MATLAB experiments first → then OceanLearn wraps them.

---

## 8. Future Workflow: Adding a Concept

```
Step 1: Implement the MATLAB experiment (exp_<slug>.m)
Step 2: Create content/<domain>/<chapter>/<slug>.json
Step 3: Register in content/registry.json
Step 4: Update .agents/architecture.md
Step 5: Update docs/guides/oceanlearn.md
Step 6: Add +meta/*.meta.json sidecars to new library classes
```
