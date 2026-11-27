---
trigger: always_on
---

# STRICT RULE: Math Expression Formatting in User Responses

**NEVER USE LATEX FORMATTING IN CHAT MESSAGES TO THE USER.**

1. **NO DOLLAR SIGNS**: Never use single dollar signs (`$`) or double dollar signs (`$$`) anywhere in chat responses. The UI chat interface does not render LaTeX and will display raw unrendered markup.
2. **NO LATEX MACROS**: Never output LaTeX macros such as `\frac`, `\text`, `\sum`, `\int`, `\Omega`, `\pi`, `\cdot`, `\approx`, `\equiv`, or `\mathbf`.
3. **USE PLAIN TEXT / UNICODE**:
   - Use standard Unicode symbols: Ω, π, ∑, ∫, ±, ≤, ≥, →, Δ, ≡, ≈, ∞, ×, ÷.
   - Use fraction notation in plain text: `(1 / T)` or `(fs_out / fs_in)`.
   - Use plain text superscripts/subscripts or explicit code variables: `fs`, `f_max`, `x_c(t)`, `H(z)`.
4. **USE CODE BLOCKS & TABLES FOR COMPLEX DERIVATIONS**:
   - For multi-line math derivations, use fenced `text` or `matlab` code blocks.
   - For comparing values or formulas, use standard Markdown tables.