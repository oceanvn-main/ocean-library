# -*- coding: utf-8 -*-
"""
exp_tokenizer_comparison.py
===========================
Experiment: Compare popular tokenization algorithms on a shared corpus.
Objectives:
  1. Run the same corpus through five tokenizer families:
       - Whitespace (word-level baseline)
       - Character (character-level baseline)
       - BPE  (Byte-Pair Encoding — Sennrich et al. 2016 / GPT-2)
       - WordPiece (Schuster & Nakamura 2012 / BERT)
       - Unigram  (Kudo 2018 / SentencePiece)
  2. Fit each sub-word tokenizer from scratch on the training corpus
     with a fixed target vocabulary size so results are comparable.
  3. Measure and compare:
       - Vocabulary size (actual vs target)
       - Mean tokens per sentence
       - Token-length distribution (histogram)
       - Fertility (tokens / words for sub-word methods)
       - Out-of-vocabulary (OOV) rate on a held-out test set
       - Compression ratio (chars / token)
  4. Visualise all metrics in a single matplotlib diagnostic dashboard.
Assumptions:
  - All tokenizers are trained ONLY on the training sentences.
  - Test set information is NEVER used during training (anti-leakage).
  - Vocabulary sizes are set identically for all sub-word methods so that
    comparisons are fair.
  - The corpus is embedded in this script so the experiment is fully
    self-contained and reproducible without external data files.
  - rng is seeded for any stochastic component (Unigram EM sampling).
Dependencies:
  pip install tokenizers sentencepiece matplotlib numpy rich
  (tokenizers  = HuggingFace tokenizers library — provides BPE & WordPiece)
  (sentencepiece = Google SentencePiece — provides Unigram LM)
Author: Ocean Library   Date: 2026-09-30
"""
# ---------------------------------------------------------------------------
# 0. Standard library + third-party imports
# ---------------------------------------------------------------------------
import io
import os
import sys
import json
import random
import tempfile
import textwrap
import warnings
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Force UTF-8 on stdout so Unicode symbols work on Windows (cp1252 terminals).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
# Rich is used for a readable console report; gracefully degrade if absent.
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
# HuggingFace tokenizers library (BPE + WordPiece)
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordPiece
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace as HFWhitespace
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARNING] 'tokenizers' not installed. BPE and WordPiece will be skipped.")
    print("          Install with:  pip install tokenizers")
# SentencePiece (Unigram LM)
try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False
    print("[WARNING] 'sentencepiece' not installed. Unigram will be skipped.")
    print("          Install with:  pip install sentencepiece")
# ---------------------------------------------------------------------------
# 1. Experiment Configuration
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
TARGET_VOCAB_SIZE = 500   # Identical target for BPE / WordPiece / Unigram
TRAIN_RATIO      = 0.80
SPECIAL_TOKENS   = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
PLOT_STYLE = {
    "bg":    "#0f1117",
    "panel": "#1a1d27",
    "text":  "#e8eaf6",
    "grid":  "#2a2d3a",
    "c0":    "#7c83fd",   # BPE
    "c1":    "#fd7c7c",   # WordPiece
    "c2":    "#7cfd9a",   # Unigram
    "c3":    "#fdd97c",   # Whitespace
    "c4":    "#c47cfd",   # Character
}
# ---------------------------------------------------------------------------
# 2. Embedded Corpus
# ---------------------------------------------------------------------------
# A compact but diverse English corpus that exercises:
#   - common words, rare words, punctuation, numbers, mixed-case
# Using only this script guarantees full reproducibility.
RAW_CORPUS = textwrap.dedent("""\\
    The quick brown fox jumps over the lazy dog.
    Natural language processing enables computers to understand human text.
    Byte-pair encoding is a data compression technique adapted for tokenization.
    WordPiece was introduced for the Japanese and Korean Voice Search.
    SentencePiece treats text as a sequence of Unicode characters.
    The Transformer architecture revolutionised deep learning in NLP.
    BERT uses WordPiece subword tokenization with a 30,000-token vocabulary.
    GPT-2 uses byte-level BPE, which can encode any Unicode text without unknowns.
    Unigram language-model tokenization is based on maximum entropy principles.
    Tokenization affects downstream model quality, training speed, and memory.
    Rare words like "supercalifragilisticexpialidocious" are hard to handle.
    Numbers, dates (2024-01-15), and codes (ISO 3166-1) appear frequently.
    Mixed-case text: iPhone, GPT-4, NASA, BERT, RoBERTa, XLNet, AlBERT.
    Punctuation handling: "Hello, world!" said the NLP engineer.
    Character-level models treat every character as a single token unit.
    Whitespace tokenizers simply split on spaces and newlines.
    Sub-word tokenization balances vocabulary size and sequence length.
    A larger vocabulary means fewer tokens per sentence (higher compression).
    A smaller vocabulary means more tokens but rarer out-of-vocabulary events.
    Fertility measures how many sub-word pieces one word is split into on average.
    High fertility indicates the tokenizer fragments words excessively.
    The OOV rate measures how often unknown tokens appear on unseen text.
    Byte-level BPE has zero OOV rate because it covers all 256 byte values.
    Compression ratio is defined as characters divided by the number of tokens.
    Scientific text often contains domain-specific vocabulary and symbols.
    Mathematical notation: f(x) = σ(Wx + b) where σ is the sigmoid function.
    Code snippets: import numpy as np; arr = np.array([1, 2, 3], dtype=float).
    Multilingual models require tokenizers that handle diverse Unicode scripts.
    Tokenization is the first and often most critical step in any NLP pipeline.
    Pre-tokenizers determine where word boundaries are before sub-word splitting.
    The choice of tokenizer has lasting consequences for every downstream task.
    Language models learn statistical regularities from enormous text corpora.
    Attention is all you need — the famous Transformer paper title.
    Fine-tuning pre-trained models dramatically reduces the required training data.
    Sequence-to-sequence models require careful handling of special delimiter tokens.
    Named entities: New York City, Barack Obama, Pacific Ocean, Eiffel Tower.
    Idiomatic expressions are challenging for statistical language models.
    Morphologically rich languages benefit most from sub-word tokenization schemes.
    Finnish and Turkish have complex agglutinative morphology.
    Stemming reduces words to their root form but loses grammatical information.
    Lemmatization maps inflected forms to their dictionary lemma.
    Stop words like the, a, and, in are very frequent but carry little meaning.
    TF-IDF weighting down-weights frequent terms and up-weights rare ones.
    Word embeddings like Word2Vec map tokens to dense real-valued vectors.
    Positional encodings allow Transformers to use sequence order information.
    Self-attention computes pairwise interactions between all token positions.
    The cross-entropy loss is commonly used for language modelling objectives.
    Perplexity measures how well a probability model predicts a sample.
    Zero-shot learning evaluates model generalisation to unseen task categories.
    Few-shot prompting demonstrates the remarkable in-context learning ability.
    Chain-of-thought prompting improves reasoning on multi-step problems.
    Retrieval-augmented generation combines parametric and non-parametric memory.
""").strip().split("\n")
# ---------------------------------------------------------------------------
# 3. Train / Test Split (chronological — no shuffling to respect order)
# ---------------------------------------------------------------------------
n_train = int(len(RAW_CORPUS) * TRAIN_RATIO)
TRAIN_SENTENCES: List[str] = RAW_CORPUS[:n_train]
TEST_SENTENCES:  List[str] = RAW_CORPUS[n_train:]
print(f"\n{'='*70}")
print("  TOKENIZER COMPARISON EXPERIMENT")
print(f"{'='*70}")
print(f"  Corpus     : {len(RAW_CORPUS)} sentences total")
print(f"  Train split: {len(TRAIN_SENTENCES)} sentences (first {TRAIN_RATIO*100:.0f}%)")
print(f"  Test split : {len(TEST_SENTENCES)} sentences (last {(1-TRAIN_RATIO)*100:.0f}%)")
print(f"  Target vocab size (sub-word): {TARGET_VOCAB_SIZE}")
print(f"  Seed: {SEED}")
print(f"{'='*70}\n")
# ---------------------------------------------------------------------------
# 4. Tokenizer Implementations
# ---------------------------------------------------------------------------
@dataclass
class TokenizerResult:
    """Container for one tokenizer's experiment outputs."""
    name:             str
    color:            str
    train_tokens:     List[List[str]] = field(default_factory=list)  # [[tok, ...], ...]
    test_tokens:      List[List[str]] = field(default_factory=list)
    vocab:            List[str]       = field(default_factory=list)   # full vocabulary
    oov_count:        int             = 0
    total_test_tokens: int            = 0
    # ----- Derived metrics (filled by compute_metrics) -----
    vocab_size:       int   = 0
    mean_tokens_train: float = 0.0
    mean_tokens_test:  float = 0.0
    fertility:        float = 0.0   # mean sub-word pieces per word
    oov_rate:         float = 0.0
    compression_train: float = 0.0  # chars per token
    tokenize_fn: Optional[Callable[[str], List[str]]] = field(default=None, repr=False, compare=False)
    def tokenize(self, text: str) -> List[str]:
        """Tokenize arbitrary user-supplied text with this trained tokenizer."""
        if self.tokenize_fn is None:
            return []
        return self.tokenize_fn(text)
    def compute_metrics(self):
        self.vocab_size        = len(self.vocab)
        self.mean_tokens_train = float(np.mean([len(t) for t in self.train_tokens])) if self.train_tokens else 0.0
        self.mean_tokens_test  = float(np.mean([len(t) for t in self.test_tokens]))  if self.test_tokens  else 0.0
        self.total_test_tokens = sum(len(t) for t in self.test_tokens)
        # Fertility: tokens / whitespace words (only for sub-word methods)
        all_train_words = sum(len(s.split()) for s in TRAIN_SENTENCES)
        all_train_toks  = sum(len(t) for t in self.train_tokens)
        self.fertility  = all_train_toks / max(all_train_words, 1)
        # OOV rate on test
        self.oov_rate = self.oov_count / max(self.total_test_tokens, 1) * 100.0
        # Compression: total chars / total tokens on train
        total_chars  = sum(len(s) for s in TRAIN_SENTENCES)
        self.compression_train = total_chars / max(all_train_toks, 1)
# ---- 4a. Whitespace (word-level baseline) ---------------------------------
# Keep punctuation as tokens instead of deleting it.
# Examples: "hi!" -> ["hi", "!"], "wait..." -> ["wait", "..."],
#           "well--maybe" -> ["well", "--", "maybe"].
WORD_PUNCT_PATTERN = re.compile(
    r"\.\.\.|--|[\w]+(?:['’][\w]+)?|[^\w\s]",
    flags=re.UNICODE,
)

class WhitespaceTokenizerWrapper:
    name = "Whitespace"
    def tokenize(self, sentence: str) -> List[str]:
        return WORD_PUNCT_PATTERN.findall(sentence)
def run_whitespace() -> TokenizerResult:
    tok = WhitespaceTokenizerWrapper()
    vocab_set = set()
    train_toks, test_toks = [], []
    for s in TRAIN_SENTENCES:
        t = tok.tokenize(s)
        train_toks.append(t)
        vocab_set.update(t)
    oov = 0
    for s in TEST_SENTENCES:
        t = tok.tokenize(s)
        test_toks.append(t)
        oov += sum(1 for w in t if w not in vocab_set)
    res = TokenizerResult(name="Whitespace", color=PLOT_STYLE["c3"],
                          train_tokens=train_toks, test_tokens=test_toks,
                          vocab=sorted(vocab_set), oov_count=oov,
                          tokenize_fn=tok.tokenize)
    res.compute_metrics()
    return res
# ---- 4b. Character-level baseline -----------------------------------------
class CharTokenizerWrapper:
    name = "Character"
    def tokenize(self, sentence: str) -> List[str]:
        return list(sentence)   # every character is a token
def run_character() -> TokenizerResult:
    tok = CharTokenizerWrapper()
    vocab_set = set()
    train_toks, test_toks = [], []
    for s in TRAIN_SENTENCES:
        t = tok.tokenize(s)
        train_toks.append(t)
        vocab_set.update(t)
    oov = 0
    for s in TEST_SENTENCES:
        t = tok.tokenize(s)
        test_toks.append(t)
        oov += sum(1 for c in t if c not in vocab_set)
    res = TokenizerResult(name="Character", color=PLOT_STYLE["c4"],
                          train_tokens=train_toks, test_tokens=test_toks,
                          vocab=sorted(vocab_set), oov_count=oov,
                          tokenize_fn=tok.tokenize)
    res.compute_metrics()
    return res
# ---- 4c. BPE via HuggingFace tokenizers -----------------------------------
def run_bpe() -> TokenizerResult:
    if not HF_AVAILABLE:
        print("  [SKIP] BPE — 'tokenizers' library not installed.")
        return None
    model = BPE(unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = HFWhitespace()
    trainer = BpeTrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(TRAIN_SENTENCES, trainer=trainer)
    vocab = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    vocab_set = set(vocab)
    train_toks = [tokenizer.encode(s).tokens for s in TRAIN_SENTENCES]
    test_enc   = [tokenizer.encode(s) for s in TEST_SENTENCES]
    test_toks  = [e.tokens for e in test_enc]
    oov = sum(1 for e in test_enc for tok in e.tokens if tok == "[UNK]")
    res = TokenizerResult(name="BPE", color=PLOT_STYLE["c0"],
                          train_tokens=train_toks, test_tokens=test_toks,
                          vocab=vocab, oov_count=oov,
                          tokenize_fn=lambda text, _tok=tokenizer: _tok.encode(text).tokens)
    res.compute_metrics()
    return res
# ---- 4d. WordPiece via HuggingFace tokenizers -----------------------------
def run_wordpiece() -> TokenizerResult:
    if not HF_AVAILABLE:
        print("  [SKIP] WordPiece — 'tokenizers' library not installed.")
        return None
    model = WordPiece(unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = HFWhitespace()
    trainer = WordPieceTrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(TRAIN_SENTENCES, trainer=trainer)
    vocab = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    train_toks = [tokenizer.encode(s).tokens for s in TRAIN_SENTENCES]
    test_enc   = [tokenizer.encode(s) for s in TEST_SENTENCES]
    test_toks  = [e.tokens for e in test_enc]
    oov = sum(1 for e in test_enc for tok in e.tokens if tok == "[UNK]")
    res = TokenizerResult(name="WordPiece", color=PLOT_STYLE["c1"],
                          train_tokens=train_toks, test_tokens=test_toks,
                          vocab=vocab, oov_count=oov,
                          tokenize_fn=lambda text, _tok=tokenizer: _tok.encode(text).tokens)
    res.compute_metrics()
    return res
# ---- 4e. Unigram LM via SentencePiece -------------------------------------
def run_unigram() -> TokenizerResult:
    if not SPM_AVAILABLE:
        print("  [SKIP] Unigram — 'sentencepiece' library not installed.")
        return None
    import re as _re
    # SentencePiece requires a file or string input; use a NamedTemporaryFile.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write("\n".join(TRAIN_SENTENCES))
        corpus_path = f.name
    model_prefix = os.path.join(tempfile.gettempdir(), "ocean_spm_unigram")
    def _train(vocab_size: int):
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=vocab_size,
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            user_defined_symbols=",".join(["[MASK]", "[CLS]", "[SEP]"]),
        )
    try:
        _train(TARGET_VOCAB_SIZE)
    except Exception as e:
        # Corpus too small: SentencePiece reports the actual maximum.
        # Example message: "Vocabulary size too high (500). Please set it to a value <= 432."
        m = _re.search(r'<= (\d+)', str(e))
        if m:
            capped = int(m.group(1))
            print(f"\n    [Unigram] Corpus too small for vocab={TARGET_VOCAB_SIZE}; "
                  f"retrying with vocab={capped}...", end="", flush=True)
            try:
                _train(capped)
            except Exception as e2:
                print(f"  [SKIP] Unigram retry failed: {e2}")
                if os.path.exists(corpus_path): os.unlink(corpus_path)
                return None
        else:
            print(f"  [SKIP] Unigram training failed: {e}")
            if os.path.exists(corpus_path): os.unlink(corpus_path)
            return None
    finally:
        if os.path.exists(corpus_path):
            os.unlink(corpus_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_prefix + ".model")
    # Gather vocabulary
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    train_toks = [sp.encode(s, out_type=str) for s in TRAIN_SENTENCES]
    test_toks  = [sp.encode(s, out_type=str) for s in TEST_SENTENCES]
    # SentencePiece maps unknowns to the <unk> piece
    unk_piece = sp.id_to_piece(sp.unk_id())
    oov = sum(1 for tl in test_toks for t in tl if t == unk_piece)
    res = TokenizerResult(name="Unigram", color=PLOT_STYLE["c2"],
                          train_tokens=train_toks, test_tokens=test_toks,
                          vocab=vocab, oov_count=oov,
                          tokenize_fn=lambda text, _sp=sp: _sp.encode(text, out_type=str))
    res.compute_metrics()
    # Clean up model files
    for ext in [".model", ".vocab"]:
        p = model_prefix + ext
        if os.path.exists(p):
            os.unlink(p)
    return res
# ---------------------------------------------------------------------------
# 5. Run All Tokenizers
# ---------------------------------------------------------------------------
def run_all() -> Dict[str, TokenizerResult]:
    results = {}
    runners = [
        ("Whitespace", run_whitespace),
        ("Character",  run_character),
        ("BPE",        run_bpe),
        ("WordPiece",  run_wordpiece),
        ("Unigram",    run_unigram),
    ]
    for name, fn in runners:
        print(f"  Training {name} tokenizer...", end="", flush=True)
        try:
            r = fn()
            if r is not None:
                results[name] = r
                print(f" ✓  vocab={r.vocab_size}, "
                      f"mean_tok/sent={r.mean_tokens_train:.1f}, "
                      f"fertility={r.fertility:.2f}, "
                      f"OOV={r.oov_rate:.1f}%, "
                      f"compression={r.compression_train:.2f} chars/tok")
            else:
                print(" [skipped]")
        except Exception as exc:
            print(f" [ERROR: {exc}]")
    return results
print("[Step 1] Fitting tokenizers on training corpus...\n")
results = run_all()
print(f"\n  {len(results)} tokenizer(s) ready.\n")
# ---------------------------------------------------------------------------
# 6. Console Report
# ---------------------------------------------------------------------------
def print_report(results: Dict[str, TokenizerResult]):
    print(f"\n{'='*70}")
    print("  EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")
    header = ["Tokenizer", "Vocab", "Tok/Sent (train)", "Tok/Sent (test)",
              "Fertility", "OOV %", "Chars/Tok"]
    rows = []
    for r in results.values():
        rows.append([
            r.name,
            str(r.vocab_size),
            f"{r.mean_tokens_train:.2f}",
            f"{r.mean_tokens_test:.2f}",
            f"{r.fertility:.3f}",
            f"{r.oov_rate:.2f}",
            f"{r.compression_train:.2f}",
        ])
    if RICH_AVAILABLE:
        table = Table(title="Tokenizer Comparison", box=box.ROUNDED, style="dim")
        for h in header:
            table.add_column(h, justify="right" if h != "Tokenizer" else "left")
        for row in rows:
            table.add_row(*row)
        console.print(table)
    else:
        col_w = [max(len(header[i]), max((len(r[i]) for r in rows), default=0)) + 2
                 for i in range(len(header))]
        sep = "+" + "+".join("-" * w for w in col_w) + "+"
        def fmt_row(r):
            return "|" + "|".join(f" {cell:{col_w[i]-1}}" for i, cell in enumerate(r)) + "|"
        print(sep)
        print(fmt_row(header))
        print(sep)
        for row in rows:
            print(fmt_row(row))
        print(sep)
    # Print sample tokenization for each method
    sample_sentence = "Natural language processing enables computers to understand human text."
    print(f"\n  Sample tokenization of:\n  >>> {sample_sentence!r}\n")
    for r in results.values():
        sample = r.tokenize(sample_sentence)
        tok_str = ", ".join(json.dumps(t, ensure_ascii=False) for t in sample[:30])
        suffix = ", ..." if len(sample) > 30 else ""
        print(f"  [{r.name:12s}] ({len(sample)} tokens): {tok_str}{suffix}")

print_report(results)
# ---------------------------------------------------------------------------
# 7. Interactive Tokenization
# ---------------------------------------------------------------------------

def interactive_tokenization(results: Dict[str, TokenizerResult]):
    """Read arbitrary strings from the keyboard and print each tokenizer's tokens."""
    if not results:
        return

    print(f"\n{'='*70}")
    print("  INTERACTIVE TOKENIZATION")
    print(f"{'='*70}")
    print("  Type any text, including punctuation such as !, ..., --, ?, commas, etc.")
    print("  Type 'exit' or 'quit' to finish.\n")

    while True:
        try:
            user_text = input("  Text > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.strip().lower() in {"exit", "quit"}:
            break
        if user_text == "":
            continue

        print()
        for r in results.values():
            tokens = r.tokenize(user_text)
            rendered = ", ".join(json.dumps(tok, ensure_ascii=False) for tok in tokens)
            print(f"  [{r.name:12s}] {rendered}")
        print()

interactive_tokenization(results)

# ---------------------------------------------------------------------------
# 8. Visualization Dashboard
# ---------------------------------------------------------------------------
def plot_dashboard(results: Dict[str, TokenizerResult]):
    if not results:
        print("\n  [SKIP] No results to plot.")
        return
    matplotlib.rcParams.update({
        "figure.facecolor":  PLOT_STYLE["bg"],
        "axes.facecolor":    PLOT_STYLE["panel"],
        "axes.edgecolor":    PLOT_STYLE["grid"],
        "axes.labelcolor":   PLOT_STYLE["text"],
        "xtick.color":       PLOT_STYLE["text"],
        "ytick.color":       PLOT_STYLE["text"],
        "text.color":        PLOT_STYLE["text"],
        "grid.color":        PLOT_STYLE["grid"],
        "font.family":       "sans-serif",
        "axes.grid":         True,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
    })
    names  = list(results.keys())
    colors = [results[n].color for n in names]
    rs     = [results[n] for n in names]
    fig = plt.figure(figsize=(18, 13), facecolor=PLOT_STYLE["bg"])
    fig.suptitle("Tokenizer Algorithm Comparison Dashboard",
                 color=PLOT_STYLE["text"], fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)
    def style_ax(ax, title):
        ax.set_title(title, color=PLOT_STYLE["text"], fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors=PLOT_STYLE["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_STYLE["grid"])
    # --- Plot A: Vocabulary Size ---
    ax_vocab = fig.add_subplot(gs[0, 0])
    bars = ax_vocab.bar(names, [r.vocab_size for r in rs], color=colors, edgecolor="none", alpha=0.88)
    for bar, r in zip(bars, rs):
        ax_vocab.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                      str(r.vocab_size), ha="center", va="bottom", fontsize=8,
                      color=PLOT_STYLE["text"])
    ax_vocab.axhline(TARGET_VOCAB_SIZE, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.5,
                     label=f"Target = {TARGET_VOCAB_SIZE}")
    ax_vocab.legend(fontsize=7, labelcolor=PLOT_STYLE["text"], facecolor=PLOT_STYLE["panel"],
                    edgecolor="none")
    style_ax(ax_vocab, "A — Vocabulary Size")
    ax_vocab.set_ylabel("# Unique Tokens", fontsize=8)
    # --- Plot B: Mean Tokens per Sentence (Train vs Test) ---
    ax_toklen = fig.add_subplot(gs[0, 1])
    x = np.arange(len(names))
    w = 0.35
    ax_toklen.bar(x - w/2, [r.mean_tokens_train for r in rs], w, color=colors, alpha=0.88, label="Train")
    ax_toklen.bar(x + w/2, [r.mean_tokens_test  for r in rs], w, color=colors, alpha=0.50, label="Test",
                  edgecolor=[c for c in colors], linewidth=1.2)
    ax_toklen.set_xticks(x); ax_toklen.set_xticklabels(names, fontsize=8)
    ax_toklen.legend(fontsize=7, labelcolor=PLOT_STYLE["text"], facecolor=PLOT_STYLE["panel"],
                     edgecolor="none")
    style_ax(ax_toklen, "B — Mean Tokens / Sentence")
    ax_toklen.set_ylabel("Tokens", fontsize=8)
    # --- Plot C: Fertility ---
    ax_fert = fig.add_subplot(gs[0, 2])
    bars_f = ax_fert.bar(names, [r.fertility for r in rs], color=colors, alpha=0.88, edgecolor="none")
    for bar, r in zip(bars_f, rs):
        ax_fert.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{r.fertility:.2f}", ha="center", va="bottom", fontsize=8,
                     color=PLOT_STYLE["text"])
    ax_fert.axhline(1.0, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.5, label="Word-level (1.0)")
    ax_fert.legend(fontsize=7, labelcolor=PLOT_STYLE["text"], facecolor=PLOT_STYLE["panel"], edgecolor="none")
    style_ax(ax_fert, "C — Fertility (tokens / word)")
    ax_fert.set_ylabel("Fertility", fontsize=8)
    # --- Plot D: OOV Rate ---
    ax_oov = fig.add_subplot(gs[1, 0])
    bars_o = ax_oov.bar(names, [r.oov_rate for r in rs], color=colors, alpha=0.88, edgecolor="none")
    for bar, r in zip(bars_o, rs):
        ax_oov.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{r.oov_rate:.1f}%", ha="center", va="bottom", fontsize=8,
                    color=PLOT_STYLE["text"])
    style_ax(ax_oov, "D — OOV Rate on Test Set (%)")
    ax_oov.set_ylabel("OOV %", fontsize=8)
    # --- Plot E: Compression Ratio ---
    ax_comp = fig.add_subplot(gs[1, 1])
    bars_c = ax_comp.bar(names, [r.compression_train for r in rs], color=colors, alpha=0.88, edgecolor="none")
    for bar, r in zip(bars_c, rs):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{r.compression_train:.2f}", ha="center", va="bottom", fontsize=8,
                     color=PLOT_STYLE["text"])
    style_ax(ax_comp, "E — Compression (chars / token)")
    ax_comp.set_ylabel("Chars per Token", fontsize=8)
    # --- Plot F: Token-Length Distribution (histogram, all methods) ---
    ax_hist = fig.add_subplot(gs[1, 2])
    all_lens = {r.name: [len(tok) for tl in r.train_tokens for tok in tl] for r in rs}
    max_len  = max((max(v) for v in all_lens.values() if v), default=20)
    bins = np.arange(0, min(max_len + 2, 25)) + 0.5
    for r in rs:
        lens = all_lens[r.name]
        ax_hist.hist(lens, bins=bins, color=r.color, alpha=0.5, label=r.name,
                     density=True, histtype="stepfilled", edgecolor=r.color, linewidth=0.8)
    ax_hist.legend(fontsize=7, labelcolor=PLOT_STYLE["text"], facecolor=PLOT_STYLE["panel"], edgecolor="none")
    style_ax(ax_hist, "F — Token-Length Distribution (train)")
    ax_hist.set_xlabel("Token length (chars)", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    # --- Plot G: Top-20 Most Frequent Train Tokens (sub-word methods only) ---
    ax_top = fig.add_subplot(gs[2, :2])
    subword = [r for r in rs if r.name not in ("Whitespace", "Character")]
    if subword:
        n_methods = len(subword)
        group_w   = 0.75
        bar_w     = group_w / n_methods
        # Use the first sub-word method's top-20 tokens as shared x-axis labels.
        ref_freq  = Counter(tok for tl in subword[0].train_tokens for tok in tl)
        ref_top20 = ref_freq.most_common(20)
        ref_labels = [t for t, _ in ref_top20]
        n_show = len(ref_labels)  # may be < 20 on tiny corpora
        for mi, r in enumerate(subword):
            freq  = Counter(tok for tl in r.train_tokens for tok in tl)
            # Align counts to ref_labels order (0 if label absent in this tokenizer)
            counts = [freq.get(lbl, 0) for lbl in ref_labels]
            xs = np.arange(n_show) + mi * bar_w - group_w / 2
            ax_top.bar(xs, counts, bar_w * 0.92, color=r.color, alpha=0.85,
                       edgecolor="none", label=r.name)
        ax_top.set_xticks(np.arange(n_show))
        ax_top.set_xticklabels(ref_labels, rotation=60, ha="right", fontsize=7)
        ax_top.legend(fontsize=7, labelcolor=PLOT_STYLE["text"],
                      facecolor=PLOT_STYLE["panel"], edgecolor="none")
    style_ax(ax_top, "G — Top-20 Most Frequent Tokens (sub-word methods, train)")
    ax_top.set_ylabel("Count", fontsize=8)
    # --- Plot H: Cumulative Vocabulary Coverage Curve ---
    ax_cov = fig.add_subplot(gs[2, 2])
    for r in rs:
        freq = Counter(tok for tl in r.train_tokens for tok in tl)
        total = sum(freq.values())
        if total == 0:
            continue
        sorted_counts = sorted(freq.values(), reverse=True)
        cumulative = np.cumsum(sorted_counts) / total
        vocab_pct   = np.arange(1, len(cumulative) + 1) / len(freq) * 100
        ax_cov.plot(vocab_pct, cumulative * 100, color=r.color, linewidth=1.5,
                    label=r.name, alpha=0.88)
    ax_cov.axhline(95, color="#ffffff", linestyle="--", linewidth=0.7, alpha=0.5, label="95% coverage")
    ax_cov.set_xlim(0, 40)
    ax_cov.set_ylim(0, 102)
    ax_cov.legend(fontsize=7, labelcolor=PLOT_STYLE["text"],
                  facecolor=PLOT_STYLE["panel"], edgecolor="none")
    style_ax(ax_cov, "H — Vocabulary Coverage Curve")
    ax_cov.set_xlabel("Vocab used (%)", fontsize=8)
    ax_cov.set_ylabel("Token coverage (%)", fontsize=8)
    plt.savefig("tokenizer_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=PLOT_STYLE["bg"])
    print("\n  Dashboard saved -> tokenizer_comparison.png")
    plt.show()
print("\n[Step 3] Generating diagnostic dashboard...")
plot_dashboard(results)
print(f"\n{'='*70}")
print("  EXPERIMENT COMPLETE")
print(f"{'='*70}\n")
