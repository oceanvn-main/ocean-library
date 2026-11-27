"""Generate jiwer_cases.json: jiwer 4.x reference counts for Ocean WER/CER tests.

Run from the repo root:
    ./.venv/Scripts/python.exe "+OceanTest/+OceanNLP/+Speech/+Evaluation/fixtures/make_jiwer_fixture.py"

Inputs are real strings: reference word sequences are built from the FSDD
recording labels (digit in the file name -> English word); hypotheses are seeded
random edits of them (substitute / delete / insert digit words), plus a block of
case / punctuation / whitespace / Unicode variants. Every number in the JSON is
computed by jiwer (process_words / process_characters with their default
transforms) -- nothing is typed in by hand.
"""
import json
import random
import re
from pathlib import Path

import jiwer

REPO = Path(__file__).resolve().parents[5]
REC = REPO / "Data" / "Datasets" / "Speech" / "FSDD" / "recordings"
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def fsdd_labels():
    labels = []
    for f in sorted(REC.glob("*.wav")):
        m = re.match(r"^(\d)_([A-Za-z]+)_(\d+)\.wav$", f.name)
        if m:
            labels.append((m.group(2), WORDS[int(m.group(1))]))
    if len(labels) != 2000:
        raise SystemExit(f"expected 2000 FSDD files, found {len(labels)}")
    return labels


def perturb(words, rng):
    out = list(words)
    for _ in range(rng.randint(0, 3)):
        op = rng.choice("SDI")
        if op == "S" and out:
            out[rng.randrange(len(out))] = rng.choice(WORDS)
        elif op == "D" and out:
            del out[rng.randrange(len(out))]
        elif op == "I":
            out.insert(rng.randrange(len(out) + 1), rng.choice(WORDS))
    return out


def counts(o):
    return {"S": o.substitutions, "D": o.deletions, "I": o.insertions, "H": o.hits}


def main():
    rng = random.Random(20261006)
    labels = fsdd_labels()
    by_speaker = {}
    for spk, w in labels:
        by_speaker.setdefault(spk, []).append(w)

    pairs = []
    for spk in sorted(by_speaker):
        pool = by_speaker[spk]
        for _ in range(60):
            ref = rng.sample(pool, rng.randint(1, 6))
            hyp = perturb(ref, rng)
            pairs.append((" ".join(ref), " ".join(hyp), "fsdd-" + spk))

    variants = [
        ("Seven, eight -- NINE!", "seven eight nine", "punct"),
        ("one two three", "One two, three.", "punct"),
        ("  four   five  ", "four five", "spaces"),
        ("four five", "four  five", "spaces"),
        ("six\tseven", "six seven", "tab"),
        ("zero one", "", "empty-hyp"),
        ("", "two", "empty-ref"),
        ("", "", "empty-both"),
        ("café naïve", "cafe naive", "latin1"),
        ("naïve café", "naïve café", "latin1"),
        ("eight \U0001F600 nine", "eight nine", "astral"),
        ("aaaa bbbb", "bbbb aaaa", "swap"),
        ("one one one one", "one one", "repeat"),
        ("two", "two two two two", "repeat"),
        ("three four five six", "five six three four", "rotate"),
    ]
    pairs += variants

    cases = []
    for ref, hyp, tag in pairs:
        w = jiwer.process_words(ref, hyp)
        c = jiwer.process_characters(ref, hyp)
        cases.append({"ref": ref, "hyp": hyp, "tag": tag, "word": counts(w), "char": counts(c)})

    # corpora: each FSDD speaker block, everything with a non-empty total reference
    corpora = []
    blocks = {}
    for ref, hyp, tag in pairs:
        blocks.setdefault(tag if tag.startswith("fsdd-") else "variants", []).append((ref, hyp))
    blocks["all"] = [(r, h) for r, h, _ in pairs]
    for name, items in sorted(blocks.items()):
        refs = [r for r, _ in items]
        hyps = [h for _, h in items]
        w = jiwer.process_words(refs, hyps)
        c = jiwer.process_characters(refs, hyps)
        corpora.append({"name": name, "refs": refs, "hyps": hyps,
                        "wer": w.wer, "cer": c.cer, "word": counts(w), "char": counts(c)})

    # jiwer behaviour when every reference is empty (Ocean deliberately errors instead)
    e = jiwer.process_words(["", " "], ["one", ""])
    empty_total = {"refs": ["", " "], "hyps": ["one", ""], "wer": e.wer, "word": counts(e)}

    out = {
        "generator": "make_jiwer_fixture.py",
        "jiwer_version": getattr(jiwer, "__version__", None) or __import__("importlib.metadata").metadata.version("jiwer"),
        "seed": 20261006,
        "cases": cases,
        "corpora": corpora,
        "empty_total_reference": empty_total,
    }
    path = Path(__file__).with_name("jiwer_cases.json")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases, {len(corpora)} corpora)")


if __name__ == "__main__":
    main()
