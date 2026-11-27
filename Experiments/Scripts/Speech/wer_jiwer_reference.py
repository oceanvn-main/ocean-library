"""jiwer reference counts for validate_wer_jiwer.m.

Usage:  python wer_jiwer_reference.py <input.json> <output.json>

input:  {"sets": [{"name": str, "refs": [str], "hyps": [str]}, ...],
         "normalize": [str]}          # raw strings to push through the jiwer normalizer pipeline
output: {"jiwer_version": str,
         "sets": [{"name", "wer", "cer", "word": {S,D,I,H}, "char": {S,D,I,H},
                   "per_word": [[S,D,I,H], ...], "per_char": [[S,D,I,H], ...]}],
         "normalized": [str]}

Word level uses jiwer's default wer transform, character level jiwer's default cer
transform; nothing here is Ocean code.
"""
import json
import sys
from importlib.metadata import version

import jiwer


def counts(o):
    return {"S": o.substitutions, "D": o.deletions, "I": o.insertions, "H": o.hits}


def row(o):
    return [o.substitutions, o.deletions, o.insertions, o.hits]


def main(inp, outp):
    data = json.loads(open(inp, encoding="utf-8").read())
    sets = []
    for s in data["sets"]:
        refs, hyps = s["refs"], s["hyps"]
        w = jiwer.process_words(refs, hyps)
        c = jiwer.process_characters(refs, hyps)
        sets.append({
            "name": s["name"], "wer": w.wer, "cer": c.cer,
            "word": counts(w), "char": counts(c),
            "per_word": [row(jiwer.process_words(r, h)) for r, h in zip(refs, hyps)],
            "per_char": [row(jiwer.process_characters(r, h)) for r, h in zip(refs, hyps)],
        })
    pipe = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                          jiwer.RemoveWhiteSpace(replace_by_space=True),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    out = {"jiwer_version": version("jiwer"), "sets": sets,
           "normalized": [pipe(t) for t in data.get("normalize", [])]}
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
