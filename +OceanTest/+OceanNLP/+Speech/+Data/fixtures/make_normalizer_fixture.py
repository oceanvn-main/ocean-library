"""Generate normalizer_cases.json: reference data for TranscriptNormalizer tests.

Run from the repo root:
    ./.venv/Scripts/python.exe "+OceanTest/+OceanNLP/+Speech/+Data/fixtures/make_normalizer_fixture.py"

* punctuation: every code point in U+0000..U+00FF and U+2000..U+206F whose
  Unicode general category starts with "P" (Python unicodedata) -- the set
  TranscriptNormalizer.PunctuationRanges must reproduce exactly.
* strings: input sentences (FSDD digit words with case / punctuation / whitespace
  variants) and the output of the equivalent jiwer pipeline
  Compose([ToLowerCase(), RemovePunctuation(), RemoveWhiteSpace(replace_by_space=True),
           RemoveMultipleSpaces(), Strip()]).
"""
import json
import unicodedata
from importlib.metadata import version
from pathlib import Path

import jiwer

BLOCKS = [(0x0000, 0x00FF), (0x2000, 0x206F)]

INPUTS = [
    "Zero, one, two.",
    "THREE -- four; five!",
    "  six\tseven\n eight  ",
    "Nine? ¿Nine?",
    "o'clock & twenty-one",
    "“seven” — eight…",
    "ÉCOLE café § 4",
    "a+b=c $5 #tag @user",
    "[one] (two) {three}",
    "",
    "   ",
]


def main():
    punct = [c for lo, hi in BLOCKS for c in range(lo, hi + 1)
             if unicodedata.category(chr(c)).startswith("P")]
    pipe = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                          jiwer.RemoveWhiteSpace(replace_by_space=True),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    strings = [{"input": s, "jiwer": pipe(s)} for s in INPUTS]
    out = {"generator": "make_normalizer_fixture.py", "jiwer_version": version("jiwer"),
           "unicode_version": unicodedata.unidata_version,
           "blocks": BLOCKS, "punctuation": punct, "strings": strings}
    path = Path(__file__).with_name("normalizer_cases.json")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(punct)} punctuation code points, {len(strings)} strings)")


if __name__ == "__main__":
    main()
