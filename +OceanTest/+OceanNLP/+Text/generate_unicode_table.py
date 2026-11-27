"""Generates +OceanNLP/+Text/unicode_category_table.mat: the Unicode general category of every code point
(run-length encoded), from Python's unicodedata (Unicode 15.0 for Python 3.12).

Why: MATLAB's bundled JVM is Java 8 (Unicode 6.2). Java's Character.getType misclassifies ~35,000 letters,
~700 digits and ~200 punctuation marks added since (Adlam, Georgian Mtavruli, CJK extensions C-G, ...), which made
the GPT-2 / BERT pre-tokenizers disagree with their references on newer scripts. Categories therefore come from this
table; only normalization (NFD/NFKC) still uses java.text.Normalizer.

Category codes follow java.lang.Character numbering: Lu1 Ll2 Lt3 Lm4 Lo5 Mn6 Me7 Mc8 Nd9 Nl10 No11 Zs12 Zl13 Zp14
Cc15 Cf16 Co18 Cs19 Pd20 Ps21 Pe22 Pc23 Po24 Sm25 Sc26 Sk27 So28 Pi29 Pf30, unassigned (Cn) 0.

Run:  py generate_unicode_table.py   (writes next to +OceanNLP/+Text/Unicode.m)
"""
import os
import unicodedata

import numpy as np
from scipy.io import savemat

CODES = {"Lu": 1, "Ll": 2, "Lt": 3, "Lm": 4, "Lo": 5, "Mn": 6, "Me": 7, "Mc": 8, "Nd": 9, "Nl": 10, "No": 11,
         "Zs": 12, "Zl": 13, "Zp": 14, "Cc": 15, "Cf": 16, "Co": 18, "Cs": 19, "Pd": 20, "Ps": 21, "Pe": 22,
         "Pc": 23, "Po": 24, "Sm": 25, "Sc": 26, "Sk": 27, "So": 28, "Pi": 29, "Pf": 30, "Cn": 0}
starts, cats = [], []
prev = None
for cp in range(0x110000):
    c = CODES[unicodedata.category(chr(cp))]
    if c != prev:
        starts.append(cp)
        cats.append(c)
        prev = c
# per-character full lowercase mappings (what Rust char::to_lowercase / Python chr(cp).lower() give; NO final-sigma context rule)
lsrc, lstart, llen, lflat = [], [], [], []
for cp in range(0x110000):
    if 0xD800 <= cp <= 0xDFFF:
        continue
    ch = chr(cp)
    lo = ch.lower()
    if lo != ch:
        lsrc.append(cp)
        lstart.append(len(lflat))
        llen.append(len(lo))
        lflat.extend(ord(c) for c in lo)
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "..", "..", "+OceanNLP", "+Text", "unicode_category_table.mat")
savemat(out, {"starts": np.array(starts, dtype=np.int32), "cats": np.array(cats, dtype=np.int8),
              "unicode_version": unicodedata.unidata_version,
              "lower_src": np.array(lsrc, dtype=np.int32), "lower_start": np.array(lstart, dtype=np.int32),
              "lower_len": np.array(llen, dtype=np.int8), "lower_flat": np.array(lflat, dtype=np.int32)})
print("lowercase mappings:", len(lsrc), "ranges:", len(starts), "unicode", unicodedata.unidata_version, "->", os.path.normpath(out))
