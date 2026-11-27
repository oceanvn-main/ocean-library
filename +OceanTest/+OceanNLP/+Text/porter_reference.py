"""Porter stemmer reference: NLTK PorterStemmer(mode="ORIGINAL_ALGORITHM") stems for every
distinct word of the real Cranfield collection (+ a few hand-picked edge cases), written to
porter_reference.json for compare_text_in_matlab.m.

Words: lowercase \w+ tokens (any length >= 1, digits included) from +OceanTest/+OceanNLP/+Retrieval/cranfield_texts.json
(created by dump_cranfield_texts.py). Run:  py porter_reference.py
"""
import json
import os
import re
from nltk.stem.porter import PorterStemmer

here = os.path.dirname(os.path.abspath(__file__))
texts = json.load(open(os.path.join(here, "..", "+Retrieval", "cranfield_texts.json"), encoding="utf-8"))
words = set()
for t in texts["docs"] + list(texts["queries"].values()):
    words.update(re.findall(r"[a-z0-9]+", t.lower()))
words.update(["a", "as", "is", "ss", "s", "y", "yy", "ed", "ing", "eed", "ies", "sses", "caresses", "ponies",
              "ties", "feed", "agreed", "plastered", "bled", "motoring", "sing", "conflated", "troubled",
              "sized", "hopping", "tanned", "falling", "hissing", "fizzed", "failing", "filing", "happy",
              "sky", "relational", "conditional", "rational", "valenci", "hesitanci", "digitizer",
              "conformabli", "radicalli", "differentli", "vileli", "analogousli", "vietnamization",
              "predication", "operator", "feudalism", "decisiveness", "hopefulness", "callousness",
              "formaliti", "sensitiviti", "sensibiliti", "triplicate", "formative", "formalize",
              "electriciti", "electrical", "hopeful", "goodness", "revival", "allowance", "inference",
              "airliner", "gyroscopic", "adjustable", "defensible", "irritant", "replacement",
              "adjustment", "dependent", "adoption", "homologou", "communism", "activate",
              "angulariti", "homologous", "effective", "bowdlerize", "probate", "rate", "cease",
              "controll", "roll", "generalizations", "oscillators", "logi", "fulli", "alli", "bli"])
st = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
out = {w: st.stem(w) for w in sorted(words)}
json.dump(out, open(os.path.join(here, "porter_reference.json"), "w", encoding="utf-8"))
print("words:", len(out), "changed by stemming:", sum(1 for w, s in out.items() if w != s))


# ---- analyzer chain (plan validation 4b): lowercase -> Lucene English stop list -> Porter, on real docs
LUCENE_STOP = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not",
               "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was",
               "will", "with"}
chain = []
for t in texts["docs"][:200] + list(texts["queries"].values())[:40]:
    toks = [w for w in re.findall(r"[a-z0-9_]+", t.lower())]           # regex \w+ on ASCII text (Cranfield is pure ASCII)
    toks = [w for w in toks if w not in LUCENE_STOP]
    chain.append({"text": t, "tokens": [st.stem(w) for w in toks]})
json.dump(chain, open(os.path.join(here, "analyzer_reference.json"), "w", encoding="utf-8"))
print("analyzer chain cases:", len(chain))
