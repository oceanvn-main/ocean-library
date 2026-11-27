"""Dumps Cranfield doc/query texts to cranfield_texts.json so neural_reference.py
can run in the Python environment that has torch (ir_datasets may not be
installed there). Parses the archive ir_datasets already cached at
~/.ir_datasets/cranfield/cran.tar.gz directly: cran.all.1400 / cran.qry use the
classic SMART format (.I id, .T title, .A author, .B bib, .W abstract text).
Doc order = file order (same order as ir_datasets' docs_iter, so doc index i
matches cranfield_validation_data.mat). Text = the .W field.

Run:  py dump_cranfield_texts.py
"""
import json
import os
import re
import tarfile

path = os.path.join(os.path.expanduser("~"), ".ir_datasets", "cranfield", "cran.tar.gz")


def parse(raw):
    recs, cur, field = [], None, None
    for line in raw.splitlines():
        m = re.match(r"^\.([ITABW])\s*(.*)$", line)
        if m:
            tag, rest = m.groups()
            if tag == "I":
                cur = {"id": rest.strip(), "W": []}
                recs.append(cur)
                field = None
            else:
                field = tag
        elif cur is not None and field == "W":
            cur["W"].append(line)
    return [(r["id"], " ".join(" ".join(r["W"]).split())) for r in recs]


with tarfile.open(path) as t:
    docs = parse(t.extractfile("cran.all.1400").read().decode("latin-1"))
    qrys = parse(t.extractfile("cran.qry").read().decode("latin-1"))

assert len(docs) == 1400, len(docs)
json.dump({"docs": [text for _, text in docs],
           "queries": {str(int(i)): text for i, text in qrys}},
          open("cranfield_texts.json", "w", encoding="utf-8"))
print("wrote cranfield_texts.json:", len(docs), "docs,", len(qrys), "queries")
