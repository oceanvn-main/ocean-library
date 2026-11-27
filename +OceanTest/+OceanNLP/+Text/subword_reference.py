"""Subword-tokenizer references (Python) on real Cranfield text + the Unicode torture set.

  (1) INFERENCE: pretrained GPT-2 (openai-community/gpt2, vocab.json + merges.txt) via
      transformers GPT2TokenizerFast -> ids and token strings for every text.
  (2) TRAINING : HF `tokenizers` BpeTrainer (ByteLevel, add_prefix_space=False, vocab 1000, on all
      1400 Cranfield docs) -> its merge list and the ids/tokens it gives to the first 150 docs.

Writes gpt2_vocab.json, gpt2_merges.txt, subword_reference.json, bpe_train_reference.json.
Run:  py subword_reference.py     (needs pretok_reference.json's inputs: cranfield_texts.json)
"""
import json
import os
import shutil

from huggingface_hub import hf_hub_download
from transformers import GPT2TokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

here = os.path.dirname(os.path.abspath(__file__))
texts = json.load(open(os.path.join(here, "..", "+Retrieval", "cranfield_texts.json"), encoding="utf-8"))
pre = json.load(open(os.path.join(here, "pretok_reference.json"), encoding="utf-8"))
torture = [c["text"] for c in pre[:39]]
fuzz = [c["text"] for c in pre[249:249 + 400] if chr(0) not in c["text"]]   # Unicode fuzz strings (pretok_reference.py)
docs = texts["docs"]

# ---- (1) pretrained GPT-2 inference
for f in ("vocab.json", "merges.txt"):
    shutil.copy(hf_hub_download("openai-community/gpt2", f), os.path.join(here, "gpt2_" + f))
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
cases = torture + fuzz + docs[:300] + list(texts["queries"].values())[:60]
out = []
for t in cases:
    ids = tok(t)["input_ids"]
    out.append({"text": t, "ids": ids, "tokens": tok.convert_ids_to_tokens(ids), "decoded": tok.decode(ids)})
json.dump(out, open(os.path.join(here, "subword_reference.json"), "w", encoding="utf-8"))
print("gpt2 inference cases:", len(out))

# ---- (2) HF BPE training
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=1000, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                              show_progress=False, special_tokens=[])
t.train_from_iterator(docs, trainer=trainer)
model = json.loads(t.to_str())["model"]
merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
enc = [t.encode(d) for d in docs[:150]]
json.dump({"vocab_size": len(model["vocab"]), "merges": merges,
           "encodings": [{"ids": e.ids, "tokens": e.tokens} for e in enc]},
          open(os.path.join(here, "bpe_train_reference.json"), "w", encoding="utf-8"))
print("HF BPE trained: vocab", len(model["vocab"]), "merges", len(merges))

# ---- (3) HF WordPiece training (BERT-style: lowercase/strip-accents normalizer + BertPreTokenizer)
from tokenizers import normalizers
wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wp.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)
wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
wtr = trainers.WordPieceTrainer(vocab_size=1000, special_tokens=["[UNK]"], continuing_subword_prefix="##", show_progress=False)
wp.train_from_iterator(docs, trainer=wtr)
vocab = json.loads(wp.to_str())["model"]["vocab"]
pieces = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
wenc = [wp.encode(d) for d in docs[:150]]
json.dump({"pieces": pieces, "encodings": [{"tokens": e.tokens} for e in wenc]},
          open(os.path.join(here, "wordpiece_train_reference.json"), "w", encoding="utf-8"))
print("HF WordPiece trained: vocab", len(pieces))


# ---- (4) additional training configurations (tie-heavy synthetic, multilingual, other vocab sizes)
import random


def train_pair(corpus, vocab):
    bt = Tokenizer(models.BPE())
    bt.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bt.train_from_iterator(corpus, trainer=trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                              show_progress=False, special_tokens=[]))
    bm = json.loads(bt.to_str())["model"]
    merges_ = [m if isinstance(m, str) else " ".join(m) for m in bm["merges"]]
    wt = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    wt.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)
    wt.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    wt.train_from_iterator(corpus, trainer=trainers.WordPieceTrainer(vocab_size=vocab, special_tokens=["[UNK]"],
                                                                     continuing_subword_prefix="##", show_progress=False))
    wv = json.loads(wt.to_str())["model"]["vocab"]
    pieces_ = [k for k, _ in sorted(wv.items(), key=lambda kv: kv[1])]
    probe = corpus[:60]
    return {"corpus": corpus, "vocab": vocab, "merges": merges_, "pieces": pieces_,
            "bpe_tokens": [bt.encode(x).tokens for x in probe], "wp_tokens": [wt.encode(x).tokens for x in probe]}


rnd = random.Random(7)
tie = ["".join(rnd.choice("abcdef") for _ in range(rnd.randint(2, 6))) for _ in range(300)]
tie_corpus = [" ".join(tie[i:i + 8]) for i in range(0, 296, 8)]
cyr = "\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u043a\u043b\u043c\u043d\u043e\u043f"
cjk = "\u4e2d\u6587\u5b57\u7b26\u6d4b\u8bd5\u6570\u636e"
multi = [" ".join("".join(rnd.choice(rnd.choice([cyr, cjk, "abcdefgh", "xyz"])) for _ in range(rnd.randint(2, 6))) for _ in range(8)) for _ in range(60)]
extra = {
    "cranfield200_v400": train_pair(docs[:200], 400),
    "cranfield600_v2500": train_pair(docs[:600], 2500),
    "tie_heavy_v420": train_pair(tie_corpus, 420),
    "multilingual_v300": train_pair(multi, 300),
}
json.dump(extra, open(os.path.join(here, "extra_train_reference.json"), "w", encoding="utf-8"))
print("extra training configs:", list(extra))
