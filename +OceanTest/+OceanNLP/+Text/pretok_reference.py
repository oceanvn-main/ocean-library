"""Pre-tokenizer references on real Cranfield text + a Unicode torture set.

  gpt2      : the exact GPT-2 pattern evaluated by the `regex` module (Unicode \p{L}, \p{N})
  bert_un   : HF `tokenizers` BertNormalizer(lowercase=True) + BertPreTokenizer   (uncased BERT BasicTokenizer)
  bert_cs   : HF `tokenizers` BertNormalizer(lowercase=False) + BertPreTokenizer  (cased BERT BasicTokenizer)
              (transformers 5 no longer ships BasicTokenizer; the Rust normalizer+pre-tokenizer is its
               production equivalent. No never-split list: [MASK] is split like any text.)

Writes pretok_reference.json = list of {text, gpt2, bert_un, bert_cs}. Run: py pretok_reference.py
Needs: regex, tokenizers, and ../+Retrieval/cranfield_texts.json (dump_cranfield_texts.py).
"""
import json
import os

import regex
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer

here = os.path.dirname(os.path.abspath(__file__))
GPT2 = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
nu = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)
nc = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=False)
pre = BertPreTokenizer()


def bert(norm, t):
    return [w for w, _ in pre.pre_tokenize_str(norm.normalize_str(t))]

torture = [
    "Hello world's  it's 42!!", "  leading and trailing  ", "a\tb\nc", "don't I'll we've they're I'M I'D",
    "x   y", "café naïve Über", "中文 text 日本語", "smile \U0001F600 done \U0001F468‍\U0001F469",
    "3.14159 and 1,000,000", "a--b -- c", "[MASK] is [CLS] and [SEP]", "“quoted” — dash …",
    "under_score snake_case", "  ", "", "'s", "'S", "ll", " 'll", " nbsp ", "tab\t\tdouble",
    "trailing space ", "\n\nnewlines\n", "e-mail@example.com", "\x01ctrl\x02", "x́ combining", "٣٤ arabic-indic",
    "Müller-Lyon's café's", "  'quoted'  ", "a  \n  b", "hello    ", "    hello", "!!!???", "abc123def456",
    "퟿￿", " sep ", "½ ² x²", "ALL CAPS Title Case", "İstanbul ẞ",
]
texts = json.load(open(os.path.join(here, "..", "+Retrieval", "cranfield_texts.json"), encoding="utf-8"))
real = texts["docs"][:150] + list(texts["queries"].values())[:60]
import random
rnd = random.Random(12345)
pools = [
    [chr(c) for c in range(32, 127)] * 3,                                             # ASCII printable
    [chr(c) for c in (9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 0x1680, 0x2000, 0x2003, 0x200A,
                      0x2028, 0x2029, 0x202F, 0x205F, 0x3000)],                       # whitespace-ish
    [chr(c) for c in list(range(0, 9)) + [14, 15, 16, 127, 128, 133, 159, 173, 0x200B, 0x200E, 0x202E, 0xFEFF]],   # controls / format
    [chr(c) for c in range(0xA1, 0x180)],                                             # Latin-1 / Latin Extended-A
    [chr(c) for c in range(0x0300, 0x0370, 3)],                                       # combining marks
    [chr(c) for c in range(0x0391, 0x03C9)] + [chr(c) for c in range(0x0410, 0x0450)],  # Greek, Cyrillic
    [chr(c) for c in range(0x4E00, 0x4E40)] + [chr(0x3042), chr(0x30A2), chr(0xAC00)],  # CJK, kana, hangul
    [chr(c) for c in range(0x1E900, 0x1E920)] + [chr(c) for c in range(0x1C90, 0x1CB0)],  # Adlam, Georgian Mtavruli (new in Unicode 9/11)
    [chr(c) for c in range(0x30000, 0x30010)] + [chr(c) for c in range(0x2A700, 0x2A710)],  # CJK ext C/G
    [chr(c) for c in range(0x1F600, 0x1F640)] + [chr(0x1F468), chr(0x200D)],         # emoji
    [chr(c) for c in (0xE000, 0xF8FF, 0xF0000, 0x378, 0xD7FF, 0xFFFF, 0xFFFE, 0x10FFFF, 0xFFFD)],   # private use / unassigned / specials
    [chr(c) for c in (0x1D7CE, 0x0663, 0x00B2, 0x2160, 0x00BD, 0x0F33)],              # digits / numbers of other kinds
    [chr(c) for c in list(range(0x0915, 0x0940)) + [0x0903, 0x0BCD, 0x20DD, 0x0488]],   # Indic letters + Mc/Me marks (must survive accent stripping)
    [chr(c) for c in (0x2018, 0x2019, 0x201C, 0x201D, 0x2014, 0x2026, 0x00A1, 0x00BF, 0x3001, 0x00AB)],   # Unicode punctuation
]
fuzz = []
for _ in range(1200):
    n = rnd.randint(1, 14)
    fuzz.append("".join(rnd.choice(rnd.choice(pools)) for _ in range(n)))
out = []
for t in torture + real + fuzz:
    out.append({"text": t, "gpt2": GPT2.findall(t), "bert_un": bert(nu, t), "bert_cs": bert(nc, t)})
json.dump(out, open(os.path.join(here, "pretok_reference.json"), "w", encoding="utf-8"))
print("cases:", len(out), "(torture:", len(torture), "real:", len(real), "fuzz:", len(fuzz), ")")
