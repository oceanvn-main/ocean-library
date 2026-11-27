"""Data selection: AffectNet JPEGs only, exact duplicates removed, disjoint seeded split."""
import os

import numpy as np
from PIL import Image

import data


def _img(path, value, mode="RGB", size=96):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((size, size, 3) if mode == "RGB" else (size, size), value, np.uint8)
    Image.fromarray(arr, mode).save(path, quality=95)


def _make(root):
    _img(f"{root}/Train/anger/a.jpg", 10)
    _img(f"{root}/Train/anger/b.jpg", 20)
    _img(f"{root}/Train/happy/ffhq_1.png", 30)                  # FFHQ: excluded
    _img(f"{root}/Test/fear/d.jpg", 40, mode="L")               # grey JPEG: converted to RGB
    _img(f"{root}/Test/Sad/e.jpg", 50)
    with open(f"{root}/Train/anger/a.jpg", "rb") as f:          # byte copy in Test: removed
        raw = f.read()
    os.makedirs(f"{root}/Test/Sad", exist_ok=True)
    with open(f"{root}/Test/Sad/copy_of_a.jpg", "wb") as f:
        f.write(raw)


def test_selection_dedup_and_split(tmp_path):
    _make(str(tmp_path / "unzipped" / "AffectNet"))              # the zip may add a folder level
    s = data.select_and_split(str(tmp_path / "unzipped"), test_fraction=0.25, split_seed=0)
    r = s.report
    assert r["files"] == 6
    assert r["affectnet_photos"] == 5 and r["excluded_non_affectnet"] == 1
    assert r["duplicates_removed"] == 1 and r["unique"] == 4
    assert r["train"] + r["test"] == 4 and r["test"] == 1
    assert not set(s.train_files) & set(s.test_files)
    assert not any(os.path.basename(p).startswith("ffhq_") for p in s.train_files + s.test_files)
    s2 = data.select_and_split(str(tmp_path / "unzipped"), test_fraction=0.25, split_seed=0)
    assert s2.train_files == s.train_files and s2.test_files == s.test_files   # reproducible


def test_load_images_shape_and_rgb(tmp_path):
    _make(str(tmp_path))
    s = data.select_and_split(str(tmp_path), test_fraction=0.25, split_seed=0)
    X = data.load_images(s.train_files + s.test_files, 64)
    assert X.dtype == np.uint8 and X.shape == (4, 3, 64, 64)
