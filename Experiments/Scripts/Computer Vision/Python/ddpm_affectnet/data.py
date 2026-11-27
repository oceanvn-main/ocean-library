"""AffectNet faces for an UNCONDITIONAL DDPM (labels are not used).

Selection (the "option b" protocol):
  * only AffectNet photographs (*.jpg); FFHQ images (ffhq_*.png, the happy /
    neutral folders of this copy) are excluded, so all faces share one source
    and photo style;
  * byte-identical files are removed (MD5 of the file bytes; this copy has
    ~1,688 test files that are copies of train files); the first path in
    sorted order is kept;
  * the provided Train/Test folders are pooled and re-split with a seeded
    random permutation (test_fraction held out for evaluation), so no image
    appears in both splits.

Images are resized once (Lanczos) to image_size x image_size RGB and kept in
memory as uint8 (N, 3, H, W) -- ~12 KB per face at 64 x 64, nothing is written
back to disk.

Only numpy + Pillow are needed here (no torch), so this module is unit-tested
on any machine.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class FaceSplit:
    train_files: list
    test_files: list
    report: dict


def find_root(data_dir: str) -> str:
    """Directory that contains the Train/ and Test/ folders (the zip may add a level)."""
    for here, dirs, _ in os.walk(data_dir):
        names = {d.lower() for d in dirs}
        if "train" in names and "test" in names:
            return here
    raise FileNotFoundError(f"No folder with Train/ and Test/ found under {data_dir}")


def list_images(root: str) -> list:
    """All image files under Train/ and Test/ (any class folder), sorted."""
    out = []
    for split in os.listdir(root):
        if split.lower() not in ("train", "test"):
            continue
        for here, _, files in os.walk(os.path.join(root, split)):
            out += [os.path.join(here, f) for f in files]
    return sorted(out)


def is_affectnet_photo(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name.endswith((".jpg", ".jpeg")) and not name.startswith("ffhq_")


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def select_and_split(data_dir: str, test_fraction: float = 0.2, split_seed: int = 0) -> FaceSplit:
    root = find_root(data_dir)
    files = list_images(root)
    photos = [p for p in files if is_affectnet_photo(p)]
    seen, unique = set(), []
    for p in photos:                                   # sorted order: first copy wins
        h = md5(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(unique))
    n_test = int(round(test_fraction * len(unique)))
    test = sorted(unique[i] for i in order[:n_test])
    train = sorted(unique[i] for i in order[n_test:])
    report = {"root": root, "files": len(files), "affectnet_photos": len(photos),
              "excluded_non_affectnet": len(files) - len(photos),
              "duplicates_removed": len(photos) - len(unique), "unique": len(unique),
              "train": len(train), "test": len(test), "test_fraction": test_fraction, "split_seed": split_seed}
    return FaceSplit(train, test, report)


def load_images(files: list, image_size: int) -> np.ndarray:
    """uint8 array (N, 3, image_size, image_size); grey JPEGs are converted to RGB."""
    out = np.empty((len(files), 3, image_size, image_size), dtype=np.uint8)
    for i, p in enumerate(files):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (image_size, image_size):
                im = im.resize((image_size, image_size), Image.LANCZOS)
            out[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
    return out
