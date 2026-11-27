"""Python half of validate_features_external.m: librosa / scipy references on real audio.

Called by the MATLAB script as
    python validate_features_external.py <job.json>
job.json: {"out": <dir>, "files": [{"key": "f01", "path": <abs wav>}, ...],
           "resample": [{"key": ..., "path": ..., "fs_out": ..., "half_width": ..., "beta": ...}],
           "n_mels": 40, "fmin": 50, "fmax": 400}
Writes <out>/<key>_<what>.npy. Every number is computed here by librosa/scipy;
the MATLAB script computes the Ocean side and the differences.

Frame alignment: Ocean frames are L samples, zero-padded at the END to NFFT;
librosa frames are n_fft samples with the window centred. Feeding librosa the
signal padded by (n_fft-L)//2 zeros on the left (and the remainder on the right)
makes librosa's windowed samples of frame t exactly Ocean's [tH, tH+L)
(center=False). pyin uses frame_length = 2L, padded by L/2 on both sides, so its
frame t is centred on Ocean frame t's centre.
"""
import json
import os
import sys

import librosa
import numpy as np
import scipy.signal
from scipy.io import wavfile


def read_wav(path):
    sr, x = wavfile.read(path)
    assert x.dtype == np.int16, (path, x.dtype)
    x = x.astype(np.float64) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x


def grid(sr):
    # MATLAB round() (half away from zero), NOT Python's banker's rounding:
    # at 44.1 kHz 0.025 fs = 1102.5 -> 1103 as in OceanNLP.Speech.Core.FrameGrid
    L = int(np.floor(0.025 * sr + 0.5))
    H = int(np.floor(0.010 * sr + 0.5))
    n_fft = int(2 ** np.ceil(np.log2(L)))
    return L, H, n_fft


def main(job_path):
    with open(job_path) as f:
        job = json.load(f)
    out = job["out"]
    os.makedirs(out, exist_ok=True)
    save = lambda name, a: np.save(os.path.join(out, name + ".npy"), np.asarray(a, dtype=np.float64))
    n_mels = job["n_mels"]
    for item in job["files"]:
        sr, y = read_wav(item["path"])
        L, H, n_fft = grid(sr)
        lpad = (n_fft - L) // 2
        yp = np.pad(y, (lpad, n_fft - L - lpad))
        W = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2, htk=True, norm=None,
                                dtype=np.float64)
        M = librosa.feature.melspectrogram(y=yp, sr=sr, n_fft=n_fft, hop_length=H, win_length=L, window="hann",
                                           center=False, power=2.0, n_mels=n_mels, fmin=0.0, fmax=sr / 2, htk=True,
                                           norm=None, dtype=np.float64)
        logmel = np.log(np.maximum(M, 1e-10))
        mf = librosa.feature.mfcc(S=logmel, n_mfcc=13, dct_type=2, norm="ortho", lifter=0)
        d1 = librosa.feature.delta(mf, width=5, order=1, mode="nearest")
        d2 = librosa.feature.delta(d1, width=5, order=1, mode="nearest")
        k = item["key"]
        save(k + "_melbank", W)
        save(k + "_melpow", M)
        save(k + "_logmel", logmel)
        save(k + "_mfcc", mf)
        save(k + "_d1", d1)
        save(k + "_d2", d2)
        if job.get("pitch", True) and sr <= 16000:
            fl = 2 * L
            ypp = np.pad(y, (L // 2, fl - L - L // 2))
            f0, vflag, vprob = librosa.pyin(ypp, fmin=job["fmin"], fmax=job["fmax"], sr=sr, frame_length=fl,
                                            hop_length=H, center=False, fill_na=0.0)
            save(k + "_pyin_f0", f0)
            save(k + "_pyin_voiced", vflag.astype(np.float64))
    for item in job["resample"]:
        sr, y = read_wav(item["path"])
        g = np.gcd(int(sr), int(item["fs_out"]))
        up, down = int(item["fs_out"]) // g, int(sr) // g
        r = max(up, down)
        h = scipy.signal.firwin(2 * item["half_width"] * r + 1, 1.0 / r, window=("kaiser", item["beta"]))
        save(item["key"] + "_scipy", scipy.signal.resample_poly(y, up, down, window=h))
        save(item["key"] + "_scipy_default", scipy.signal.resample_poly(y, up, down))
    with open(os.path.join(out, "versions.json"), "w") as f:
        json.dump({"librosa": librosa.__version__, "scipy": scipy.__version__, "numpy": np.__version__}, f)


if __name__ == "__main__":
    main(sys.argv[1])
