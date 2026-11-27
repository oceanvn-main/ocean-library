"""librosa reference fixtures for OceanNLP.Speech.Features (LogMel, MFCC, DeltaFeatures, EnergyContour).

All references are computed by librosa/numpy on REAL audio (FSDD + the MATLAB
Audio Toolbox sample SpeechDFT-16-8-mono-5secs.wav). Re-run from the repo root:

    ./.venv/Scripts/python.exe +OceanTest/+OceanNLP/+Speech/+Features/fixtures/make_feature_fixtures.py

FRAME ALIGNMENT WITH librosa (documented convention difference):
librosa.stft always cuts frames of length n_fft and zero-pads a shorter window
(win_length < n_fft) into the CENTRE of that frame (pad_center, left pad
(n_fft - L)//2). The Ocean grid (contract sec. 3) cuts frames of length L and
zero-pads at the END to NFFT. The power spectrum is invariant to that shift, so
for center=False we feed librosa the signal padded with (n_fft-L)//2 zeros on
the left and the rest on the right; librosa's windowed region of frame t is then
exactly Ocean's samples [tH, tH+L). For center=True librosa pads n_fft//2, which
places the window at tH - floor(L/2) whenever n_fft - L is even -- the Ocean
convention -- so those cases use librosa directly.
"""
import json
import os

import librosa
import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
FSDD = os.path.join(ROOT, "Data", "Datasets", "Speech", "FSDD", "recordings")
MATLAB_SAMPLES = r"C:/Program Files/MATLAB/R2026a/toolbox/audio/samples"
FLOOR = 1e-10


def save(name, arr):
    np.save(os.path.join(HERE, name + ".npy"), np.asarray(arr, dtype=np.float64))


def read_wav(path):
    sr, x = wavfile.read(path)
    assert x.dtype == np.int16
    return sr, x.astype(np.float64) / 32768.0


def librosa_mel(y, sr, L, H, n_fft, window, n_mels, fmin, fmax, htk, norm, power, center, pad_mode="constant"):
    if not center:
        lpad = (n_fft - L) // 2
        y = np.pad(y, (lpad, n_fft - L - lpad))
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=H, win_length=L, window=window, center=center, pad_mode=pad_mode,
        power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm, dtype=np.float64)


cases = [
    dict(name="A", file="FSDD:3_theo_10.wav", L=200, H=80, n_fft=256, window="hann", n_mels=40, fmin=0.0,
         fmax=4000.0, htk=True, norm=None, power=2.0, center=False, pad_mode="constant"),
    dict(name="B", file="MATLAB:SpeechDFT-16-8-mono-5secs.wav", L=256, H=128, n_fft=256, window="hann", n_mels=26,
         fmin=100.0, fmax=3800.0, htk=False, norm="slaney", power=1.0, center=False, pad_mode="constant"),
    dict(name="C", file="FSDD:7_nicolas_3.wav", L=200, H=100, n_fft=255, window="hamming", n_mels=24, fmin=0.0,
         fmax=4000.0, htk=True, norm=None, power=2.0, center=False, pad_mode="constant"),
    dict(name="D", file="FSDD:3_theo_10.wav", L=200, H=80, n_fft=256, window="hann", n_mels=40, fmin=0.0,
         fmax=4000.0, htk=True, norm=None, power=2.0, center=True, pad_mode="constant"),
    dict(name="E", file="FSDD:3_theo_10.wav", L=200, H=80, n_fft=256, window="hann", n_mels=40, fmin=0.0,
         fmax=4000.0, htk=True, norm=None, power=2.0, center=True, pad_mode="reflect"),
]

for c in cases:
    src, fname = c["file"].split(":")
    path = os.path.join(FSDD if src == "FSDD" else MATLAB_SAMPLES, fname)
    sr, y = read_wav(path)
    c["sr"] = sr
    save(f"{c['name']}_input", y)
    M = librosa_mel(y, sr, c["L"], c["H"], c["n_fft"], c["window"], c["n_mels"], c["fmin"], c["fmax"], c["htk"],
                    c["norm"], c["power"], c["center"], c["pad_mode"])
    logmel = np.log(np.maximum(M, FLOOR))
    save(f"{c['name']}_logmel", logmel)
    save(f"{c['name']}_mfcc13", librosa.feature.mfcc(S=logmel, n_mfcc=13, dct_type=2, norm="ortho", lifter=0))
    save(f"{c['name']}_mfcc13_lift22", librosa.feature.mfcc(S=logmel, n_mfcc=13, dct_type=2, norm="ortho", lifter=22))
    if c["name"] == "A":
        mf = librosa.feature.mfcc(S=logmel, n_mfcc=13, dct_type=2, norm="ortho", lifter=0)
        for N in (1, 2, 3):
            d1 = librosa.feature.delta(mf, width=2 * N + 1, order=1, mode="nearest")
            d2 = librosa.feature.delta(d1, width=2 * N + 1, order=1, mode="nearest")
            save(f"A_delta_N{N}", d1)
            save(f"A_ddelta_N{N}", d2)
        frames = librosa.util.frame(y, frame_length=c["L"], hop_length=c["H"], axis=-1)
        save("A_energy", np.log(np.maximum((frames ** 2).sum(axis=0), FLOOR)))
    c["norm"] = c["norm"] if c["norm"] is not None else "none"

meta = {"cases": cases, "log_floor": FLOOR,
        "generator": {"librosa": librosa.__version__, "numpy": np.__version__}}
with open(os.path.join(HERE, "feature_fixtures.json"), "w") as f:
    json.dump(meta, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote", HERE)
