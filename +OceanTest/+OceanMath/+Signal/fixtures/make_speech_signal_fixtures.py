"""Reference fixtures for the speech signal-math tests (OceanMath.Signal).

Every value written here is computed by an INDEPENDENT library (numpy, scipy,
librosa) from real audio or from a seeded random input that is saved alongside
it. Nothing is typed in. Re-run from the repository root:

    ./.venv/Scripts/python.exe +OceanTest/+OceanMath/+Signal/fixtures/make_speech_signal_fixtures.py

Files: <name>.npy (read in MATLAB with OceanData.Adapters.NpyReader, same index
meaning as numpy) and speech_signal_fixtures.json (scalar parameters).
"""
import json
import os

import librosa
import numpy as np
import scipy.fft
import scipy.signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FSDD = os.path.join(ROOT, "Data", "Datasets", "Speech", "FSDD", "recordings")
MATLAB_SAMPLES = r"C:/Program Files/MATLAB/R2026a/toolbox/audio/samples"


def save(name, arr):
    np.save(os.path.join(HERE, name + ".npy"), np.asarray(arr, dtype=np.float64))


def read_wav(path):
    """int16 PCM -> float in [-1, 1) exactly as MATLAB audioread(...,'double')."""
    sr, x = wavfile.read(path)
    assert x.dtype == np.int16, x.dtype
    return sr, x.astype(np.float64) / 32768.0


meta = {}

# ---------------------------------------------------------------- windows
for L in (200, 7):
    for periodic in (True, False):
        tag = f"L{L}_{'periodic' if periodic else 'symmetric'}"
        save(f"win_hann_{tag}", scipy.signal.get_window("hann", L, fftbins=periodic))
        save(f"win_hamming_{tag}", scipy.signal.get_window("hamming", L, fftbins=periodic))
        # Kaldi "povey" = hann ** 0.85 (computed from scipy's hann, not from Ocean)
        save(f"win_povey_{tag}", scipy.signal.get_window("hann", L, fftbins=periodic) ** 0.85)

# ---------------------------------------------------------------- framing
sr, x = read_wav(os.path.join(FSDD, "0_jackson_0.wav"))
seg = x[:1001]
save("frame_input", seg)
meta["frame_input_file"] = "recordings/0_jackson_0.wav"
meta["frame_input_n"] = len(seg)
for L, H in ((200, 80), (7, 3)):
    P = L // 2
    for mode in ("constant", "reflect"):
        padded = np.pad(seg, P, mode=mode)
        fr = librosa.util.frame(padded, frame_length=L, hop_length=H, axis=-1)  # L x T
        save(f"frames_center_{mode}_L{L}_H{H}", fr)
    fr = librosa.util.frame(seg, frame_length=L, hop_length=H, axis=-1)
    save(f"frames_nocenter_L{L}_H{H}", fr)

# ---------------------------------------------------------------- power spectrum
win = scipy.signal.get_window("hann", 200, fftbins=True)
fr = librosa.util.frame(seg, frame_length=200, hop_length=80, axis=-1) * win[:, None]
for nfft in (256, 255, 200):
    save(f"pow_nfft{nfft}", np.abs(np.fft.rfft(fr, n=nfft, axis=0)) ** 2)

# ---------------------------------------------------------------- mel filter banks
mel_cfgs = [
    dict(name="mel_8k_256_40_htk_none", sr=8000, n_fft=256, n_mels=40, fmin=0.0, fmax=4000.0, htk=True, norm=None),
    dict(name="mel_16k_512_64_slaney_slaney", sr=16000, n_fft=512, n_mels=64, fmin=20.0, fmax=7600.0, htk=False, norm="slaney"),
    dict(name="mel_8k_255_26_htk_slaney", sr=8000, n_fft=255, n_mels=26, fmin=60.0, fmax=3800.0, htk=True, norm="slaney"),
    dict(name="mel_44k_1024_80_slaney_none", sr=44100, n_fft=1024, n_mels=80, fmin=0.0, fmax=22050.0, htk=False, norm=None),
]
for c in mel_cfgs:
    W = librosa.filters.mel(sr=c["sr"], n_fft=c["n_fft"], n_mels=c["n_mels"], fmin=c["fmin"], fmax=c["fmax"],
                            htk=c["htk"], norm=c["norm"], dtype=np.float64)
    centres = librosa.mel_frequencies(n_mels=c["n_mels"] + 2, fmin=c["fmin"], fmax=c["fmax"], htk=c["htk"])[1:-1]
    save(c["name"], W)
    save(c["name"] + "_centres", centres)
meta["mel_cfgs"] = [{k: (v if v is not None else "none") for k, v in c.items()} for c in mel_cfgs]
fgrid = np.linspace(0.0, 11025.0, 513)
save("hz_grid", fgrid)
save("hz2mel_htk", librosa.hz_to_mel(fgrid, htk=True))
save("hz2mel_slaney", librosa.hz_to_mel(fgrid, htk=False))

# ---------------------------------------------------------------- DCT / lifter
rng = np.random.default_rng(20261006)
X = rng.standard_normal((40, 7))
save("dct_input", X)
save("dct_ortho_13", scipy.fft.dct(X, type=2, norm="ortho", axis=0)[:13])
save("dct_none_13", scipy.fft.dct(X, type=2, norm=None, axis=0)[:13])
save("dct_ortho_40", scipy.fft.dct(X, type=2, norm="ortho", axis=0))
save("mfcc_lifter22", librosa.feature.mfcc(S=X, n_mfcc=13, dct_type=2, norm="ortho", lifter=22))

# ---------------------------------------------------------------- resampling
def resample_ref(x, fs_in, fs_out, half_width, beta):
    g = np.gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    r = max(up, down)
    h = scipy.signal.firwin(2 * half_width * r + 1, 1.0 / r, window=("kaiser", beta))
    return scipy.signal.resample_poly(x, up, down, window=h), h * up, up, down


res_cases = []
_, x8 = read_wav(os.path.join(FSDD, "3_theo_10.wav"))
save("res_in_8k", x8)
sr44, x44 = read_wav(os.path.join(MATLAB_SAMPLES, "Counting-16-44p1-mono-15secs.wav"))
assert sr44 == 44100
x44 = x44[sr44 : 2 * sr44]
save("res_in_44k", x44)
for name, xin, fin, fout, hw, beta in (
    ("res_8k_16k", x8, 8000, 16000, 16, 8.6),
    ("res_8k_11025", x8, 8000, 11025, 16, 8.6),
    ("res_44k_8k", x44, 44100, 8000, 16, 8.6),
    ("res_8k_16k_hw10_b5", x8, 8000, 16000, 10, 5.0),
):
    y, h, up, down = resample_ref(xin, fin, fout, hw, beta)
    save(name, y)
    save(name + "_h", h)
    res_cases.append(dict(name=name, fs_in=fin, fs_out=fout, half_width=hw, beta=beta, up=up, down=down))
meta["res_cases"] = res_cases
meta["res_in_44k_file"] = "toolbox/audio/samples/Counting-16-44p1-mono-15secs.wav (samples 44101..88200, second 1 s)"
meta["generator"] = {"numpy": np.__version__, "scipy": scipy.__version__, "librosa": librosa.__version__}

with open(os.path.join(HERE, "speech_signal_fixtures.json"), "w") as f:
    json.dump(meta, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote fixtures to", HERE)
