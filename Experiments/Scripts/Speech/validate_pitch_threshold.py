"""Python half of validate_pitch_threshold.m: librosa.pyin voicing flags on real audio.

    python validate_pitch_threshold.py <job.json>
job.json: {"out": <dir>, "files": [{"key": ..., "path": <abs wav>}], "fmin": 50, "fmax": 400}
Same frame alignment as validate_features_external.py: pyin frame_length = 2L, the
signal padded by L/2 (left) so pyin frame t is centred on Ocean frame t's centre
(Ocean grid: L = round(0.025 fs), H = round(0.010 fs), center=False).
"""
import json
import os
import sys

import librosa
import numpy as np
from scipy.io import wavfile


def main(job_file):
    with open(job_file) as f:
        job = json.load(f)
    out = job["out"]
    os.makedirs(out, exist_ok=True)
    for item in job["files"]:
        sr, x = wavfile.read(item["path"])
        y = x.astype(np.float64) / 32768.0
        L = int(np.floor(0.025 * sr + 0.5))
        H = int(np.floor(0.010 * sr + 0.5))
        fl = 2 * L
        ypp = np.pad(y, (L // 2, fl - L - L // 2))
        f0, vflag, _ = librosa.pyin(ypp, fmin=job["fmin"], fmax=job["fmax"], sr=sr, frame_length=fl,
                                    hop_length=H, center=False, fill_na=0.0)
        np.save(os.path.join(out, item["key"] + "_f0.npy"), f0)
        np.save(os.path.join(out, item["key"] + "_voiced.npy"), vflag.astype(np.float64))
    with open(os.path.join(out, "versions.json"), "w") as f:
        json.dump({"librosa": librosa.__version__}, f)


if __name__ == "__main__":
    main(sys.argv[1])
