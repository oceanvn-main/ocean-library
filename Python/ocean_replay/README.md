# ocean_replay — record once, replay in MATLAB (Python side)

This is the Python mirror of the MATLAB package `+OceanReplay`. Both implement the same contract:
`Planning/System/Replay/Replay_Format_Spec.md`. That means:
- a recording made here (for example on a Google Cloud GPU) opens in MATLAB's **Replay Studio**;
- a recording made in MATLAB opens here.

The only requirements are `numpy` and `h5py`. The diffusers adapter also needs `torch`, and its
real-diffusers test needs `diffusers`; run those in Cloud Shell or on the VM.

## Pieces

| Module | What it does |
|---|---|
| `info`, `sampling`, `spec` | Axes, channels and views, the sampling rules, and the recording spec (JSON) |
| `writer`, `reader`, `schema` | HDF5 streaming writer, `Recording` (nearest stored point, never blended), `validate` |
| `recorder` | `IRecordable`, `Recorder` (pause → record → continue), `NullRecorder`, `verify_invariance` |
| `extract` | Archive → smaller file in the same format: `python -m ocean_replay extract ...` |
| `adapters/diffusers_ddpm` | `DiffusersDDPMRecordable`: the diffusers DDPM loop of `sample.py`, with the same channel names as MATLAB's `DiffusionRecordable` |
| `cloud/record_ddpm.py` | Records the **trained** AffectNet DDPM from its checkpoint (no training) |

## Typical flow (the trained 64×64 AffectNet DDPM)

1. **On the VM or in Cloud Shell,** record the full archive. It's about 197 MB, capped at 1 GB by the
   spec.
   ```
   python -m ocean_replay.cloud.record_ddpm --ddpm-code <ddpm_affectnet> --config <ddpm_affectnet>/configs/affectnet64.json \
       --checkpoint checkpoint.pt --spec ocean_replay/cloud/spec_ddpm_archive.json --out archive.h5 --seed 7
   ```
2. **Extract a small replay file** (about 11 MB). Cloud Shell is free for this step.
   ```
   python -m ocean_replay extract archive.h5 --spec ocean_replay/cloud/extract_ddpm_small.json --out small.h5
   ```
3. **In MATLAB,** open it: `OceanViz.App.Replay.ReplayStudio("C:\...\small.h5")`. The Diffusion
   layout opens automatically, and you can play or step through it with its legends.

## Tests

```
cd Python
python -m unittest discover -s ocean_replay/tests -t .       # 16 local tests (numpy, h5py, torch)
python -m ocean_replay.tests.make_fixtures                    # rewrite the Python fixtures for MATLAB
```

- **`test_cross_language.py`** reads MATLAB's fixtures and checks:
  - the values are identical;
  - the sampling cases (36) select the same points;
  - the extracts are dataset-identical.
- **MATLAB's `OceanTest.OceanReplay.CrossLanguageTest`** reads the Python fixtures, including a
  diffusion recording opened in Replay Studio.
- **`test_diffusers_cloud.py`** runs against the real diffusers and is skipped locally. It checks:
  - recording is bit-identical to the plain loop;
  - `varNext` equals `_get_variance`;
  - the recorded μ is the mean diffusers used.

## Notation (diffusers → book, Lai et al. Sec. 2.2)

- The step index is i = t + 1 (diffusers `t` is 0-based).
- `alphas_cumprod[t]` = ᾱ_i², and `betas[t]` = β_i².
- `pred_original_sample` = x_φ, clipped when `clip_sample` is set, so `identityResidual` then shows
  the effect of the clipping.
- The variance ("fixed_small") = σ²(i) = β̃_i² (Eq. 2.2.5).
