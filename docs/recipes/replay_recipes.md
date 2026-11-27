# Replay recipes — record once, replay in MATLAB

These recipes cover OceanReplay (MATLAB `+OceanReplay` and Python `Python/ocean_replay`) and Replay
Studio. Design and format: `Planning/System/Replay/`.

## 1. Open a recording

```matlab
addpath('C:\Ocean Library\Matlab code practice');
s = OceanViz.App.Replay.ReplayStudio("C:\path\to\recording.h5");   % or ReplayStudio.open() for a file picker
s.setCoord("step", 500);   s.play();   s.pause();                     % also: first / last / stepForward / stepBack
```

**Controls:**
- ▶ Play / ⏸ Pause, the speed menu, Space to play or pause, ← → to step.
- ⓘ **Details**: equations, how each view was made, references.
- A card's ⓘ: that card's explanation only.

**What each card shows:** the NEAREST STORED point, with its real axis value (e.g. "stored at step
500 · nearest stored to 507"). Values are never blended.

## 2. Record in MATLAB (any domain)

```matlab
S = OceanReplay.Core.Sampling;
spec = OceanReplay.Core.Spec() ...
    .addGroup("numbers", 0, ["alphaBar" "sigma" "epsRms"]) ...                        % every step
    .addGroup("images",  1, ["x" "eps" "x0hat" "mu"], struct('step', S.every(20))) ...
    .addGroup("detail",  2, ["x" "eps" "x0hat" "mu"], struct('step', S.window(50, 1))) ...
    .addView("occlusion", "occlusion", struct('size', 4, 'stride', 2), struct('step', S.list([500 50])));
d = OceanDL.Generative.Inspect.DiffusionRecordable(predict, schedule, sampler, [16 16 1], 'Seed', 7);
OceanReplay.Record.Recorder.run(d, spec, "run.h5", 'Title', "my run");
```

Ready-made launchers:
- `Experiments/Scripts/Replay/exp_replay_record_diffusion.m`: the trained 16×16 AffectNet DDPM.
- `Experiments/Scripts/Replay/exp_replay_record_kmeans.m`: k-means (ESL 14.3.6).
- `Experiments/Scripts/Replay/exp_replay_convert_mandelbrot.m`: the legacy NN training checkpoint.

**From the Model Lab:** Reverse process tab → set it up → **Save recording…**. It records exactly the
run that "Run reverse process" shows, then opens Replay Studio.

## 3. Make a new domain recordable

Implement `OceanReplay.Record.IRecordable`:

```matlab
axes = axesInfo(obj)            % OceanReplay.Core.Axis (empty Values = open axis)
[channels, views] = catalog(obj)% ChannelInfo / ViewInfo (Symbol, SymbolLatex, Description, HowMade, Reference, InTexts)
out = run(obj, recorder)        % the real loop; at each point: recorder.point(coords, struct('x', @() x, ...))
```

- **Providers are lazy** (function handles), so channels nobody asked for cost nothing.
- **Prove the recording changes nothing:** `OceanReplay.Record.verifyInvariance(@() MyDomain(), spec)`.
- **Optional drawing:** a domain layout in `+OceanViz/+App/+Replay/+Layouts`. Otherwise
  `DefaultLayout` draws every kind.

## 4. Record in Python / on Google Cloud (no local PyTorch needed)

- **Package:** `Python/ocean_replay`, with a README.
- **The trained AffectNet 64×64 DDPM in Cloud Shell** (free, CPU, about 20 min, no training):

  ```bash
  gcloud storage cp gs://your-bucket/code/ocean_replay/cloud/cloudshell_record.sh ~/ && bash ~/cloudshell_record.sh your-bucket affectnet64_uncond 7
  ```

  It records the archive to `gs://your-bucket/recordings/<run-id>/` and cuts the small replay
  file (`…_small.h5`, about 11 MB). Download it with `cloudshell download <file>`.

## 5. Extract, validate

```matlab
OceanReplay.IO.extract("archive.h5", "extract.json", "small.h5");   % same format, fewer points
issues = OceanReplay.IO.Schema.validate("small.h5");                % [] = valid
```

```bash
python -m ocean_replay extract archive.h5 --spec extract.json --out small.h5
python -m ocean_replay validate small.h5
```

## 6. Tests

```matlab
runtests('OceanTest.OceanReplay')                         % core, IO, recorder, domains, MATLAB <-> Python parity
runtests('OceanTest.OceanViz.ReplayStudioTest')           % GUI
```

```bash
cd Python && python -m unittest discover -s ocean_replay/tests -t .
```
