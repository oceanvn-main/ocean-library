# Speech & ASR recipes (`OceanNLP.Speech`)

Native speech-feature, ASR training, decoding, evaluation and inspection stack built from
*Speech and Language Processing* (Jurafsky & Martin, 3rd ed. draft) Ch. 14–15.
Design: `Planning/NLP/Speech_and_ASR_System_Implementation_Plan.md`; frozen interfaces:
`Planning/NLP/Speech_ASR_Phase0_Contracts.md`.

## 1. System diagram

```text
                 Data/Datasets/Speech/FSDD (real audio, CC BY-SA 4.0)   MATLAB Audio Toolbox speech samples (unlabeled)
                                   │                                              │
 ┌─────────────── OceanNLP.Speech.Data ───────────────┐                           │
 │ SpeechManifest ─ canonicalFSDDSplit / splitBySpeaker│                           │
 │ UtteranceComposer (multi-word from real recordings) │                           │
 │ TranscriptNormalizer · SpeechVocabulary (train-only)│                           │
 │ AudioDataset (lazy, resample, cache) ─► SpeechBatcher (CBT pad + Mask + Lengths)│
 │ AudioAugment (gain/noise/SpecAugment, seeded)       │                           │
 └────────────┬────────────────────────────────────────┘                           │
              │ Waveform (S x C, fs, provenance)                                   │
 ┌────────────▼──────────── OceanNLP.Speech.Features ──────────────┐               │
 │ FrameGrid ─► LogMel ─► MFCC ─► DeltaFeatures   EnergyContour      │◄── OceanMath.Signal:
 │ PitchContour (YIN)   FeaturePipeline (IFeatureTransform strategy) │    FrameSignal, ResampleSignal,
 └────────────┬─────────────────────────────────────────────────────┘    MelFilterBank, CepstralTransform,
              │ FeatureSequence (F x T, times, Hz, config)                SpectralAnalyzer.framePowerSpectrum
 ┌────────────▼──────────── OceanNLP.Speech.Models ─────────────────────────────────▼───────┐
 │ ISpeechRecognizer (forward / loss / transcribe / config)   RecognizerFactory.fromConfig  │
 │  ├ CTCRecognizer ── encoders: ConvEncoder · RecurrentEncoder · ConformerEncoder · HuBERT │
 │  └ EncoderDecoderRecognizer ── ConformerEncoder + OceanDL.Transformer.DecoderBlock       │
 │ +Pretrain: UnitCodebook (k-means) ─► HuBERTModel (masked unit prediction) ─► MaskedUnitTrainer
 └────────────┬──────────────────────────────────────────────────────────────────────────────┘
              │ logits K x N x T'          ▲ OceanDL: Conv1d, SequenceMask, CTCLoss, CTCBeamSearch,
 ┌────────────▼─── Decoding ───────┐       │         ConformerBlock, DecoderBlock, MultiHeadAttention,
 │ CTCPrefixBeamDecoder (+TokenNGramLM)    │         ModuleOptimizer, WarmupCosine
 │ AttentionBeamDecoder            │       │
 └────────────┬────────────────────┘  ┌────┴──── Training ─────────────────────────────┐
              │ Transcript            │ SpeechTrainer (AdamW, warmup-cosine, clip,      │
 ┌────────────▼─── Evaluation ─────┐  │ val WER/CER per epoch, best ckpt)               │
 │ EditAlignment · WER · CER       │  │ SpeechCheckpoint (exact resume)                 │
 │ CorpusEvaluator (per-utt/speaker)│ └─────────────────────────────────────────────────┘
 └────────────┬────────────────────┘
              ▼
 OceanNLP.Speech.Inspect (SpeechTrace, SpeechProcessAdapter → MemoryReader)
              ▼
 OceanViz: Core.plot("SpeechWaveform" | "Spectrogram" | "MelFilterBank" | "SpeechFeatures" |
           "PitchContour" | "CTCPosterior" | "BeamCandidates" | "ErrorAlignment")
           App.NLP.SpeechExplorer · App.NLP.ProcessExplorer
```

Dependency direction: `OceanViz → OceanNLP.Speech → OceanDL → OceanMath`; algorithms never call
OceanViz; Audio Toolbox / librosa / jiwer / torch appear only in tests and validation scripts.

## 2. Workflow

1. **Corpus** — `SpeechManifest.fromFSDD()` reads every file's real duration/rate;
   `canonicalFSDDSplit()` holds out speaker `theo`; `validate()` and `audit()` before use.
2. **Text side** — `TranscriptNormalizer`, then `SpeechVocabulary.fit(trainTranscripts)` (indices
   1 blank, 2 pad, 3 bos, 4 eos, 5 unk, then units; `|` = word boundary for chars).
3. **Features** — choose an `IFeatureTransform` (`LogMel`, `MFCC`, `FeaturePipeline({MFCC, DeltaFeatures})`).
4. **Dataset/batches** — `AudioDataset` (lazy, cached) → `SpeechBatcher` (`X` F x N x T, `Lengths`, `Mask`).
5. **Model** — encoders + recogniser; or pretrain a `HuBERTModel` on unlabeled audio first and
   fine-tune it as the encoder of a `CTCRecognizer`.
6. **Train** — `SpeechTrainer.fit()`; checkpoints via `SpeechCheckpoint` (exact resume).
7. **Decode** — `transcribe(waveform, 'Decoder', "greedy"|"beam", ...)` → `Transcript`.
8. **Evaluate** — `CorpusEvaluator.evaluate(refs, hyps, 'Speakers', ...)` (WER/CER identical to jiwer).
9. **Inspect** — `SpeechProcessAdapter.run(model, waveform)` → reader of every real intermediate;
   plot with `OceanViz.Core`, browse in `OceanViz.App.NLP.SpeechExplorer`.

## 3. Pipeline example (CTC, end to end)

```matlab
import OceanNLP.Speech.*
man   = Data.SpeechManifest.fromFSDD();                 % real FSDD, root-relative paths
split = man.canonicalFSDDSplit();
train = split.bySplit("train");   test = split.bySplit("test");    % test = speaker theo
vocab = Data.SpeechVocabulary.fit(train.Entries.Transcript, 'Unit', "char");
feat  = Features.LogMel(8000, 'NumMels', 40);           % 25 ms Hann, 10 ms hop
dsTr  = Data.AudioDataset(train, feat, 'Vocabulary', vocab, 'SampleRate', 8000);
dsTe  = Data.AudioDataset(test,  feat, 'Vocabulary', vocab, 'SampleRate', 8000);

conv  = Models.Encoders.ConvEncoder(40, 'Channels', [128 128], 'KernelSizes', [5 3], 'Strides', [2 1]);
rnn   = Models.Encoders.RecurrentEncoder(conv.OutputDim, 'HiddenSize', 128, 'Cell', "gru");
model = Models.CTCRecognizer(feat, vocab, {conv, rnn});

val   = split.bySplit("validation");                     % speaker-dependent, model selection only
dsVa  = Data.AudioDataset(val, feat, 'Vocabulary', vocab, 'SampleRate', 8000);
bTr   = Data.SpeechBatcher(dsTr, 'BatchSize', 32, 'Shuffle', true, 'Seed', 0, 'SortByLength', true);
bVa   = Data.SpeechBatcher(dsVa, 'BatchSize', 32, 'Shuffle', false);
trainer = Training.SpeechTrainer(model, bTr, 'ValBatcher', bVa, 'Epochs', 10, 'LR', 2e-3, ...
    'BestCheckpoint', "ctc.mat");                        % best = lowest validation WER (needs ValBatcher)
trainer.fit();
trainer.restoreBest();

item = dsTe.item(1);                                                % real held-out utterance
w  = item.Waveform;
tr = model.transcribe(w, 'Decoder', "beam", 'BeamWidth', 8);       % Transcript (Text, Score, Candidates)
bTe = Data.SpeechBatcher(dsTe, 'BatchSize', 32, 'Shuffle', false);
hyps = strings(0, 1);
for k = 1:bTe.numBatches()
    hyps = [hyps; model.transcribe(bTe.batch(k)).texts()'];          %#ok<AGROW>
end
report = Evaluation.CorpusEvaluator().evaluate(test.Entries.Transcript, hyps, ...
    'Ids', test.Entries.Id, 'Speakers', test.Entries.Speaker);     % WER, CER, per-speaker

[tr2, reader] = Inspect.SpeechProcessAdapter.run(model, w, 'Decoder', "beam");   % captured stages
viz = OceanViz.Core('Theme', 'Dark');                                  % then viz.plot(ax, "CTCPosterior", ...)
```

The complete, executed versions are the experiment scripts (consumers only):

| Script | What it runs | Report |
| --- | --- | --- |
| `Experiments/Scripts/Speech/validate_features_external.m` | Ocean features vs librosa / MATLAB Audio Toolbox on real audio | `Results/Speech/feature_validation.md` |
| `Experiments/Scripts/Speech/validate_wer_jiwer.m` | Ocean WER/CER vs jiwer | `Results/Speech/wer_validation.md` |
| `Experiments/Scripts/Speech/exp_speech_ctc_fsdd.m` | CTC baseline, greedy / beam / beam+LM | `Results/Speech/ctc_fsdd_report.md` |
| `Experiments/Scripts/Speech/exp_speech_encdec_fsdd.m` | Conformer encoder–decoder | `Results/Speech/encdec_fsdd_report.md` |
| `Experiments/Scripts/Speech/exp_speech_hubert_fsdd.m` | HuBERT pretraining + CTC fine-tuning vs scratch | `Results/Speech/hubert_fsdd_report.md` |
| `Experiments/Scripts/Speech/exp_speech_inspect_demo.m` | Inspection figures for a held-out utterance | `Results/Speech/figures/*.png` |

Set `OCEAN_SPEECH_QUICK=1` for a tiny debugging run of the CTC script (not a result).

## 4. Tests

```matlab
r = runtests({'OceanTest.OceanNLP.Speech', 'OceanTest.OceanViz.Speech'}, 'IncludeSubpackages', true);
```
plus the speech primitives in `OceanTest.OceanMath.Signal.{FrameSignal,MelFilterBank,CepstralTransform,ResampleSignal}Test`,
`OceanTest.OceanDL.nn.Conv1dTest`, `OceanTest.OceanDL.Layout`, `OceanTest.OceanDL.Recurrent`, and the new
`OceanTest.OceanDL.Transformer` block tests.

## 5. Results

See the reports in `Results/Speech/`; the summary table is kept in the plan's living log (§9).

## 6. Speech Pipeline Studio (GUI)

A MATLAB app that runs the real pipeline (`OceanNLP.Speech.Inference.RecognitionPipeline`) on an FSDD
utterance, a WAV file or the microphone and shows every stage. Plan: `Planning/NLP/SpeechPipelineStudio-plan.md`.

```matlab
addpath('C:\Ocean Library\Matlab code practice');
app = OceanViz.App.NLP.SpeechStudio.launch();
```

* **Sidebar:** input (FSDD split / utterance / random held-out, WAV browse, microphone record + level meter,
  typed reference for WAV / microphone), model (checkpoints found by `ModelCatalog`; decoder, beam width;
  LM weight / bonus / order for CTC beam search; CTC rescoring weight for the encoder–decoder; controls that
  do not apply to the selected model are hidden), robustness (`AudioConditioner` / `SpeechActivityDetector`
  options; `auto` segmentation splits only input longer than the max segment and can split mid-word on fast
  continuous speech).
* **▶ Run** computes everything once (progress dialog); changing any setting marks the run stale and greys
  out the stepper until Run. **♪ Play** plays the conditioned audio with a moving playhead.
* **Stages:** 1. input & conditioning 2. frames 3. spectrum 4. mel 5. features 6. encoder 7. output
  probabilities 8. decoding 9. transcript & evaluation (stages 2–8 are disabled when no speech was
  detected; stage 9 then says so). Each stage shows LaTeX notation, the shapes / parameters of this
  run, the Ocean class + file that computed them, and an ⓘ drawer with the explanation. Hover a time axis
  to move the linked cursor; click it to select a frame (frames / spectrum stages).
* **Workflow:** corpus & split (`SpeechManifest.audit`), training curves (checkpoint History), model
  comparison (`ModelCatalog.list` + tables of the generated reports), validation & tests.
* Headless / scripted: `app = OceanViz.App.NLP.SpeechStudio.launch('Visible', false)`;
  `app.ViewModel.selectUtterance("3_theo_10"); app.run(); app.ViewModel.selectStage(7); app.stage(7).Rows`;
  `app.close()`. (Stages draw when selected, so select a stage before reading its table.)
  Inject microphone samples without a device:
  `rec = OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(x, fs); launch('Recorder', rec)`.
* Tests: `runtests('OceanTest.OceanViz.SpeechStudio', 'IncludeSubpackages', true)`.
  Screenshots: `Results/Speech/figures/studio_*.png`.
* Honest limits: the models know only the FSDD digit words (3 training speakers, 8 kHz); the app says so
  for WAV / microphone input.
