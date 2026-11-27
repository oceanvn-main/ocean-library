% OceanNLP.Speech  Inspectable speech and automatic-speech-recognition system.
%
% PIPELINE MAP
%   Audio source
%       -> Core.Waveform                         sampled audio + provenance
%       -> Inference.AudioConditioner            mono/rate/level preparation
%       -> Inference.SpeechActivityDetector      speech regions / long-file pieces
%       -> Features.IFeatureTransform            frame-level acoustic observations
%       -> Data.SpeechBatcher                     padded F x N x T model batches
%       -> Models.ISpeechRecognizer              acoustic encoder + output head
%       -> Decoding                              token search / language-model fusion
%       -> Core.Transcript                       text + decoder provenance
%       -> Evaluation / Inspect                  metrics and process evidence
%
% CORE CONTRACTS
%   Core.Waveform            S x C samples, sample rate, source timeline and history.
%   Core.FrameGrid           One shared definition of frame starts, times and FFT bins.
%   Core.FeatureSequence     F x T features with time/frequency axes and configuration.
%   Core.Transcript          Recognition result; unknown alignments/confidence stay empty.
%   Core.IFeatureTransform   Strategy boundary used by datasets and recognizers.
%
% FEATURE FRONT END
%   Features.LogMel          Waveform -> framed spectrum -> mel energy -> log features.
%   Features.MFCC            LogMel followed by an orthonormal DCT-II and optional lifter.
%   Features.DeltaFeatures   Static features plus temporal delta orders.
%   Features.EnergyContour   One short-time log-energy value per shared frame.
%   Features.PitchContour    YIN F0 and voicing strength on the shared frame grid.
%   Features.FeaturePipeline Ordered composition presented as one feature transform.
%
% DATA AND LABELS
%   Data.SpeechManifest      Corpus records and train/validation/test split membership.
%   Data.AudioDataset        Loads Waveform, features, normalized text and token IDs.
%   Data.SpeechBatcher       Pads variable-length examples and records valid lengths.
%   Data.SpeechVocabulary    Token/string conversion and special-token contract.
%   Data.TranscriptNormalizer Reproducible text normalization before scoring/training.
%   Data.AudioAugment        Seeded waveform and feature-domain augmentation.
%
% MODELS AND DECODING
%   Models.CTCRecognizer            Encoder + CTC head; alignment-free training.
%   Models.EncoderDecoderRecognizer Encoder + autoregressive attention decoder.
%   Models.Encoders                 Swappable convolutional/recurrent/Conformer encoders.
%   Models.Pretrain                 HuBERT-style unit discovery and masked prediction.
%   Decoding.CTCPrefixBeamDecoder   Greedy/prefix-beam CTC search, optional n-gram LM.
%   Decoding.AttentionBeamDecoder   Cached autoregressive beam search and CTC reranking.
%
% TRAIN, EVALUATE, INSPECT, INFER
%   Training.SpeechTrainer          Deterministic optimisation and exact resume state.
%   Training.SpeechCheckpoint       Model/config/trainer persistence boundary.
%   Evaluation                     Edit alignment, WER, CER and corpus breakdowns.
%   Inspect                        Records real intermediate artifacts, never mock stages.
%   Inference.RecognitionPipeline  One-call conditioning, detection, recognition and merge.
%
% SHAPE LEGEND
%   S = audio samples, C = audio channels, F = feature bins/channels,
%   N = batch items, T = time frames, K = vocabulary classes.
%   Public features are F x T. Model batches are F x N x T.
%
% DESIGN RULES
%   - Resampling is explicit; feature transforms never silently change sample rate.
%   - Padding is accompanied by valid lengths/masks and must not affect loss/decoding.
%   - Saved configuration must rebuild the exact feature/model architecture.
%   - Decoder scores are search scores, not calibrated confidence probabilities.
%   - Timed transcript segments are stored only when a real timing process produced them.

