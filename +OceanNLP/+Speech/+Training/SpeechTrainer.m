classdef SpeechTrainer < handle
    % SPEECHTRAINER  Epoch-based trainer for ISpeechRecognizer models, exactly resumable.
    %
    %   tr = OceanNLP.Speech.Training.SpeechTrainer(model, trainBatcher, 'ValBatcher', vb, ...
    %          'Epochs', 10, 'LR', 1e-3, 'WeightDecay', 0.01, 'WarmupSteps', [], 'MinLR', [], ...
    %          'Clip', 5, 'Seed', 0, 'Decoder', "greedy", 'BestCheckpoint', "", 'LogFile', "", 'Verbose', true)
    %   tr.fit()                  train until Epochs are done (validating after every epoch)
    %   tr.trainSteps(n)          exactly n optimisation steps (no validation)
    %   r  = tr.validate()        val loss + WER/CER (CorpusEvaluator) on ValBatcher
    %   st = tr.state()           everything needed to resume (see SpeechCheckpoint)
    %   tr2 = SpeechTrainer.resume(ck, trainBatcher, 'ValBatcher', vb)   from SpeechCheckpoint.load
    %
    %   ONE STEP s (global step counter, 1-based):
    %     lr_s  = WarmupCosine(LR, WarmupSteps, TotalSteps = Epochs * numBatches, MinLR).at(s)
    %     g     = grad_theta model.loss(batch)       (CTC / model objective, autograd)
    %     g    <- g * min(1, Clip / ||g||_2)           (OceanDL.nn.clipGradNorm, global norm)
    %     theta <- AdamW(theta, g; lr_s, beta = (0.9, 0.999), eps = 1e-8, lambda)
    %   with lambda = WeightDecay, except 0 for biases and LayerNorm parameters
    %   (OceanDL.optim.ModuleOptimizer.forModule default rule).
    %   Batches: TrainBatcher epoch e (0-based) in its deterministic order
    %   (SpeechBatcher: a pure function of Seed + e).
    %
    %   RANDOMNESS: the trainer owns a private copy of the global RNG state
    %   (twister seeded with Seed). Each step swaps it in, runs (dropout draws come
    %   from the global generator), stores it back and restores the caller's RNG --
    %   so training is reproducible regardless of what else the session draws, and
    %   the stored state makes resume exact.
    %   VALIDATION (after each epoch, eval mode, no graph): batch-size-weighted mean
    %   loss, greedy (or 'Decoder') transcripts -> CorpusEvaluator WER and CER.
    %   Model selection: lowest (val WER, then val loss); the best weights are kept
    %   in BestWeights and, if BestCheckpoint is set, saved with SpeechCheckpoint.
    %   Validation draws no random numbers and does not touch the training RNG.
    %
    %   EXACT RESUME: state() = Step, Epoch, NextBatch, RngState, History, best
    %   score, options, train-batcher config, and the per-parameter optimizer
    %   objects. resume() rebuilds the same schedule and batch order and continues
    %   bit-for-bit (tested: N steps + save/load + N steps == 2N steps on CPU).
    %   Mid-epoch resume with an ENABLED AudioAugment is refused
    %   (':AugmentResume'): its private stream position cannot be restored.

    properties (SetAccess = private)
        Model
        TrainBatcher
        ValBatcher
        Options struct
        Optimizer
        Schedule
        Evaluator
        TotalSteps (1,1) double
        Step (1,1) double = 0
        Epoch (1,1) double = 0          % current 0-based epoch of the train batcher
        NextBatch (1,1) double = 1      % next batch index within Epoch
        RngState
        History struct
        BestScore (1,2) double = [inf inf]
        BestEpoch (1,1) double = NaN
        BestWeights = []
    end

    methods
        function obj = SpeechTrainer(model, trainBatcher, opts)
            arguments
                model (1,1) OceanNLP.Speech.Models.ISpeechRecognizer
                trainBatcher (1,1) OceanNLP.Speech.Data.SpeechBatcher
                opts.ValBatcher = []
                opts.Epochs (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.LR (1,1) double {mustBePositive} = 1e-3
                opts.WeightDecay (1,1) double {mustBeNonnegative} = 0.01
                opts.WarmupSteps double {mustBeScalarOrEmpty} = []
                opts.MinLR double {mustBeScalarOrEmpty} = []
                opts.Clip (1,1) double {mustBeNonnegative} = 5
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.Decoder (1,1) string {mustBeMember(opts.Decoder, ["greedy", "beam"])} = "greedy"
                opts.BestCheckpoint (1,1) string = ""
                opts.LogFile (1,1) string = ""
                opts.Verbose (1,1) logical = true
            end
            if ~isempty(opts.ValBatcher) && ~isa(opts.ValBatcher, 'OceanNLP.Speech.Data.SpeechBatcher')
                error('OceanNLP:Speech:Training:SpeechTrainer:ValBatcher', 'ValBatcher must be a SpeechBatcher or [].');
            end
            if strlength(opts.BestCheckpoint) > 0 && isempty(opts.ValBatcher)
                error('OceanNLP:Speech:Training:SpeechTrainer:BestNeedsValidation', ...
                    'BestCheckpoint is chosen by validation WER: pass a ValBatcher (otherwise no checkpoint would ever be written).');
            end
            obj.Model = model;
            obj.TrainBatcher = trainBatcher;
            obj.ValBatcher = opts.ValBatcher;
            K = trainBatcher.numBatches();
            obj.TotalSteps = opts.Epochs * K;
            if isempty(opts.WarmupSteps), opts.WarmupSteps = max(1, round(0.05 * obj.TotalSteps)); end
            if isempty(opts.MinLR), opts.MinLR = opts.LR / 20; end
            obj.Options = rmfield(opts, 'ValBatcher');
            obj.Schedule = OceanDL.optim.WarmupCosine(opts.LR, 'WarmupSteps', opts.WarmupSteps, ...
                'TotalSteps', obj.TotalSteps, 'MinLR', opts.MinLR);
            obj.Optimizer = OceanDL.optim.ModuleOptimizer.forModule(model, ...
                @(wd) OceanStats.Optim.AdamWOptimizer('LR', opts.LR, 'WeightDecay', wd), 'WeightDecay', opts.WeightDecay);
            obj.Evaluator = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', model.Vocabulary.Normalizer);
            prev = rng;
            rng(opts.Seed, 'twister');
            obj.RngState = rng;
            rng(prev);
            obj.History = OceanNLP.Speech.Training.SpeechTrainer.emptyHistory();
            trainBatcher.setEpoch(0);
        end

        function fit(obj)
            obj.log(sprintf('training %s: %d epochs x %d batches = %d steps (from step %d)', ...
                class(obj.Model), obj.Options.Epochs, obj.TrainBatcher.numBatches(), obj.TotalSteps, obj.Step));
            while obj.Step < obj.TotalSteps
                obj.step();
                if obj.NextBatch > obj.TrainBatcher.numBatches()
                    obj.endEpoch();
                end
            end
            if obj.NextBatch > obj.TrainBatcher.numBatches() && numel(obj.History.Epochs) < obj.Epoch + 1
                obj.endEpoch();
            end
        end

        function trainSteps(obj, n)
            arguments
                obj
                n (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            for i = 1:n
                obj.step();
                if obj.NextBatch > obj.TrainBatcher.numBatches()
                    obj.endEpoch();             % same epoch bookkeeping as fit()
                end
            end
        end

        function L = step(obj)
            % STEP  One optimisation step on the next training batch.
            if obj.Step >= obj.TotalSteps
                error('OceanNLP:Speech:Training:SpeechTrainer:Finished', 'All %d steps are done.', obj.TotalSteps);
            end
            if obj.NextBatch > obj.TrainBatcher.numBatches()
                obj.Epoch = obj.Epoch + 1;
                obj.NextBatch = 1;
                obj.TrainBatcher.setEpoch(obj.Epoch);
            end
            t0 = tic;
            batch = obj.TrainBatcher.batch(obj.NextBatch);
            % Dropout uses MATLAB's global generator.  Temporarily install the
            % trainer-owned state, then restore the user's session RNG on exit.
            % Saving RngState after the step is what makes resume bit-identical.
            prev = rng;
            rng(obj.RngState);
            restore = onCleanup(@() rng(prev));
            s = obj.Step + 1;
            lr = obj.Schedule.at(s);
            obj.Model.train();
            obj.Optimizer.setLR(lr);
            obj.Optimizer.zeroGrad();
            % Autograd builds the graph in model.loss; backward accumulates each
            % parameter gradient, clipping bounds the global norm, and AdamW then
            % consumes those gradients exactly once.
            Lt = obj.Model.loss(batch);
            Lt.backward();
            gn = NaN;
            if obj.Options.Clip > 0
                gn = OceanDL.nn.clipGradNorm(obj.Optimizer.Params, obj.Options.Clip);
            end
            obj.Optimizer.step();
            obj.RngState = rng;
            clear restore
            L = double(gather(Lt.Data));
            obj.Step = s;
            obj.NextBatch = obj.NextBatch + 1;
            h = obj.History;
            h.StepLoss(s) = L;
            h.StepLR(s) = lr;
            h.StepGradNorm(s) = gn;
            h.StepEpoch(s) = obj.Epoch;
            h.StepSeconds(s) = toc(t0);
            obj.History = h;
        end

        function r = validate(obj)
            % VALIDATE  Weighted val loss, WER and CER (no graph, eval mode, no RNG draws).
            r = struct('Loss', NaN, 'WER', NaN, 'CER', NaN, 'Result', []);
            vb = obj.ValBatcher;
            if isempty(vb)
                return;
            end
            wasTraining = obj.Model.Training;
            obj.Model.eval();
            % Validation is observational: restore mode afterward and disable the
            % gradient graph so it cannot alter training state or waste memory.
            restore = onCleanup(@() obj.Model.train(wasTraining));
            grad = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            totL = 0; totN = 0; refs = strings(1, 0); hyps = strings(1, 0); ids = strings(1, 0); spk = strings(1, 0);
            for k = 1:vb.numBatches()
                b = vb.batch(k);
                N = numel(b.Lengths);
                L = obj.Model.loss(b);
                totL = totL + N * double(gather(L.Data));
                totN = totN + N;
                tr = obj.Model.transcribe(b, 'Decoder', obj.Options.Decoder);
                refs = [refs, b.Text]; %#ok<AGROW>
                hyps = [hyps, tr.texts()]; %#ok<AGROW>
                ids = [ids, b.Ids]; %#ok<AGROW>
                spk = [spk, b.Speakers]; %#ok<AGROW>
            end
            res = obj.Evaluator.evaluate(refs, hyps, 'Ids', ids, 'Speakers', spk);
            r = struct('Loss', totL / totN, 'WER', res.WER.Rate, 'CER', res.CER.Rate, 'Result', res);
        end

        function st = state(obj)
            % STATE  Resumable training state (the optimizer objects are live handles;
            %   SpeechCheckpoint.save serialises them).
            named = obj.Model.namedParameters();
            st = struct('Type', "SpeechTrainerState", 'Version', 1, 'Step', obj.Step, 'Epoch', obj.Epoch, ...
                'NextBatch', obj.NextBatch, 'TotalSteps', obj.TotalSteps, 'RngState', obj.RngState, ...
                'History', obj.History, 'BestScore', obj.BestScore, 'BestEpoch', obj.BestEpoch, ...
                'BestWeights', {obj.BestWeights}, ...
                'Options', obj.Options, 'BatcherConfig', obj.TrainBatcher.config(), ...
                'NumTrainItems', obj.TrainBatcher.Dataset.length(), ...
                'ParamNames', {{named.Name}}, ...
                'Optimizers', {obj.Optimizer.Optimizers});
        end

        function loadState(obj, st)
            % LOADSTATE  Restore a state() (normally from SpeechCheckpoint.load) into this trainer.
            arguments
                obj
                st (1,1) struct
            end
            if ~isfield(st, 'Type') || string(st.Type) ~= "SpeechTrainerState"
                error('OceanNLP:Speech:Training:SpeechTrainer:State', 'Not a SpeechTrainer state.');
            end
            bc = obj.TrainBatcher.config();
            sc = st.BatcherConfig;
            fields = ["BatchSize", "Shuffle", "Seed", "SortByLength"];
            for f = fields
                if ~isequal(bc.(f), sc.(f))
                    error('OceanNLP:Speech:Training:SpeechTrainer:BatcherMismatch', ...
                        'Train batcher %s is %s, the state was saved with %s.', f, mat2str(bc.(f)), mat2str(sc.(f)));
                end
            end
            if obj.TrainBatcher.Dataset.length() ~= st.NumTrainItems || obj.TotalSteps ~= st.TotalSteps
                error('OceanNLP:Speech:Training:SpeechTrainer:DataMismatch', ...
                    'Training set size / total steps differ from the saved state (%d/%d vs %d/%d).', ...
                    obj.TrainBatcher.Dataset.length(), obj.TotalSteps, st.NumTrainItems, st.TotalSteps);
            end
            named = obj.Model.namedParameters();
            if ~isequal({named.Name}, st.ParamNames)
                error('OceanNLP:Speech:Training:SpeechTrainer:ParamMismatch', 'Model parameters differ from the saved state.');
            end
            aug = obj.TrainBatcher.Dataset.Augment;
            if st.NextBatch > 1 && st.NextBatch <= obj.TrainBatcher.numBatches() && ~isempty(aug) && aug.isEnabled()
                error('OceanNLP:Speech:Training:SpeechTrainer:AugmentResume', ...
                    'Cannot resume mid-epoch with augmentation enabled (augmenter stream position is not restorable).');
            end
            queue = containers.Map({'k'}, {0});
            saved = st.Optimizers;
            obj.Optimizer = OceanDL.optim.ModuleOptimizer({named.Param}, @() popSaved(queue, saved));
            obj.Step = st.Step;
            obj.Epoch = st.Epoch;
            obj.NextBatch = st.NextBatch;
            obj.RngState = st.RngState;
            obj.History = st.History;
            obj.BestScore = st.BestScore;
            obj.BestEpoch = st.BestEpoch;
            if isfield(st, 'BestWeights')               % absent in states saved before 2026-10-07
                obj.BestWeights = st.BestWeights;
            end
            obj.TrainBatcher.setEpoch(obj.Epoch);
        end

        function restoreBest(obj)
            % RESTOREBEST  Load the best validation weights into the model.
            if isempty(obj.BestWeights)
                error('OceanNLP:Speech:Training:SpeechTrainer:NoBest', 'No validated epoch yet.');
            end
            obj.Model.loadStateDict(obj.BestWeights);
        end
    end

    methods (Static)
        function tr = resume(ck, trainBatcher, opts)
            % RESUME  Trainer for ck.Model (from SpeechCheckpoint.load) continuing ck.TrainerState.
            arguments
                ck (1,1) struct
                trainBatcher (1,1) OceanNLP.Speech.Data.SpeechBatcher
                opts.ValBatcher = []
                opts.LogFile = []
                opts.BestCheckpoint = []
            end
            o = ck.TrainerState.Options;
            if ~isempty(opts.LogFile), o.LogFile = opts.LogFile; end
            if ~isempty(opts.BestCheckpoint), o.BestCheckpoint = opts.BestCheckpoint; end
            args = namedargs2cell(o);
            tr = OceanNLP.Speech.Training.SpeechTrainer(ck.Model, trainBatcher, 'ValBatcher', opts.ValBatcher, args{:});
            tr.loadState(ck.TrainerState);
        end

        function h = emptyHistory()
            h = struct('StepLoss', zeros(1, 0), 'StepLR', zeros(1, 0), 'StepGradNorm', zeros(1, 0), ...
                'StepEpoch', zeros(1, 0), 'StepSeconds', zeros(1, 0), ...
                'Epochs', struct('Epoch', {}, 'Steps', {}, 'TrainLoss', {}, 'ValLoss', {}, 'ValWER', {}, ...
                'ValCER', {}, 'LR', {}, 'TrainSeconds', {}, 'ValSeconds', {}));
        end
    end

    methods (Access = private)
        function endEpoch(obj)
            t0 = tic;
            v = obj.validate();
            h = obj.History;
            inEpoch = h.StepEpoch == obj.Epoch;
            e = struct('Epoch', obj.Epoch + 1, 'Steps', obj.Step, 'TrainLoss', mean(h.StepLoss(inEpoch)), ...
                'ValLoss', v.Loss, 'ValWER', v.WER, 'ValCER', v.CER, 'LR', h.StepLR(end), ...
                'TrainSeconds', sum(h.StepSeconds(inEpoch)), 'ValSeconds', toc(t0));
            h.Epochs(end + 1) = e;
            obj.History = h;
            improved = ~isnan(v.WER) && (v.WER < obj.BestScore(1) || (v.WER == obj.BestScore(1) && v.Loss < obj.BestScore(2)));
            if improved
                obj.BestScore = [v.WER, v.Loss];
                obj.BestEpoch = obj.Epoch + 1;
                obj.BestWeights = obj.Model.stateDict();
                if strlength(obj.Options.BestCheckpoint) > 0
                    OceanNLP.Speech.Training.SpeechCheckpoint.save(obj.Options.BestCheckpoint, obj.Model, obj.state());
                end
            end
            obj.log(sprintf('epoch %2d  step %5d  train loss %.4f  val loss %.4f  val WER %.4f  CER %.4f  lr %.2e  (%.0f s train, %.0f s val)%s', ...
                e.Epoch, e.Steps, e.TrainLoss, e.ValLoss, e.ValWER, e.ValCER, e.LR, e.TrainSeconds, e.ValSeconds, ...
                repmat('  *best', 1, improved)));
        end

        function log(obj, msg)
            if obj.Options.Verbose
                fprintf('%s\n', msg);
            end
            if strlength(obj.Options.LogFile) > 0
                fid = fopen(obj.Options.LogFile, 'a');
                if fid > 0
                    fprintf(fid, '%s  %s\n', string(datetime('now', 'Format', 'HH:mm:ss')), msg);
                    fclose(fid);
                end
            end
        end
    end
end

function o = popSaved(queue, saved)
% Hand out the saved optimizer objects in parameter order (containers.Map is a handle).
k = queue('k') + 1;
queue('k') = k;
o = saved{k};
end
