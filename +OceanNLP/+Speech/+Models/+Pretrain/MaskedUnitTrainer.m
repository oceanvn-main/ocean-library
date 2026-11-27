classdef MaskedUnitTrainer < handle
    % MASKEDUNITTRAINER  HuBERT masked-unit pretraining loop on UNLABELED feature sequences.
    %
    %   tr = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(model, inputs, 'Ids', ids, ...
    %          'Epochs', 10, 'BatchSize', 32, 'LR', 5e-4, 'WeightDecay', 0.01, 'WarmupFraction', 0.08, ...
    %          'Clip', 5, 'Seed', 0, 'InputNorm', "utterance", 'ValInputs', {}, 'ValIds', [], ...
    %          'Checkpoint', "", 'LogFile', "", 'Verbose', true)
    %   tr.setTargets(units, 'Rate', "input", 'Codebook', cb)     units: cell of 1 x T unit rows
    %   tr.setValTargets(units, 'Rate', "input")
    %   tr.fit();   r = tr.evaluate("val"|"train")
    %   seqs = tr.hiddenFeatures(layer, "train"|"val")   cell of FeatureSequence (Dim x T'),
    %          input for an iteration-2 UnitCodebook (HuBERT sec. III-C re-clustering)
    %   tr.save(file);   ck = MaskedUnitTrainer.load(file)   (ck.Model, ck.History, ck.Extra, ...)
    %
    %   inputs: cell of Core.FeatureSequence (or F x T arrays) -- the model input features
    %   (log-mel). Transcripts are never seen. Targets are produced OUTSIDE from actual
    %   features by a UnitCodebook (assign) and attached with setTargets; Rate "input" means
    %   one unit per input frame (mapped to output frames with model.targetFrameIndex),
    %   "output" one unit per model output frame (e.g. units of hidden features).
    %
    %   ONE STEP s:  X = CMVN(batch) (HuBERTModel.normalizeUtterance; InputNorm "none" skips),
    %     M = model.sampleMask(T'_n) from the trainer's private mask stream,
    %     L = model.pretrainLoss(X, T, Y, M);  g = grad L;  clip ||g||_2 <= Clip;
    %     AdamW (ModuleOptimizer.forModule, no decay on biases/norms), lr = WarmupCosine(s).
    %   Batches: utterances sorted by length, chunks of BatchSize, chunk ORDER shuffled per epoch
    %   e by RandStream(Seed + e). Dropout draws use a private copy of the global RNG seeded
    %   with Seed (swapped in per step), masks a private RandStream(Seed + 7919): runs are
    %   reproducible regardless of other RNG use in the session.
    %   evaluate(): eval mode, no graph, masks from a FRESH RandStream(Seed + 104729) each call
    %   so the masked-prediction accuracy of different epochs is measured on identical masks.

    properties (SetAccess = private)
        Model
        Inputs cell
        Ids (1,:) string
        ValInputs cell = {}
        ValIds (1,:) string = strings(1, 0)
        Targets cell = {}
        ValTargets cell = {}
        TargetRate (1,1) string = "input"
        CodebookVersion (1,1) string = ""
        Options struct
        Optimizer
        Schedule
        TotalSteps (1,1) double
        Step (1,1) double = 0
        History struct
        RngState
        MaskStream
        Lengths (1,:) double
    end

    methods
        function obj = MaskedUnitTrainer(model, inputs, opts)
            arguments
                model (1,1) OceanNLP.Speech.Models.Pretrain.HuBERTModel
                inputs cell {mustBeNonempty}
                opts.Ids string = strings(1, 0)
                opts.Epochs (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.BatchSize (1,1) double {mustBeInteger, mustBePositive} = 32
                opts.LR (1,1) double {mustBePositive} = 5e-4
                opts.WeightDecay (1,1) double {mustBeNonnegative} = 0.01
                opts.WarmupFraction (1,1) double {mustBeGreaterThanOrEqual(opts.WarmupFraction, 0), mustBeLessThan(opts.WarmupFraction, 1)} = 0.08
                opts.Clip (1,1) double {mustBeNonnegative} = 5
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.InputNorm (1,1) string {mustBeMember(opts.InputNorm, ["utterance", "none"])} = "utterance"
                opts.ValInputs cell = {}
                opts.ValIds string = strings(1, 0)
                opts.Checkpoint (1,1) string = ""
                opts.LogFile (1,1) string = ""
                opts.Verbose (1,1) logical = true
            end
            if ~model.HasPretrainHead
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:NoHead', 'The model has no pretraining head.');
            end
            obj.Model = model;
            [obj.Inputs, obj.Ids] = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer.unpack(inputs, opts.Ids, model.InputDim);
            if ~isempty(opts.ValInputs)
                [obj.ValInputs, obj.ValIds] = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer.unpack(opts.ValInputs, opts.ValIds, model.InputDim);
                if ~isempty(intersect(obj.Ids, obj.ValIds))
                    error('OceanNLP:Speech:Models:MaskedUnitTrainer:Overlap', 'Train and validation ids overlap.');
                end
            end
            obj.Lengths = cellfun(@(v) size(v, 2), obj.Inputs);
            obj.Options = rmfield(opts, {'ValInputs', 'ValIds', 'Ids'});
            K = ceil(numel(obj.Inputs) / opts.BatchSize);
            obj.TotalSteps = opts.Epochs * K;
            obj.Schedule = OceanDL.optim.WarmupCosine(opts.LR, 'WarmupSteps', max(1, round(opts.WarmupFraction * obj.TotalSteps)), ...
                'TotalSteps', obj.TotalSteps, 'MinLR', opts.LR / 20);
            obj.Optimizer = OceanDL.optim.ModuleOptimizer.forModule(model, ...
                @(wd) OceanStats.Optim.AdamWOptimizer('LR', opts.LR, 'WeightDecay', wd), 'WeightDecay', opts.WeightDecay);
            prev = rng;
            rng(opts.Seed, 'twister');
            obj.RngState = rng;
            rng(prev);
            obj.MaskStream = RandStream('mt19937ar', 'Seed', opts.Seed + 7919);
            obj.History = struct('StepLoss', zeros(1, 0), 'StepMaskedAcc', zeros(1, 0), 'StepLR', zeros(1, 0), ...
                'StepMaskFraction', zeros(1, 0), 'StepSeconds', zeros(1, 0), 'Epochs', struct('Epoch', {}, ...
                'TrainLoss', {}, 'TrainMaskedAcc', {}, 'ValLoss', {}, 'ValMaskedAcc', {}, 'ValUnmaskedAcc', {}, 'Seconds', {}));
        end

        function setTargets(obj, units, opts)
            arguments
                obj
                units cell
                opts.Rate (1,1) string {mustBeMember(opts.Rate, ["input", "output"])} = "input"
                opts.Codebook = []
            end
            obj.Targets = obj.checkTargets(units, obj.Inputs, opts.Rate);
            obj.TargetRate = opts.Rate;
            if ~isempty(opts.Codebook)
                obj.CodebookVersion = opts.Codebook.Version;
                obj.Model.setCodebookVersion(opts.Codebook.Version);
            end
        end

        function setValTargets(obj, units, opts)
            arguments
                obj
                units cell
                opts.Rate (1,1) string {mustBeMember(opts.Rate, ["input", "output"])} = "input"
            end
            if opts.Rate ~= obj.TargetRate && ~isempty(obj.Targets)
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:Rate', 'Validation targets must use the training target rate.');
            end
            obj.ValTargets = obj.checkTargets(units, obj.ValInputs, opts.Rate);
        end

        function fit(obj)
            if isempty(obj.Targets)
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:NoTargets', 'Call setTargets first.');
            end
            K = obj.numBatches();
            obj.log(sprintf('HuBERT pretraining: %d utterances, %d epochs x %d batches = %d steps (codebook %s)', ...
                numel(obj.Inputs), obj.Options.Epochs, K, obj.TotalSteps, obj.CodebookVersion));
            while obj.Step < obj.TotalSteps
                e = floor(obj.Step / K);
                order = obj.batchOrder(e);
                t0 = tic;
                for k = (obj.Step - e * K + 1):K
                    obj.step(order{k});
                end
                h = obj.History;
                idx = (e * K + 1):obj.Step;
                v = struct('Loss', NaN, 'MaskedAccuracy', NaN, 'UnmaskedAccuracy', NaN);
                if ~isempty(obj.ValTargets)
                    v = obj.evaluate("val");
                end
                h.Epochs(end + 1) = struct('Epoch', e + 1, 'TrainLoss', mean(h.StepLoss(idx)), ...
                    'TrainMaskedAcc', mean(h.StepMaskedAcc(idx), 'omitnan'), 'ValLoss', v.Loss, ...
                    'ValMaskedAcc', v.MaskedAccuracy, 'ValUnmaskedAcc', v.UnmaskedAccuracy, 'Seconds', toc(t0));
                obj.History = h;
                obj.log(sprintf('epoch %d: train loss %.4f masked acc %.3f | val loss %.4f masked acc %.3f unmasked acc %.3f (%.0f s)', ...
                    e + 1, h.Epochs(end).TrainLoss, h.Epochs(end).TrainMaskedAcc, v.Loss, v.MaskedAccuracy, ...
                    v.UnmaskedAccuracy, h.Epochs(end).Seconds));
                if obj.Options.Checkpoint ~= ""
                    obj.save(obj.Options.Checkpoint);
                end
            end
        end

        function L = step(obj, idx)
            % STEP  One optimisation step on the utterances IDX.
            t0 = tic;
            [X, lens, Y] = obj.makeBatch(idx, obj.Inputs, obj.Targets);
            lensOut = obj.Model.outputLength(lens);
            M = obj.Model.sampleMask(lensOut, max([lensOut, 0]), obj.MaskStream);
            prev = rng;
            rng(obj.RngState);
            restore = onCleanup(@() rng(prev));
            s = obj.Step + 1;
            lr = obj.Schedule.at(s);
            obj.Model.train();
            obj.Optimizer.setLR(lr);
            obj.Optimizer.zeroGrad();
            [Lt, st] = obj.Model.pretrainLoss(X, lens, Y, M);
            Lt.backward();
            if obj.Options.Clip > 0
                OceanDL.nn.clipGradNorm(obj.Optimizer.Params, obj.Options.Clip);
            end
            obj.Optimizer.step();
            obj.RngState = rng;
            clear restore
            L = st.Loss;
            obj.Step = s;
            h = obj.History;
            h.StepLoss(s) = L;
            h.StepMaskedAcc(s) = st.MaskedAccuracy;
            h.StepLR(s) = lr;
            h.StepMaskFraction(s) = st.NumMasked / max(sum(lensOut), 1);
            h.StepSeconds(s) = toc(t0);
            obj.History = h;
        end

        function r = evaluate(obj, which)
            % EVALUATE  Frame-weighted masked / unmasked loss and accuracy (fixed eval masks).
            arguments
                obj
                which (1,1) string {mustBeMember(which, ["val", "train"])} = "val"
            end
            [inputs, targets] = obj.setFor(which);
            if isempty(targets)
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:NoTargets', 'No %s targets.', which);
            end
            was = obj.Model.Training;
            obj.Model.eval();
            restore = onCleanup(@() obj.Model.train(was));
            guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            stream = RandStream('mt19937ar', 'Seed', obj.Options.Seed + 104729);
            lens = cellfun(@(v) size(v, 2), inputs);
            [~, ord] = sort(lens);
            B = obj.Options.BatchSize;
            tot = struct('mL', 0, 'mN', 0, 'mHit', 0, 'uHit', 0, 'uN', 0);
            for b = 1:B:numel(ord)
                idx = ord(b:min(b + B - 1, end));
                [X, l, Y] = obj.makeBatch(idx, inputs, targets);
                lo = obj.Model.outputLength(l);
                M = obj.Model.sampleMask(lo, max([lo, 0]), stream);
                [~, st] = obj.Model.pretrainLoss(X, l, Y, M);
                if st.NumMasked > 0
                    tot.mL = tot.mL + st.MaskedLoss * st.NumMasked;
                    tot.mHit = tot.mHit + st.MaskedAccuracy * st.NumMasked;
                end
                tot.mN = tot.mN + st.NumMasked;
                if st.NumUnmasked > 0
                    tot.uHit = tot.uHit + st.UnmaskedAccuracy * st.NumUnmasked;
                end
                tot.uN = tot.uN + st.NumUnmasked;
            end
            r = struct('Loss', tot.mL / max(tot.mN, 1), 'MaskedAccuracy', tot.mHit / max(tot.mN, 1), ...
                'UnmaskedAccuracy', tot.uHit / max(tot.uN, 1), 'NumMasked', tot.mN, 'NumUnmasked', tot.uN);
            clear restore
        end

        function seqs = hiddenFeatures(obj, layer, which)
            % HIDDENFEATURES  Layer-LAYER representations of every utterance (unmasked, eval),
            %   as FeatureSequences at the OUTPUT frame rate (iteration-2 clustering input).
            arguments
                obj
                layer (1,1) double {mustBeInteger, mustBeNonnegative}
                which (1,1) string {mustBeMember(which, ["val", "train"])} = "train"
            end
            [inputs, ~] = obj.setFor(which);
            ids = obj.Ids;
            if which == "val", ids = obj.ValIds; end
            cfg = struct('Type', "HuBERTHidden", 'Layer', layer, 'Model', rmfield(obj.Model.config(), 'CodebookVersion'), ...
                'InputNorm', obj.Options.InputNorm);
            seqs = cell(1, numel(inputs));
            B = obj.Options.BatchSize;
            for b = 1:B:numel(inputs)
                idx = b:min(b + B - 1, numel(inputs));
                [X, l] = obj.makeBatch(idx, inputs, {});
                Z = obj.Model.hiddenStates(X, l, layer);
                lo = obj.Model.outputLength(l);
                for j = 1:numel(idx)
                    seqs{idx(j)} = OceanNLP.Speech.Core.FeatureSequence(reshape(Z(:, j, 1:lo(j)), size(Z, 1), lo(j)), ...
                        'FrameTimes', obj.Model.targetFrameIndex(lo(j), l(j)), 'AxisNames', ["hidden", "frame"], ...
                        'Config', cfg, 'Source', ids(idx(j)));
                end
            end
        end

        function K = numBatches(obj)
            K = ceil(numel(obj.Inputs) / obj.Options.BatchSize);
        end

        function order = batchOrder(obj, epoch)
            [~, idx] = sort(obj.Lengths);
            B = obj.Options.BatchSize;
            order = arrayfun(@(s) idx(s:min(s + B - 1, numel(idx))), 1:B:numel(idx), 'UniformOutput', false);
            order = order(randperm(RandStream('mt19937ar', 'Seed', obj.Options.Seed + epoch), numel(order)));
        end

        function save(obj, file, opts)
            arguments
                obj
                file (1,1) string
                opts.Extra struct = struct()
            end
            sd = obj.Model.stateDict();
            names = keys(sd);
            vals = cellfun(@(n) gather(sd(n)), names, 'UniformOutput', false);
            ck = struct('Format', "OceanHuBERTPretrain", 'Version', 1, ...
                'Created', string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
                'ModelConfig', obj.Model.config(), 'WeightNames', {names}, 'WeightValues', {vals}, ...
                'History', obj.History, 'Step', obj.Step, 'Options', obj.Options, ...
                'CodebookVersion', obj.CodebookVersion, 'TrainIds', obj.Ids, 'ValIds', obj.ValIds, 'Extra', opts.Extra);
            folder = fileparts(file);
            if strlength(folder) > 0 && ~isfolder(folder), mkdir(folder); end
            save(file, '-struct', 'ck', '-v7');
        end
    end

    methods (Static)
        function ck = load(file)
            arguments
                file (1,1) string
            end
            raw = load(file);
            if ~isfield(raw, 'Format') || string(raw.Format) ~= "OceanHuBERTPretrain"
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:Format', '"%s" is not a HuBERT pretraining checkpoint.', file);
            end
            model = OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig(raw.ModelConfig);
            model.loadStateDict(containers.Map(raw.WeightNames, raw.WeightValues, 'UniformValues', false));
            ck = raw;
            ck.Model = model;
        end
    end

    methods (Access = private)
        function [inputs, targets] = setFor(obj, which)
            if which == "val"
                inputs = obj.ValInputs; targets = obj.ValTargets;
            else
                inputs = obj.Inputs; targets = obj.Targets;
            end
        end

        function [X, lens, Y] = makeBatch(obj, idx, inputs, targets)
            vals = inputs(idx);
            lens = cellfun(@(v) size(v, 2), vals);
            F = obj.Model.InputDim;
            T = max([lens, 0]);
            N = numel(idx);
            X = zeros(F, N, T);
            for n = 1:N
                X(:, n, 1:lens(n)) = reshape(vals{n}, F, 1, lens(n));
            end
            if obj.Options.InputNorm == "utterance"
                X = OceanNLP.Speech.Models.Pretrain.HuBERTModel.normalizeUtterance(X, lens);
            end
            Y = [];
            if nargout > 2
                lo = obj.Model.outputLength(lens);
                Y = ones(N, max([lo, 0]));
                for n = 1:N
                    u = targets{idx(n)};
                    if obj.TargetRate == "input"
                        u = u(obj.Model.targetFrameIndex(lo(n), lens(n)));
                    end
                    Y(n, 1:lo(n)) = u(1:lo(n));
                end
            end
        end

        function units = checkTargets(obj, units, inputs, rate)
            if numel(units) ~= numel(inputs)
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:TargetCount', '%d target rows for %d utterances.', numel(units), numel(inputs));
            end
            for i = 1:numel(units)
                T = size(inputs{i}, 2);
                if rate == "output", T = obj.Model.outputLength(T); end
                u = double(units{i}(:).');
                if numel(u) ~= T
                    error('OceanNLP:Speech:Models:MaskedUnitTrainer:TargetLength', ...
                        'Utterance %d: %d targets for %d %s frames.', i, numel(u), T, rate);
                end
                if any(u < 1 | u > obj.Model.NumUnits | u ~= round(u))
                    error('OceanNLP:Speech:Models:MaskedUnitTrainer:TargetRange', 'Unit ids must lie in 1..%d.', obj.Model.NumUnits);
                end
                units{i} = u;
            end
        end

        function log(obj, msg)
            if obj.Options.Verbose
                fprintf('%s\n', msg);
            end
            if obj.Options.LogFile ~= ""
                fid = fopen(obj.Options.LogFile, 'a');
                fprintf(fid, '%s %s\n', string(datetime('now', 'Format', 'HH:mm:ss')), msg);
                fclose(fid);
            end
        end
    end

    methods (Static, Access = private)
        function [vals, ids] = unpack(inputs, ids, F)
            vals = cell(1, numel(inputs));
            src = strings(1, numel(inputs));
            for i = 1:numel(inputs)
                s = inputs{i};
                if isa(s, 'OceanNLP.Speech.Core.FeatureSequence')
                    vals{i} = s.Values; src(i) = s.Source;
                elseif isnumeric(s) && ismatrix(s)
                    vals{i} = double(s);
                else
                    error('OceanNLP:Speech:Models:MaskedUnitTrainer:Input', 'Inputs must be FeatureSequences or F x T arrays.');
                end
                if size(vals{i}, 1) ~= F
                    error('OceanNLP:Speech:Models:MaskedUnitTrainer:InputDim', 'Input %d has %d bins; the model expects %d.', i, size(vals{i}, 1), F);
                end
            end
            if isempty(ids)
                ids = src;
                if any(ids == "")
                    ids = "u" + string(1:numel(inputs));
                end
            end
            ids = reshape(string(ids), 1, []);
            if numel(ids) ~= numel(vals)
                error('OceanNLP:Speech:Models:MaskedUnitTrainer:Ids', '%d ids for %d inputs.', numel(ids), numel(vals));
            end
        end
    end
end
