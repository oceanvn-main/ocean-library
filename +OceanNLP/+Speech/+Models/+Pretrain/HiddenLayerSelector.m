classdef HiddenLayerSelector < handle
    % HIDDENLAYERSELECTOR  Data-driven choice of the hidden layer re-clustered for HuBERT iteration 2.
    %
    %   sel = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(K, 'Seeds', [0 1], ...
    %           'MaxIter', 50, 'Tol', 1e-4, 'Normalize', "none", 'MaxFrames', inf)
    %   sel.fit(trainByLayer, valByLayer, 'Layers', layers, 'Ids', trainIds, 'Reference', refVal)
    %   sel.Scores      table, one row per candidate layer
    %   sel.Layer       the selected layer;   sel.Codebook   its UnitCodebook (Seeds(1))
    %   nmi = HiddenLayerSelector.nmi(a, b)
    %
    %   trainByLayer{j} / valByLayer{j}: cell of FeatureSequence (or D x T arrays) -- the layer
    %   Layers(j) hidden features of the pretraining TRAIN / VALIDATION utterances (from
    %   MaskedUnitTrainer.hiddenFeatures). refVal (optional): cell of 1 x T' reference unit rows
    %   of the validation utterances at the same frame rate (e.g. iteration-1 MFCC units), used
    %   only for a reported diagnostic, never for the choice.
    %
    %   Hsu et al. (2021, sec. IV-D) choose the layer by phone purity / PNMI, which needs a phone
    %   alignment we do not have. Label-free replacement used here, CLUSTERING STABILITY
    %   (Ben-Hur et al. 2002; von Luxburg 2010): for each layer l, k-means codebooks are fitted on
    %   the TRAIN features with two seeds s_a, s_b (UnitCodebook, k-means++ + Lloyd); both assign
    %   the held-out VALIDATION frames, z^a, z^b, and
    %     Stability(l) = NMI(z^a, z^b),
    %     NMI(a, b) = I(a; b) / sqrt(H(a) H(b))      (Strehl & Ghosh 2002; := 1 if H(a) H(b) = 0
    %                                                  and a, b are both constant)
    %     I(a; b) = sum_{i,j} p_ij log(p_ij / (p_i. p_.j)),  H(a) = -sum_i p_i. log p_i.
    %   with p_ij the joint frame frequencies. A layer whose units do not depend on the
    %   initialisation has well-separated cluster structure on unseen audio.
    %   Selected layer = argmax_l Stability(l) (ties -> the lowest layer). With more than two
    %   seeds, Stability is the mean NMI over all seed pairs.
    %   Also reported per layer: validation usage entropy (bits) of the seed-s_a units, the
    %   held-out inertia (mean squared distance of validation frames to their centroid, relative
    %   to the validation total variance: 1 - R^2) and, if given, NMI with the reference units.
    %   Deterministic: same features + seeds -> identical Scores and Layer.

    properties (SetAccess = private)
        NumUnits (1,1) double
        Seeds (1,:) double
        MaxIter (1,1) double
        Tol (1,1) double
        Normalize (1,1) string
        MaxFrames (1,1) double
        Scores table = table()
        Layer (1,1) double = NaN
        Codebook = []
        Codebooks cell = {}
    end

    methods
        function obj = HiddenLayerSelector(numUnits, opts)
            arguments
                numUnits (1,1) double {mustBeInteger, mustBePositive}
                opts.Seeds (1,:) double {mustBeInteger, mustBeNonnegative} = [0 1]
                opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 50
                opts.Tol (1,1) double {mustBeNonnegative} = 1e-4
                opts.Normalize (1,1) string {mustBeMember(opts.Normalize, ["utterance", "none"])} = "none"
                opts.MaxFrames (1,1) double {mustBePositive} = inf
            end
            if numel(unique(opts.Seeds)) < 2
                error('OceanNLP:Speech:Models:HiddenLayerSelector:Seeds', 'Need at least two distinct seeds.');
            end
            obj.NumUnits = numUnits;
            obj.Seeds = opts.Seeds;
            obj.MaxIter = opts.MaxIter;
            obj.Tol = opts.Tol;
            obj.Normalize = opts.Normalize;
            obj.MaxFrames = opts.MaxFrames;
        end

        function obj = fit(obj, trainByLayer, valByLayer, opts)
            arguments
                obj
                trainByLayer cell {mustBeNonempty}
                valByLayer cell {mustBeNonempty}
                opts.Layers (1,:) double = 1:numel(trainByLayer)
                opts.Ids string = strings(0, 1)
                opts.Reference cell = {}
            end
            nL = numel(trainByLayer);
            if numel(valByLayer) ~= nL || numel(opts.Layers) ~= nL
                error('OceanNLP:Speech:Models:HiddenLayerSelector:Count', ...
                    '%d train layers, %d validation layers, %d layer indices.', nL, numel(valByLayer), numel(opts.Layers));
            end
            S = numel(obj.Seeds);
            stab = zeros(nL, 1); ent = zeros(nL, 1); rel = zeros(nL, 1); refN = NaN(nL, 1); iters = zeros(nL, 1);
            obj.Codebooks = cell(nL, S);
            for j = 1:nL
                z = cell(1, S);
                for s = 1:S
                    cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(obj.NumUnits, 'Seed', obj.Seeds(s), ...
                        'MaxIter', obj.MaxIter, 'Tol', obj.Tol, 'Normalize', obj.Normalize, 'MaxFrames', obj.MaxFrames);
                    cb.fit(trainByLayer{j}, 'Ids', opts.Ids);
                    obj.Codebooks{j, s} = cb;
                    u = cb.assign(valByLayer{j});
                    z{s} = [u{:}];
                end
                pairs = nchoosek(1:S, 2);
                stab(j) = mean(arrayfun(@(p) OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector.nmi( ...
                    z{pairs(p, 1)}, z{pairs(p, 2)}), 1:size(pairs, 1)));
                us = obj.Codebooks{j, 1}.usage(valByLayer{j});
                ent(j) = us.Entropy;
                iters(j) = numel(obj.Codebooks{j, 1}.Inertia);
                rel(j) = obj.heldOutInertia(obj.Codebooks{j, 1}, valByLayer{j}, z{1});
                if ~isempty(opts.Reference)
                    r = cellfun(@(v) double(v(:).'), opts.Reference, 'UniformOutput', false);
                    r = [r{:}];
                    if numel(r) ~= numel(z{1})
                        error('OceanNLP:Speech:Models:HiddenLayerSelector:Reference', ...
                            '%d reference units for %d validation frames.', numel(r), numel(z{1}));
                    end
                    refN(j) = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector.nmi(z{1}, r);
                end
            end
            [~, best] = max(stab);                   % max returns the first (lowest) on ties
            obj.Scores = table(opts.Layers(:), stab, ent, rel, refN, iters, ...
                'VariableNames', ["Layer", "Stability", "ValEntropyBits", "ValRelInertia", "NMIWithReference", "LloydIters"]);
            obj.Layer = opts.Layers(best);
            obj.Codebook = obj.Codebooks{best, 1};
        end
    end

    methods (Static)
        function v = nmi(a, b)
            % NMI  Normalised mutual information I(a;b) / sqrt(H(a) H(b)) of two label rows.
            arguments
                a (1,:) double
                b (1,:) double
            end
            if numel(a) ~= numel(b)
                error('OceanNLP:Speech:Models:HiddenLayerSelector:Length', 'Label rows differ in length.');
            end
            if isempty(a)
                v = NaN;
                return;
            end
            [~, ~, ia] = unique(a);
            [~, ~, ib] = unique(b);
            P = accumarray([ia(:), ib(:)], 1) / numel(a);
            pa = sum(P, 2); pb = sum(P, 1);
            Ha = -sum(pa .* log(pa)); Hb = -sum(pb .* log(pb));
            if Ha == 0 || Hb == 0
                v = double(Ha == 0 && Hb == 0);
                return;
            end
            nz = P > 0;
            Q = pa * pb;
            I = sum(P(nz) .* log(P(nz) ./ Q(nz)));
            v = I / sqrt(Ha * Hb);
        end
    end

    methods (Access = private)
        function r = heldOutInertia(obj, cb, seqs, z)
            vals = cellfun(@(s) OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector.values(s), seqs, 'UniformOutput', false);
            if obj.Normalize == "utterance"
                vals = cellfun(@(v) (v - mean(v, 2)) ./ sqrt(var(v, 1, 2) + 1e-8), vals, 'UniformOutput', false);
            end
            X = [vals{:}];
            C = cb.Centroids;
            J = mean(sum((X - C(:, z)).^2, 1));
            r = J / max(mean(sum((X - mean(X, 2)).^2, 1)), eps);
        end
    end

    methods (Static, Access = private)
        function v = values(s)
            if isa(s, 'OceanNLP.Speech.Core.FeatureSequence')
                v = s.Values;
            else
                v = double(s);
            end
        end
    end
end
