classdef UnitCodebook < handle
    % UNITCODEBOOK  k-means "hidden-unit" codebook over real feature frames (HuBERT targets).
    %
    %   cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(K, 'Seed', 0, 'MaxIter', 50, ...
    %          'Tol', 1e-4, 'Normalize', "utterance", 'MaxFrames', inf)
    %   cb.fit(seqs, 'Ids', ids, 'FeatureConfig', cfg)   seqs: cell of Core.FeatureSequence or F x T arrays
    %   u  = cb.assign(seq)            1 x T unit ids (1..K) of one sequence (cell in -> cell out)
    %   s  = cb.usage(seqs)            counts, probabilities, entropy (bits), perplexity
    %   sd = cb.stateDict();  cb.loadStateDict(sd);  cb2 = UnitCodebook.fromStateDict(sd)
    %
    %   This is the target generator of HuBERT (Hsu et al. 2021, sec. II-B/III-C): the
    %   pseudo-labels are cluster ids of ACTUAL acoustic frames, recomputed by fit() from the
    %   data you pass -- there is no stored label table.
    %
    %   FRAME POOL. Each sequence x_{1..T} (F x 1 frames) is optionally normalised per
    %   utterance (Normalize "utterance", the CMVN of CTCRecognizer):
    %     x~_t = (x_t - mu) ./ sqrt(sigma^2 + 1e-8),  mu, sigma^2 = mean / variance over t
    %   and all frames are pooled into X = [x~_1 ... x~_M] (F x M), utterances in sorted-id
    %   order (so the result does not depend on the order the sequences are passed in). If M > MaxFrames a
    %   seeded uniform subset of MaxFrames frames (without replacement) is used.
    %
    %   k-MEANS++ SEEDING (Arthur & Vassilvitskii 2007): c_1 = a uniformly drawn frame;
    %   c_j drawn with probability D(x)^2 / sum_x D(x)^2, D(x) = min_{i<j} ||x - c_i||.
    %   LLOYD ITERATIONS (Lloyd 1982), r = 1, 2, ...:
    %     z_m = argmin_k ||x_m - c_k||^2                     (ties -> lowest k)
    %     J_r = (1/M) sum_m ||x_m - c_{z_m}||^2              (Inertia(r), mean squared distance)
    %     c_k = mean{x_m : z_m = k}
    %   An empty cluster is re-seeded with the frame farthest from its own centroid (logged
    %   in EmptyReseeds). Stop when (J_{r-1} - J_r) / J_{r-1} < Tol or r = MaxIter.
    %   All randomness comes from RandStream('mt19937ar', 'Seed', Seed): same data + same
    %   seed -> bit-identical centroids (global RNG untouched).
    %
    %   VERSION = first 16 hex chars of SHA-256 over: the centroid bytes, K, Normalize,
    %   jsonencode(FeatureConfig) (InputHistory removed) and the SORTED training utterance
    %   ids. Any change of data, features or result changes the Version, so a target set
    %   can always be traced to the codebook that produced it.

    properties (SetAccess = private)
        NumUnits (1,1) double
        Seed (1,1) double
        MaxIter (1,1) double
        Tol (1,1) double
        Normalize (1,1) string
        MaxFrames (1,1) double
        Centroids double = []          % F x K
        FeatureConfig struct = struct()
        TrainingIds (:,1) string = strings(0, 1)
        Inertia (1,:) double = []      % mean squared distance after each assignment step
        NumFrames (1,1) double = 0     % frames used by the last fit
        EmptyReseeds (1,1) double = 0
        Version (1,1) string = ""
    end

    methods
        function obj = UnitCodebook(numUnits, opts)
            arguments
                numUnits (1,1) double {mustBeInteger, mustBePositive}
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 50
                opts.Tol (1,1) double {mustBeNonnegative} = 1e-4
                opts.Normalize (1,1) string {mustBeMember(opts.Normalize, ["utterance", "none"])} = "utterance"
                opts.MaxFrames (1,1) double {mustBePositive} = inf
            end
            obj.NumUnits = numUnits;
            obj.Seed = opts.Seed;
            obj.MaxIter = opts.MaxIter;
            obj.Tol = opts.Tol;
            obj.Normalize = opts.Normalize;
            obj.MaxFrames = opts.MaxFrames;
        end

        function obj = fit(obj, seqs, opts)
            % FIT  k-means++ seeding + Lloyd iterations on the pooled frames of SEQS.
            arguments
                obj
                seqs cell {mustBeNonempty}
                opts.Ids string = strings(0, 1)
                opts.FeatureConfig = []
            end
            [vals, ids, fcfg] = obj.unpack(seqs, opts.Ids, opts.FeatureConfig);
            [ids, ord] = sort(ids);                 % pool in id order: the result depends on the
            vals = vals(ord);                       % (id -> frames) set, not on the input order
            X = cell2mat(cellfun(@(v) obj.prep(v), vals, 'UniformOutput', false));
            M = size(X, 2);
            if M == 0
                error('OceanNLP:Speech:Models:UnitCodebook:NoFrames', 'The sequences contain no frames.');
            end
            stream = RandStream('mt19937ar', 'Seed', obj.Seed);
            if M > obj.MaxFrames
                X = X(:, sort(randperm(stream, M, obj.MaxFrames)));
                M = size(X, 2);
            end
            K = obj.NumUnits;
            if size(unique(X.', 'rows'), 1) < K
                error('OceanNLP:Speech:Models:UnitCodebook:TooFewDistinct', ...
                    'Need at least K = %d distinct frames, the data has fewer.', K);
            end
            C = OceanNLP.Speech.Models.Pretrain.UnitCodebook.kmeansPlusPlus(X, K, stream);
            J = zeros(1, 0);
            reseeds = 0;
            xx = sum(X.^2, 1);
            for r = 1:obj.MaxIter
                D = max(xx - 2 * (C.' * X) + sum(C.^2, 1).', 0);      % K x M squared distances
                [dmin, z] = min(D, [], 1);
                J(r) = mean(dmin); %#ok<AGROW>
                Cnew = zeros(size(C));
                cnt = accumarray(z(:), 1, [K 1]).';
                for f = 1:size(X, 1)
                    Cnew(f, :) = accumarray(z(:), X(f, :).', [K 1]).';
                end
                Cnew = Cnew ./ max(cnt, 1);
                for k = find(cnt == 0)
                    [~, far] = max(dmin);                               % farthest frame from its centroid
                    Cnew(:, k) = X(:, far);
                    dmin(far) = 0;
                    reseeds = reseeds + 1;
                end
                C = Cnew;
                if r > 1 && (J(r - 1) - J(r)) <= obj.Tol * J(r - 1) && all(cnt > 0)
                    break;
                end
            end
            obj.Centroids = C;
            obj.Inertia = J;
            obj.NumFrames = M;
            obj.EmptyReseeds = reseeds;
            obj.FeatureConfig = fcfg;
            obj.TrainingIds = sort(ids(:));
            obj.Version = obj.computeVersion();
        end

        function u = assign(obj, seq)
            % ASSIGN  Nearest-centroid unit id of every frame (1 x T); a cell in gives a cell out.
            arguments
                obj
                seq
            end
            if isempty(obj.Centroids)
                error('OceanNLP:Speech:Models:UnitCodebook:NotFitted', 'Call fit() (or loadStateDict) first.');
            end
            if iscell(seq)
                u = cellfun(@(s) obj.assign(s), seq, 'UniformOutput', false);
                return;
            end
            v = OceanNLP.Speech.Models.Pretrain.UnitCodebook.valuesOf(seq);
            if size(v, 1) ~= size(obj.Centroids, 1)
                error('OceanNLP:Speech:Models:UnitCodebook:Dim', 'Codebook has %d dims, sequence has %d bins.', ...
                    size(obj.Centroids, 1), size(v, 1));
            end
            if isa(seq, 'OceanNLP.Speech.Core.FeatureSequence') && ~isempty(fieldnames(obj.FeatureConfig)) ...
                    && ~isempty(fieldnames(seq.Config)) ...
                    && ~isequaln(OceanNLP.Speech.Models.Pretrain.UnitCodebook.stripHistory(seq.Config), obj.FeatureConfig)
                error('OceanNLP:Speech:Models:UnitCodebook:FeatureMismatch', ...
                    'Sequence was computed with a different feature configuration than the codebook.');
            end
            X = obj.prep(v);
            if isempty(X)
                u = zeros(1, 0);
                return;
            end
            D = sum(X.^2, 1) - 2 * (obj.Centroids.' * X) + sum(obj.Centroids.^2, 1).';
            [~, u] = min(D, [], 1);
        end

        function s = usage(obj, seqs)
            % USAGE  Unit histogram over SEQS: Counts (1 x K), P, Entropy (bits), Perplexity
            %   (2^H), UsedUnits (count > 0), MaxEntropy = log2 K.
            arguments
                obj
                seqs cell
            end
            u = obj.assign(seqs);
            u = [u{:}];
            c = accumarray(u(:), 1, [obj.NumUnits 1]).';
            p = c / max(sum(c), 1);
            nz = p > 0;
            H = -sum(p(nz) .* log2(p(nz)));
            s = struct('Counts', c, 'P', p, 'Entropy', H, 'Perplexity', 2^H, 'UsedUnits', nnz(c), ...
                'MaxEntropy', log2(obj.NumUnits), 'NumFrames', sum(c));
        end

        function sd = stateDict(obj)
            sd = struct('Type', "UnitCodebook", 'NumUnits', obj.NumUnits, 'Seed', obj.Seed, ...
                'MaxIter', obj.MaxIter, 'Tol', obj.Tol, 'Normalize', obj.Normalize, 'MaxFrames', obj.MaxFrames, ...
                'Centroids', obj.Centroids, 'FeatureConfig', obj.FeatureConfig, 'TrainingIds', obj.TrainingIds, ...
                'Inertia', obj.Inertia, 'NumFrames', obj.NumFrames, 'EmptyReseeds', obj.EmptyReseeds, ...
                'Version', obj.Version);
        end

        function loadStateDict(obj, sd)
            arguments
                obj
                sd (1,1) struct
            end
            if ~isfield(sd, 'Type') || string(sd.Type) ~= "UnitCodebook"
                error('OceanNLP:Speech:Models:UnitCodebook:StateType', 'Not a UnitCodebook state dict.');
            end
            obj.NumUnits = sd.NumUnits; obj.Seed = sd.Seed; obj.MaxIter = sd.MaxIter; obj.Tol = sd.Tol;
            obj.Normalize = sd.Normalize; obj.MaxFrames = sd.MaxFrames; obj.Centroids = sd.Centroids;
            obj.FeatureConfig = sd.FeatureConfig; obj.TrainingIds = reshape(string(sd.TrainingIds), [], 1);
            obj.Inertia = sd.Inertia; obj.NumFrames = sd.NumFrames; obj.EmptyReseeds = sd.EmptyReseeds;
            obj.Version = obj.computeVersion();
            if string(sd.Version) ~= obj.Version
                error('OceanNLP:Speech:Models:UnitCodebook:VersionMismatch', ...
                    'Stored Version %s does not match the recomputed %s (corrupt state).', string(sd.Version), obj.Version);
            end
        end
    end

    methods (Static)
        function obj = fromStateDict(sd)
            obj = OceanNLP.Speech.Models.Pretrain.UnitCodebook(sd.NumUnits);
            obj.loadStateDict(sd);
        end

        function C = kmeansPlusPlus(X, K, stream)
            % KMEANSPLUSPLUS  D^2-weighted seeding (Arthur & Vassilvitskii 2007).
            M = size(X, 2);
            C = zeros(size(X, 1), K);
            C(:, 1) = X(:, randi(stream, M));
            d2 = sum((X - C(:, 1)).^2, 1);
            for j = 2:K
                tot = sum(d2);
                if tot <= 0
                    error('OceanNLP:Speech:Models:UnitCodebook:TooFewDistinct', 'All frames coincide with chosen centroids.');
                end
                cdf = cumsum(d2) / tot;
                pick = find(cdf >= rand(stream) , 1, 'first');
                if isempty(pick), pick = find(d2 > 0, 1, 'last'); end
                C(:, j) = X(:, pick);
                d2 = min(d2, sum((X - C(:, j)).^2, 1));
            end
        end

        function c = stripHistory(c)
            if isfield(c, 'InputHistory')
                c = rmfield(c, 'InputHistory');
            end
        end
    end

    methods (Access = private)
        function X = prep(obj, v)
            X = double(v);
            if obj.Normalize == "utterance" && ~isempty(X)
                mu = mean(X, 2);
                s2 = mean((X - mu).^2, 2);
                X = (X - mu) ./ sqrt(s2 + 1e-8);
            end
        end

        function [vals, ids, fcfg] = unpack(~, seqs, ids, fcfg)
            vals = cellfun(@OceanNLP.Speech.Models.Pretrain.UnitCodebook.valuesOf, seqs, 'UniformOutput', false);
            F = cellfun(@(v) size(v, 1), vals);
            if any(F ~= F(1))
                error('OceanNLP:Speech:Models:UnitCodebook:Dim', 'All sequences must have the same number of bins.');
            end
            isFS = cellfun(@(s) isa(s, 'OceanNLP.Speech.Core.FeatureSequence'), seqs);
            if isempty(ids)
                if ~all(isFS)
                    error('OceanNLP:Speech:Models:UnitCodebook:Ids', 'Numeric sequences need explicit ''Ids''.');
                end
                ids = cellfun(@(s) s.Source, seqs);
                if any(ids == "")
                    error('OceanNLP:Speech:Models:UnitCodebook:Ids', 'Some FeatureSequences have no Source; pass ''Ids''.');
                end
            end
            ids = reshape(string(ids), [], 1);
            if numel(ids) ~= numel(seqs)
                error('OceanNLP:Speech:Models:UnitCodebook:Ids', '%d ids for %d sequences.', numel(ids), numel(seqs));
            end
            if isempty(fcfg)
                fcfg = struct();
                if all(isFS)
                    cfgs = cellfun(@(s) OceanNLP.Speech.Models.Pretrain.UnitCodebook.stripHistory(s.Config), seqs, 'UniformOutput', false);
                    if ~all(cellfun(@(c) isequaln(c, cfgs{1}), cfgs))
                        error('OceanNLP:Speech:Models:UnitCodebook:MixedFeatures', ...
                            'Sequences were computed with different feature configurations.');
                    end
                    fcfg = cfgs{1};
                end
            end
            if ~isstruct(fcfg)
                error('OceanNLP:Speech:Models:UnitCodebook:FeatureConfig', 'FeatureConfig must be a struct.');
            end
            fcfg = OceanNLP.Speech.Models.Pretrain.UnitCodebook.stripHistory(fcfg);
        end

        function v = computeVersion(obj)
            md = java.security.MessageDigest.getInstance('SHA-256');
            md.update(typecast(double(obj.Centroids(:)), 'uint8'));
            meta = sprintf('K=%d|Normalize=%s|Features=%s|Ids=%s', obj.NumUnits, obj.Normalize, ...
                jsonencode(obj.FeatureConfig), strjoin(obj.TrainingIds, ','));
            md.update(unicode2native(meta, 'UTF-8'));
            d = typecast(md.digest(), 'uint8');
            v = string(lower(reshape(dec2hex(d(1:8), 2).', 1, [])));
        end
    end

    methods (Static, Access = private)
        function v = valuesOf(s)
            if isa(s, 'OceanNLP.Speech.Core.FeatureSequence')
                v = s.Values;
            elseif isnumeric(s) && ismatrix(s)
                v = double(s);
            else
                error('OceanNLP:Speech:Models:UnitCodebook:Input', 'Expected a FeatureSequence or an F x T array, got %s.', class(s));
            end
        end
    end
end
