classdef SpeechBatcher < handle
    % SPEECHBATCHER  Deterministic, length-aware, right-zero-padded mini-batches.
    %
    %   b = OceanNLP.Speech.Data.SpeechBatcher(dataset, 'BatchSize', 16, 'Shuffle', true, ...
    %         'Seed', 0, 'SortByLength', false)
    %   b.setEpoch(e);  K = b.numBatches();  B = b.batch(k)
    %
    %   ORDER for epoch e (a pure function of (Seed, e, dataset); a fresh
    %   RandStream('mt19937ar', 'Seed', Seed + e) per epoch, global RNG untouched):
    %     SortByLength = false: idx = randperm(N) (Shuffle) or 1:N; batch k takes
    %                           idx((k-1)B+1 : min(kB, N)).
    %     SortByLength = true:  idx = stable ascending sort of dataset.lengths();
    %                           consecutive chunks of B form the batches (minimal
    %                           padding); with Shuffle the ORDER of the batches is
    %                           randperm(K) -- batch contents stay length-homogeneous.
    %   setEpoch(e) also calls dataset.setEpoch(e), so augmentation is reproducible.
    %   The constructor calls setEpoch(0).
    %
    %   BATCH (contract sec. 2, "CBT"):
    %     X        F x N x T double, T = max_n L_n; X(:, n, 1:L_n) = item n's
    %              FeatureSequence.Values, X(:, n, L_n+1:T) = 0 (right zero-padding)
    %     Lengths  1 x N frame counts L_n;   Mask  N x T logical, Mask(n,t) = t <= L_n
    %     Targets  1 x N cell of token-id rows;  Text, Ids, Speakers  1 x N string
    %     Indices  1 x N dataset indices

    properties (SetAccess = private)
        Dataset
        BatchSize (1,1) double
        Shuffle (1,1) logical
        Seed (1,1) double
        SortByLength (1,1) logical
        Epoch (1,1) double = 0
        Batches cell = {}
    end

    properties (Access = private)
        LengthCache double = []
    end

    methods
        function obj = SpeechBatcher(dataset, opts)
            arguments
                dataset (1,1) OceanNLP.Speech.Data.AudioDataset
                opts.BatchSize (1,1) double {mustBeInteger, mustBePositive} = 16
                opts.Shuffle (1,1) logical = true
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.SortByLength (1,1) logical = false
            end
            obj.Dataset = dataset;
            obj.BatchSize = opts.BatchSize;
            obj.Shuffle = opts.Shuffle;
            obj.Seed = opts.Seed;
            obj.SortByLength = opts.SortByLength;
            obj.setEpoch(0);
        end

        function setEpoch(obj, epoch)
            arguments
                obj
                epoch (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            obj.Epoch = epoch;
            obj.Dataset.setEpoch(epoch);
            N = obj.Dataset.length();
            stream = RandStream('mt19937ar', 'Seed', obj.Seed + epoch);
            B = obj.BatchSize;
            if obj.SortByLength
                if isempty(obj.LengthCache)
                    obj.LengthCache = obj.Dataset.lengths();
                end
                [~, idx] = sort(obj.LengthCache, 'ascend');      % MATLAB sort is stable
                batches = chunk(idx, B);
                if obj.Shuffle
                    batches = batches(randperm(stream, numel(batches)));
                end
            else
                if obj.Shuffle
                    idx = randperm(stream, N);
                else
                    idx = 1:N;
                end
                batches = chunk(idx, B);
            end
            obj.Batches = batches;
        end

        function K = numBatches(obj)
            K = numel(obj.Batches);
        end

        function out = batch(obj, k)
            arguments
                obj
                k (1,1) double {mustBeInteger, mustBePositive}
            end
            if k > obj.numBatches()
                error('OceanNLP:Speech:Data:SpeechBatcher:Index', 'Batch %d of %d.', k, obj.numBatches());
            end
            idx = obj.Batches{k};
            N = numel(idx);
            items = cell(1, N);
            for n = 1:N
                items{n} = obj.Dataset.item(idx(n));
            end
            F = items{1}.Features.numBins();
            L = cellfun(@(it) it.Features.numFrames(), items);
            if any(cellfun(@(it) it.Features.numBins(), items) ~= F)
                error('OceanNLP:Speech:Data:SpeechBatcher:FeatureDim', 'Items in batch %d differ in feature dimension.', k);
            end
            T = max([L, 0]);
            X = zeros(F, N, T);
            for n = 1:N
                X(:, n, 1:L(n)) = reshape(items{n}.Features.Values, F, 1, L(n));
            end
            out = struct('X', X, 'Lengths', L, 'Mask', (1:T) <= L(:), ...
                'Targets', {cellfun(@(it) it.TokenIds, items, 'UniformOutput', false)}, ...
                'Text', cellfun(@(it) it.Text, items), 'Ids', cellfun(@(it) it.Id, items), ...
                'Speakers', cellfun(@(it) it.Speaker, items), 'Indices', idx);
        end

        function cfg = config(obj)
            cfg = struct('Type', "SpeechBatcher", 'BatchSize', obj.BatchSize, 'Shuffle', obj.Shuffle, ...
                'Seed', obj.Seed, 'SortByLength', obj.SortByLength, 'Epoch', obj.Epoch);
        end
    end
end

function c = chunk(idx, B)
% CHUNK  Consecutive pieces of length B (last one shorter).
N = numel(idx);
K = ceil(N / B);
c = cell(1, K);
for k = 1:K
    c{k} = idx((k-1)*B + 1 : min(k*B, N));
end
end
