classdef ExtraTreesCandidateGenerator < OceanStats.Supervised.Tree.ICandidateGenerator
    % EXTRATREESCANDIDATEGENERATOR Extremely Randomized Trees split generator.
    %
    % OVERVIEW:
    %   Implements the split-candidate generation strategy from:
    %     Geurts, Ernst & Wehenkel (2006). Extremely Randomized Trees.
    %     Machine Learning, 63(1), 3-42.
    %
    %   Algorithm (per node):
    %     1. Sample `Mtry` feature indices at random (same as RF).
    %     2. For each selected feature j, draw `NumCandidates` (default 1)
    %        threshold values UNIFORMLY at random from the observed feature
    %        range [min(X[:,j]), max(X[:,j])] in the current node.
    %
    %   Differences from RandomFeatureCandidateGenerator:
    %     - Thresholds are random, NOT the exhaustive midpoint grid.
    %     - With NumCandidates=1 this is the canonical ExtraTrees (K=1 in paper).
    %     - Dramatically faster per tree: O(mtry) candidates vs O(N*mtry).
    %     - MinLeafSize constraint is not applied (random cut may produce tiny leaves;
    %       the DecisionTree stopping condition handles this externally).
    %
    % USAGE:
    %   gen = OceanStats.Supervised.Tree.ExtraTreesCandidateGenerator(4);
    %   gen = OceanStats.Supervised.Tree.ExtraTreesCandidateGenerator(4, 3);
    %   candidates = gen.generate(X, Y, minLeaf);
    %
    % PROPERTIES:
    %   Mtry          (double) — Features sampled per node. 0 = auto sqrt(p).
    %   NumCandidates (double) — Random thresholds per feature. Default 1.
    %
    % REFERENCE:
    %   Geurts et al. (2006). Extremely Randomized Trees. Machine Learning, 63(1).
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties
        Mtry          (1,1) double = 0   % 0 = auto at generate-time
        NumCandidates (1,1) double = 1   % K in Geurts (2006): thresholds per feature
    end

    methods
        function obj = ExtraTreesCandidateGenerator(mtry, numCandidates)
            % EXTRATREESCANDIDATEGENERATOR Constructor.
            %
            % Args:
            %   mtry          (double, optional): Features per node. Default 0 (auto).
            %   numCandidates (double, optional): Random thresholds per feature. Default 1.
            arguments
                mtry          (1,1) double = 0
                numCandidates (1,1) double = 1
            end
            obj.Mtry          = mtry;
            obj.NumCandidates = max(1, numCandidates);
        end

        function candidates = generate(obj, X, ~, ~)
            % GENERATE Sample mtry features; draw random thresholds per feature.
            %
            % Args:
            %   X       (N x P): Feature matrix at current node.
            %   ~       Y: Unused (required by ICandidateGenerator interface).
            %   ~       minLeaf: Unused (ExtraTrees does not pre-screen splits).
            %
            % Returns:
            %   candidates: struct array with fields FeatureIndex, Threshold.

            [~, P] = size(X);

            % Resolve effective mtry
            mtry_eff = obj.Mtry;
            if mtry_eff <= 0
                mtry_eff = max(1, floor(sqrt(P)));
            end
            mtry_eff = min(mtry_eff, P);

            % Sample feature subset (no replacement)
            feat_idx = randperm(P, mtry_eff);

            candidates = struct('FeatureIndex', {}, 'Threshold', {});

            for jj = 1:mtry_eff
                j     = feat_idx(jj);
                x_col = X(:, j);

                % Work only with non-NaN values
                validMask = ~isnan(x_col);
                x_valid   = x_col(validMask);

                if numel(x_valid) < 2
                    continue;   % cannot split — skip feature
                end

                x_min = min(x_valid);
                x_max = max(x_valid);

                if x_min >= x_max
                    continue;   % all values identical — no valid split
                end

                % Draw NumCandidates thresholds uniformly from (x_min, x_max)
                thresholds = x_min + rand(1, obj.NumCandidates) * (x_max - x_min);

                for k = 1:obj.NumCandidates
                    cand.FeatureIndex = j;
                    cand.Threshold    = thresholds(k);
                    candidates(end + 1) = cand; %#ok<AGROW>
                end
            end
        end
    end
end
