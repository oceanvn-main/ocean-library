classdef RandomFeatureCandidateGenerator < OceanStats.Supervised.Tree.ICandidateGenerator
    % RANDOMFEATURECANDIDATEGENERATOR Random-feature split candidate generator.
    %
    % OVERVIEW:
    %   At each node, draws `Mtry` feature indices at random (without
    %   replacement) and generates exhaustive midpoint thresholds only for
    %   those features.  This is the sole algorithmic difference between a
    %   Random Forest tree and a plain Bagging tree (Breiman, 2001).
    %
    % USAGE:
    %   gen = OceanStats.Supervised.Tree.RandomFeatureCandidateGenerator(4);
    %   candidates = gen.generate(X, Y, minLeaf);
    %
    % PROPERTIES:
    %   Mtry (double) — Number of features to sample at each node.
    %                   Default: 0  (resolved at generate-time to floor(sqrt(p))).
    %
    % REFERENCE:
    %   Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties
        Mtry (1,1) double = 0   % 0 = auto (floor(sqrt(p)) at generate-time)
    end

    methods
        function obj = RandomFeatureCandidateGenerator(mtry)
            % RANDOMFEATURECANDIDATEGENERATOR Constructor.
            %
            % Args:
            %   mtry (double, optional): Number of features per node.
            %       Default 0 → auto-select floor(sqrt(p)) at runtime.
            arguments
                mtry (1,1) double = 0
            end
            obj.Mtry = mtry;
        end

        function candidates = generate(obj, X, ~, minLeaf)
            % GENERATE Draw mtry features at random, then enumerate midpoints.
            %
            % Args:
            %   X       (N x P) Feature matrix.
            %   ~       Y target — unused (required by interface).
            %   minLeaf (double) Minimum samples per child node.
            %
            % Returns:
            %   candidates: struct array with fields FeatureIndex, Threshold.

            [~, P] = size(X);

            % Resolve effective mtry
            mtry_eff = obj.Mtry;
            if mtry_eff <= 0
                mtry_eff = max(1, floor(sqrt(P)));
            end
            mtry_eff = min(mtry_eff, P);   % cannot exceed total features

            % Draw random feature subset (no replacement). To avoid dominates feature
            feat_idx = randperm(P, mtry_eff);

            candidates = struct('FeatureIndex', {}, 'Threshold', {});

            for jj = 1:mtry_eff
                j     = feat_idx(jj);
                x_col = X(:, j);

                % Filter NaNs and sort
                validMask = ~isnan(x_col);
                x_valid   = sort(x_col(validMask));
                nValid    = numel(x_valid);

                if nValid < (2 * minLeaf)
                    continue;
                end

                % Enumerate midpoint thresholds respecting minLeaf constraints
                for i = minLeaf : (nValid - minLeaf)
                    if x_valid(i) < x_valid(i + 1) %Check are these two adjacent values actually different
                        t          = (x_valid(i) + x_valid(i + 1)) / 2;
                        cand.FeatureIndex = j;
                        cand.Threshold    = t;
                        candidates(end + 1) = cand; %#ok<AGROW>
                    end
                end
            end
        end
    end
end
