classdef ScoreWeightedEnsemble < OceanStats.Ensemble.WeightedEnsemble
    %SCOREWEIGHTEDENSEMBLE Generic ensemble using score-based weighting.
    % 
    %   The weights are calculated using the "Softmax" of the negative
    %   scores:
    %   w_i = exp (- Scale * (score_i - BestScore ))
    %   Common Configuration:
    %       1. BMA (Bayesian): Score = BIC, Scale = 0.5
    %       2. AIC Weights: Score = AIC, Scale = 0.5
    %       3. RMSE Weights: Score = MSE, Scale = (Inverse Temperature)
    %   
    %   USAGE:
    %       %BMA Mode (Default)
    %       ens = ScoreWeightedEnsemble(models, 'Method', 'BIC')
    %       % AIC Mode
    %       ens = ScoreWEightedEnsemble(models, 'Method', 'AIC')

    properties
        Name string = "Weighted Ensemble"
        SupportsAIC logical = true

        %The function that calculates the score: @(rss, k, N) -> double
        ScoringFcn function_handle

        %The scaling factor in the exponent : exp(-Scale * Delta)
        %For AIC/BIC, this is strictly 0.5.
        ScaleFactor (1, 1) double = 0.5
    end

    methods
        function obj = ScoreWeightedEnsemble(base_models, options)
            %Constructor
            % Options:
            % Methods: "BIC" (default), "AIC", "AICc"
            arguments
                base_models cell
                options.Method (1, 1) string = "BIC"
                options.Scale (1, 1) double = 0.5
            end
            obj@OceanStats.Ensemble.WeightedEnsemble(base_models);
            obj.ScaleFactor = options.Scale;

            %Configure Strategy based on Method Name
            switch upper(options.Method)
                case "BIC"
                    obj.Name = "Ensemble (BMA/BIC)";
                    obj.ScoringFcn = @(rss, k, N) OceanStats.ModelSelection.bic(rss, k, 'N', N, 'Type', 'RSS');
                case "AIC"
                    obj.Name = "Ensemble (Akaike/AIC)";
                    obj.ScoringFcn = @(rss, k, N) OceanStats.ModelSelection.aic(rss, k, 'N', N, 'Variant', 'Standard');
                case "AICC"
                    obj.Name = "Ensemble (Corrected AIC)";
                    obj.ScoringFcn = @(rss, k, N) OceanStats.ModelSelection.aic(rss, k, 'N', N, 'Variant', 'AICc');
                otherwise
                    error('Unknown weighting method: %s', options.Method);

            end
        end
        function weights = computeWeights(obj, metrics, N, base_fits)
            %  COMPUTEWEIGHTS Generalized Softmax Logic
            M = length(obj.BaseModels);

            scores = zeros(1, M);
            for i = 1:M
                %1. Get Componetns
                rss = metrics(i);
                k = obj.BaseModels{i}.getComplexity(base_fits{i});

                %2. Calculate Score (Delegated)
                % The ScoringFcn standardizes the interface to just (rss, k, N)
                scores(i) = obj.ScoringFcn(rss, k, N);
            end

            %3. Generalized Stable Softmax
            %Formula: w = exp ( - Scale * (Score - MinScore))
            min_score = min(scores);
            delta = scores - min_score;

            raw_weights = exp(-obj.ScaleFactor * delta);

            %Normalize to sum to 1
            weights = raw_weights / sum(raw_weights);
        end
    end

end