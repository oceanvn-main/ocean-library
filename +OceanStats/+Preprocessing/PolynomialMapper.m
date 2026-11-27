classdef PolynomialMapper < OceanStats.Preprocessing.IPreprocessor
    % POLYNOMIALMAPPER  Polynomial feature expansion: X → [X, X^2, ..., X^p].
    %
    %   Augments raw input coordinates with polynomial powers up to degree p.
    %   Implements IPreprocessor so it can be used anywhere an encoder is needed.
    %
    % IMPORTANT ARCHITECTURAL DISTINCTION vs PolynomialExpander:
    %   - PolynomialMapper (this class) extends IPreprocessor for use in Neural 
    %     Networks. It computes simple univariate stacks (no cross-terms like X1*X2) 
    %     because Neural Nets learn interactions natively in their hidden layers.
    %   - PolynomialExpander (in +Pipeline) extends IModelStrategy for statistical 
    %     pipelines (GLM, BMA). It computes full multivariate combinatorial expansions.
    %
    %   CONVENTION:
    %     X_raw — (d_in x N) raw input
    %     X_enc — (d_in * Degree x N) stacked powers: [X; X.^2; ...; X.^p]
    %
    %   Does NOT add a bias column (handled by NNGraph.HasBias).
    %
    %   Usage:
    %     enc   = OceanStats.Preprocessing.PolynomialMapper('Degree', 3);
    %     X_enc = enc.encode(X_raw);   % (d_in*3 x N)
    %     d_out = enc.outputDim();     % d_in * Degree

    properties
        Degree   (1,1) double {mustBePositive} = 2
        InputDim_ (1,1) double = 0
    end

    methods
        function obj = PolynomialMapper(opts)
            arguments
                opts.Degree (1,1) double {mustBePositive} = 2
            end
            obj.Degree = opts.Degree;
        end

        function X_enc = encode(obj, X_raw)
            obj.InputDim_ = size(X_raw, 1);
            parts = cell(obj.Degree, 1);
            for p = 1:obj.Degree
                parts{p} = X_raw .^ p;
            end
            X_enc = vertcat(parts{:});
        end

        function d = outputDim(obj)
            if obj.InputDim_ == 0
                error('OceanStats:Preprocessing:PolynomialMapper:NotFitted', ...
                    'Call encode() at least once before outputDim().');
            end
            d = obj.InputDim_ * obj.Degree;
        end

        function s = name(~)
            s = 'polynomial';
        end
    end
end
