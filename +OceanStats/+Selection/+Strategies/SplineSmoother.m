classdef SplineSmoother < OceanStats.Base.IModelStrategy
    % SPLINESMOOTHER Pluggable spline smoother facade implementing IModelStrategy.
    %
    % Delegates all mathematical fitting and prediction to an injected
    % ISplineStrategy. This allows any spline type to be used interchangeably
    % in Backfitting, Pipeline, BootstrapSampler, or FeatureSelectionAudit.
    %
    % AVAILABLE STRATEGIES:
    %   'PSpline'      - B-Spline basis + 2nd-order difference penalty (DEFAULT)
    %   'NaturalCubic' - ESL eigendecomp smoother, exact linear reproduction
    %   'Regression'   - Natural cubic spline basis + OLS, no penalty
    %
    % USAGE:
    %   % Default (P-Spline, TargetDF = 6)
    %   s = OceanStats.Selection.Strategies.SplineSmoother();
    %
    %   % Factory shorthand
    %   s = SplineSmoother.create('PSpline',      'TargetDF', 6);
    %   s = SplineSmoother.create('NaturalCubic', 'DF', 8);
    %   s = SplineSmoother.create('Regression',   'NumKnots', 5);
    %
    %   % Direct strategy injection (for custom or test strategies)
    %   my_strat = OceanStats.NonParametric.Strategies.PSplineStrategy('TargetDF', 5);
    %   s = OceanStats.Selection.Strategies.SplineSmoother(my_strat);
    %
    %   % Use in Backfitting
    %   backfit = OceanStats.GAM.Backfitting('Smoother', s);
    %
    % Author: Ocean Library System
    % Date:   2026-02-28

    properties
        Name        string  = "Spline Smoother"
        SupportsAIC logical = true
        Strategy    % OceanStats.NonParametric.Strategies.ISplineStrategy
    end

    methods
        function obj = SplineSmoother(strategy)
            % CONSTRUCTOR
            %   strategy - ISplineStrategy instance (optional).
            %              Defaults to PSplineStrategy with TargetDF = 6.
            if nargin < 1 || isempty(strategy)
                strategy = OceanStats.NonParametric.Strategies.PSplineStrategy('TargetDF', 6);
            end
            obj.Strategy = strategy;
            obj.Name     = "Spline Smoother (" + strategy.Name + ")";
        end

        % -----------------------------------------------------------------
        function model = train(obj, X, Y, Weights)
            % TRAIN  Delegate to strategy.fit().
            if nargin < 4, Weights = []; end
            model = obj.Strategy.fit(X, Y, Weights);
        end

        % -----------------------------------------------------------------
        function y_pred = predict(obj, model, X_new)
            % PREDICT  Delegate to strategy.predict().
            y_pred = obj.Strategy.predict(model, X_new);
        end

        % -----------------------------------------------------------------
        function err = evaluate(~, Y_true, Y_pred)
            % EVALUATE  Mean Squared Error.
            err = mean((Y_true(:) - Y_pred(:)).^2);
        end

        % -----------------------------------------------------------------
        function k = getComplexity(obj, model)
            % GETCOMPLEXITY  Effective degrees of freedom from strategy.
            k = obj.Strategy.getDF(model);
        end
    end

    % =====================================================================
    methods (Static)
        function s = create(type, varargin)
            % CREATE  Factory for named strategy types.
            %
            %   s = SplineSmoother.create('PSpline',      'TargetDF', 6)
            %   s = SplineSmoother.create('NaturalCubic', 'DF', 8)
            %   s = SplineSmoother.create('Regression',   'NumKnots', 5)
            %
            import OceanStats.NonParametric.Strategies.*
            import OceanStats.Selection.Strategies.SplineSmoother

            switch lower(strtrim(type))
                case 'pspline'
                    strat = PSplineStrategy(varargin{:});
                case {'naturalcubic', 'natural'}
                    strat = NaturalCubicStrategy(varargin{:});
                case {'regression', 'regspline'}
                    strat = RegressionSplineStrategy(varargin{:});
                otherwise
                    error('SplineSmoother:unknownType', ...
                          'Unknown strategy "%s". Options: PSpline, NaturalCubic, Regression.', type);
            end
            s = SplineSmoother(strat);
        end
    end
end