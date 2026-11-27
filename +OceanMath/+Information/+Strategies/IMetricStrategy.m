classdef IMetricStrategy < handle
    % IMETRICSTRATEGY  Abstract interface for all information-theoretic metrics.
    %
    % This is the cornerstone of the Strategy Pattern in this architecture.
    % Every mathematical formula (Shannon Entropy, KL Divergence, Entropy Rate,
    % HMM Bounds, etc.) is a pluggable module that implements this interface.
    %
    % Design Principle:
    %   The Orchestrator (InformationEngine) only ever calls calculate(). It
    %   never contains an if-else for which formula to use. New metrics are
    %   added by creating new classes — not by modifying existing ones.
    %   This is the Open/Closed Principle.
    %
    % Contract:
    %   result = calculate(obj, context)
    %     - context: a struct or IDistribution/IStochasticProcess, depending
    %                on what the concrete strategy expects.
    %     - result:  a scalar (or struct for multi-value outputs like bounds).
    %
    % Optional interface for validation:
    %   tf = isCompatible(obj, context)
    %     - Returns true if this strategy can legally operate on the given
    %       context object. Called by the engine before calculate().
    %
    % See also: OceanMath.Information.Strategies.ShannonEntropyStrategy
    %           OceanMath.Information.Strategies.KLDivergenceStrategy
    %           OceanMath.Information.Strategies.MarkovEntropyRateStrategy

    methods (Abstract)
        % CALCULATE  Execute the metric formula and return the result.
        %
        % INPUTS:
        %   context - the data object(s) the strategy operates on.
        %             Could be an IDistribution, an IStochasticProcess,
        %             or a struct bundling multiple inputs.
        %
        % OUTPUTS:
        %   result  - scalar double for single-value metrics (entropy, rate).
        %             Struct with named fields for multi-value metrics (bounds).
        result = calculate(obj, context)

        % ISCOMPATIBLE  Validate that the context is appropriate for this strategy.
        %
        % INPUTS:
        %   context - the data object to check
        %
        % OUTPUTS:
        %   tf - logical scalar; false means calculate() would likely error
        tf = isCompatible(obj, context)

        % NAME  Return a human-readable name for display and logging.
        %
        % OUTPUTS:
        %   name - char or string (e.g. "Shannon Entropy H(X)")
        name = getName(obj)
    end

    methods
        % SAFERUNCALCULATE  Validate, then calculate. Throws a clear error
        % if the context is incompatible, rather than a cryptic MATLAB error.
        %
        % OUTPUTS:
        %   result - same as calculate()
        function result = safeCalculate(obj, context)
            if ~obj.isCompatible(context)
                error('OceanMath:Information:IMetricStrategy:Incompatible', ...
                    'Strategy "%s" cannot operate on context of class "%s".', ...
                    obj.getName(), class(context));
            end
            result = obj.calculate(context);
        end
    end
end
