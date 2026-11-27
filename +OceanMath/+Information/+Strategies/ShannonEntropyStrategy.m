classdef ShannonEntropyStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % SHANNONENTROPYSTRATEGY  Compute H(X) = -sum p(x) * log2(p(x)).
    %
    % This strategy calculates the Shannon entropy of a discrete probability
    % distribution. It accepts any IDistribution object as its context.
    %
    % For a DiscreteDistribution, this is exact.
    % For a ContinuousDistribution, this computes the differential entropy
    %   h(X) = -integral( f(x) * log2(f(x)) ) dx.
    %
    % Usage:
    %   strategy = OceanMath.Information.Strategies.ShannonEntropyStrategy();
    %   dist     = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF(...
    %                  [1 2 3], [0.5 0.3 0.2]);
    %   H = strategy.calculate(dist);   % 1.4855 bits
    %
    %   % Or use the safe wrapper:
    %   H = strategy.safeCalculate(dist);
    %
    % Name-Value Options at construction:
    %   Base (double): Log base. 2 = bits (default), exp(1) = nats, 10 = bans.
    %
    % See also: OceanMath.Information.Strategies.IMetricStrategy
    %           OceanMath.Information.Strategies.KLDivergenceStrategy

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Base_ (1,1) double {mustBePositive} = 2
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = ShannonEntropyStrategy(options)
            % SHANNONENTROPYSTRATEGY  Construct with optional log base.
            %
            % Name-Value Options:
            %   Base (double): logarithm base. Default: 2 (bits).
            %                  Use exp(1) for nats, 10 for bans/hartleys.
            arguments
                options.Base (1,1) double {mustBePositive} = 2
            end
            obj.Base_ = options.Base;
        end

        % ---- IMetricStrategy Implementation ---- %

        function result = calculate(obj, context)
            % CALCULATE  Compute Shannon entropy H(X) of the distribution.
            %
            % INPUTS:
            %   context - an IDistribution object (Discrete or Continuous)
            %
            % OUTPUTS:
            %   result  - scalar double, entropy in the configured base units
            %
            % The strategy delegates to the distribution's own entropy()
            % method, which each subclass overrides for maximum efficiency.
            % Base conversion is applied afterward if needed.
            h = context.entropy();   % Always returns bits (log base 2)

            % Convert to configured base if not bits
            if obj.Base_ ~= 2
                h = h * log(2) / log(obj.Base_);
            end
            result = h;
        end

        function tf = isCompatible(~, context)
            % ISCOMPATIBLE  True if context is an IDistribution.
            tf = isa(context, 'OceanMath.Information.Distributions.IDistribution');
        end

        function name = getName(obj)
            % GETNAME  Return display name with base unit label.
            switch obj.Base_
                case 2,       unit = 'bits';
                case exp(1),  unit = 'nats';
                case 10,      unit = 'bans';
                otherwise,    unit = sprintf('base-%.4g units', obj.Base_);
            end
            name = sprintf('Shannon Entropy H(X) [%s]', unit);
        end
    end
end
