classdef MarkovEntropyRateStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % MARKOVENTROPYRATESTRATEGY  Compute the entropy rate H(X) of a Markov chain.
    %
    % The entropy rate of a stationary Markov chain is the expected per-step
    % uncertainty, computed using the shortcut formula:
    %
    %   H(X) = -sum_i sum_j  mu_i * P_ij * log2(P_ij)
    %
    % where mu is the stationary distribution and P is the transition matrix.
    % This formula avoids computing the full joint entropy of long sequences
    % and is exact in the stationary (ergodic) regime.
    %
    % Accepts any IStochasticProcess as its context — including MarkovChain
    % and RandomWalkGraph (once implemented).
    %
    % Usage:
    %   P  = [0.7 0.3; 0.4 0.6];
    %   mc = OceanMath.Information.Processes.MarkovChain(P);
    %
    %   strategy = OceanMath.Information.Strategies.MarkovEntropyRateStrategy();
    %   hRate = strategy.calculate(mc);    % bits / step
    %
    % Name-Value Options at construction:
    %   Base (double): Log base. 2 = bits (default), exp(1) = nats.
    %   WarnIfNotErgodic (logical): print a warning if isErgodic() is false.
    %       Default: true. The formula still runs but may not be meaningful.
    %
    % See also: OceanMath.Information.Strategies.IMetricStrategy
    %           OceanMath.Information.Strategies.ShannonEntropyStrategy
    %           OceanMath.Information.Processes.MarkovChain

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Base_            (1,1) double  {mustBePositive} = 2
        WarnIfNotErgodic (1,1) logical                  = true
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = MarkovEntropyRateStrategy(options)
            % MARKOVENTROPYRATESTRATEGY  Construct with optional settings.
            %
            % Name-Value Options:
            %   Base            (double):  Log base. Default: 2 (bits).
            %   WarnIfNotErgodic (logical): Warn when chain is not ergodic.
            arguments
                options.Base             (1,1) double  {mustBePositive} = 2
                options.WarnIfNotErgodic (1,1) logical                  = true
            end
            obj.Base_            = options.Base;
            obj.WarnIfNotErgodic = options.WarnIfNotErgodic;
        end

        % ---- IMetricStrategy Implementation ---- %

        function result = calculate(obj, context)
            % CALCULATE  Compute entropy rate H(X) for the stochastic process.
            %
            % INPUTS:
            %   context - an IStochasticProcess object
            %
            % OUTPUTS:
            %   result  - scalar double, entropy rate in bits/step (or
            %             configured base units)

            % Ergodicity warning
            if obj.WarnIfNotErgodic && ~context.isErgodic()
                warning('OceanMath:Information:MarkovEntropyRateStrategy:NotErgodic', ...
                    ['Chain is not ergodic. The entropy rate formula H = '   ...
                     '-sum mu_i P_ij log P_ij assumes stationarity. '        ...
                     'Results may not be meaningful.']);
            end

            % Get stationary distribution as a probability vector
            mu_dist = context.getStationaryDistribution();
            supp    = mu_dist.getSupport();
            mu      = mu_dist.evaluate(supp);   % (K x 1) column
            mu      = mu(:);

            % Get transition matrix
            P = context.getTransitionMatrix();   % (K x K)

            % Compute P_ij * log2(P_ij) safely (0 * log(0) = 0)
            logP       = zeros(size(P));
            mask       = P > 0;
            logP(mask) = log2(P(mask));

            % H(X) = -sum_i mu_i * sum_j P_ij * log2(P_ij)
            condH  = sum(P .* logP, 2);   % (K x 1): H(X_{t+1} | X_t = i)
            result = -sum(mu .* condH);   % scalar: expected conditional entropy

            % Base conversion
            if obj.Base_ ~= 2
                result = result * log(2) / log(obj.Base_);
            end
        end

        function tf = isCompatible(~, context)
            % ISCOMPATIBLE  True if context is an IStochasticProcess.
            tf = isa(context, 'OceanMath.Information.Processes.IStochasticProcess');
        end

        function name = getName(obj)
            % GETNAME  Return display name with base unit label.
            switch obj.Base_
                case 2,       unit = 'bits/step';
                case exp(1),  unit = 'nats/step';
                otherwise,    unit = sprintf('base-%.4g units/step', obj.Base_);
            end
            name = sprintf('Markov Entropy Rate H(X) [%s]', unit);
        end
    end
end
