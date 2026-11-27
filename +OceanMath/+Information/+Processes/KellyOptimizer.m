classdef KellyOptimizer
    % KELLYOPTIMIZER Computes optimal betting fractions, growth rates, and relative entropy metrics.
    %
    % This class provides mathematical optimization tools for horse race gambling
    % using the Kelly Criterion. It relates information-theoretic concepts
    % (entropy, Kullback-Leibler divergence) to investment growth rates.
    %
    % Core Mathematical Formulas (Plain Text):
    %   True entropy: H(p) = -sum(p_i * log2(p_i))
    %   Implied probability from odds: r_i = 1 / o_i
    %   Growth rate: W(b, p, o) = sum(p_i * log2(b_i * o_i + 1 - sum(b)))
    %   Betting loss (KL Divergence): D(p || b) = sum(p_i * log2(p_i / b_i))
    %   Maximum growth rate (odds divergence): W*(p) = D(p || r) (when fair and b = p)
    %
    % Examples:
    %   p = [0.5, 0.3, 0.2];
    %   odds = [2.0, 3.0, 5.0];
    %   H = OceanMath.Information.Processes.KellyOptimizer.entropy(p);
    %   b_opt = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p, odds);
    %   W = OceanMath.Information.Processes.KellyOptimizer.computeExpectedGrowth(p, b_opt, odds);
    %
    % See also: OceanMath.Information.Processes.RaceSimulator,
    %           OceanMath.Information.Processes.GamblingSimulator

    methods (Static)
        function H = entropy(p)
            % ENTROPY Compute Shannon entropy H(p) in bits.
            %
            % Gracefully handles boundary cases like p_i == 0 where 0 * log2(0) = 0.
            %
            % INPUTS:
            %   p - (1 x K) double vector of probabilities, or a DiscreteDistribution.
            %
            % OUTPUTS:
            %   H - scalar double, entropy in bits.
            arguments
                p {mustBeA(p, ["double", "OceanMath.Information.Distributions.DiscreteDistribution"])}
            end

            if isa(p, 'OceanMath.Information.Distributions.DiscreteDistribution')
                p_vec = p.evaluate(p.getSupport());
            else
                p_vec = p;
            end

            if any(p_vec < 0)
                error('OceanMath:Information:KellyOptimizer:NegativeProbabilities', ...
                    'Probabilities must be non-negative.');
            end

            p_active = p_vec(p_vec > 0);
            H = -sum(p_active .* log2(p_active));
        end

        function D = divergence(p, b)
            % DIVERGENCE Compute Kullback-Leibler divergence D(p || b) in bits.
            %
            % Measures the betting loss when using subjective probabilities b instead of true probabilities p.
            % Gracefully handles p_i == 0 and returns Inf if b_i == 0 for p_i > 0.
            %
            % INPUTS:
            %   p - (1 x K) double vector of true probabilities.
            %   b - (1 x K) double vector of betting fractions or belief probabilities.
            %
            % OUTPUTS:
            %   D - scalar double, divergence in bits.
            arguments
                p (1,:) double {mustBeNonnegative}
                b (1,:) double {mustBeNonnegative}
            end

            if length(p) ~= length(b)
                error('OceanMath:Information:KellyOptimizer:SizeMismatch', ...
                    'Vectors p and b must have the same length.');
            end

            p_active = p > 0;
            if any(b(p_active) == 0)
                D = Inf;
            else
                D = sum(p(p_active) .* log2(p(p_active) ./ b(p_active)));
            end
        end

        function b_opt = computeOptimalFractions(p, odds)
            % COMPUTEOPTIMALFRACTIONS Solves for optimal Kelly betting fractions under constraints.
            %
            % Solves: max sum(p_i * log2(b_i * o_i + 1 - sum(b))) subject to sum(b_i) <= 1 and b_i >= 0.
            % For standard fair odds, b_opt = p. For unfair odds or track-takes, it implements
            % an analytical water-filling / thresholding search.
            %
            % INPUTS:
            %   p    - (1 x K) double vector of true probabilities.
            %   odds - (1 x K) double vector of payoff odds (o_i).
            %
            % OUTPUTS:
            %   b_opt - (1 x K) double vector of optimal betting fractions.
            arguments
                p (1,:) double {mustBeNonnegative}
                odds (1,:) double {mustBePositive}
            end

            if length(p) ~= length(odds)
                error('OceanMath:Information:KellyOptimizer:SizeMismatch', ...
                    'Vectors p and odds must have the same length.');
            end

            % Ensure row vectors
            p = p(:)';
            odds = odds(:)';
            r = 1 ./ odds;

            % Normalize p if needed
            if abs(sum(p) - 1.0) > 1e-9
                p = p / sum(p);
            end

            K = length(p);
            
            % Standard proportional betting for fair odds or when sum(r) == 1
            if abs(sum(r) - 1.0) < 1e-9
                b_opt = p;
                return;
            end

            % Sort horses by p_i * o_i in descending order
            expected_returns = p .* odds;
            [sorted_returns, sort_idx] = sort(expected_returns, 'descend');

            % Iterate to find the optimal set S of horses to bet on
            S = [];
            R = 1.0;
            for k = 1:K
                idx = sort_idx(k);
                test_S = [S, idx];
                sum_p = sum(p(test_S));
                sum_r = sum(r(test_S));

                if sum_r >= 1.0
                    break;
                end

                R_cand = (1 - sum_p) / (1 - sum_r);
                if sorted_returns(k) > R_cand
                    S = test_S;
                    R = R_cand;
                else
                    break;
                end
            end

            b_opt = zeros(1, K);
            if ~isempty(S)
                b_opt(S) = p(S) - r(S) * R;
            end

            % Guard bounds
            b_opt = max(0, b_opt);
            if sum(b_opt) > 1.0
                b_opt = b_opt / sum(b_opt);
            end
        end

        function W = computeExpectedGrowth(p, b, odds)
            % COMPUTEEXPECTEDGROWTH Calculates the analytical expected log-growth rate W.
            %
            % Formula: W = sum(p_i * log2(b_i * o_i + 1 - sum(b)))
            %
            % INPUTS:
            %   p    - (1 x K) double vector of true probabilities.
            %   b    - (1 x K) double vector of betting fractions.
            %   odds - (1 x K) double vector of payoff odds.
            %
            % OUTPUTS:
            %   W - scalar double, growth rate in bits per race.
            arguments
                p (1,:) double {mustBeNonnegative}
                b (1,:) double {mustBeNonnegative}
                odds (1,:) double {mustBePositive}
            end

            if length(p) ~= length(b) || length(p) ~= length(odds)
                error('OceanMath:Information:KellyOptimizer:SizeMismatch', ...
                    'Vectors p, b, and odds must have the same length.');
            end

            p = p(:)';
            b = b(:)';
            odds = odds(:)';

            if abs(sum(p) - 1.0) > 1e-9
                p = p / sum(p);
            end

            cash = 1 - sum(b);
            W = 0;
            p_active = p > 0;
            
            % Compute sum(p_i * log2(b_i * o_i + cash))
            % Handles boundary cases where payoff is zero
            vals = b .* odds + cash;
            for i = find(p_active)
                if vals(i) <= 0
                    W = -Inf;
                    return;
                else
                    W = W + p(i) * log2(vals(i));
                end
            end
        end
    end
end
