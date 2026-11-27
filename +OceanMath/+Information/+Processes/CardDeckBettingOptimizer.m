classdef CardDeckBettingOptimizer
    % CARDDECKBETTINGOPTIMIZER Analytical utility class for deck-drawing sequential portfolios.
    %
    % This class computes generalized theoretical wealth and payoff multipliers
    % under symmetric or asymmetric compositions and odds configurations.
    %
    % Core Mathematical Formulas (Plain Text):
    %   Total sequence combinations: Combinations = nchoosek(R + B, R)
    %   Step Payoff multiplier (Scheme 1):
    %       If Red is drawn:   payoff = o_Red * b_Red + cash
    %       If Black is drawn: payoff = o_Black * b_Black + cash
    %   Constant Final Wealth Generalized Invariant:
    %       S_N* = S_0 * (o_Red^R * o_Black^B) / nchoosek(R+B, R)
    %
    % See also: OceanMath.Information.Processes.CardDeckSimulator

    methods (Static)
        function finalWealth = computeTheoreticalFinalWealth(R, B, oRed, oBlack, startingWealth)
            % COMPUTETHEORETICALFINALWEALTH Computes the generalized invariant final wealth.
            %
            % Uses logarithmic calculations with gammaln to ensure absolute numerical
            % stability and prevent overflow for large R and B configurations.
            %
            % INPUTS:
            %   R              - Total number of Red cards (R >= 0)
            %   B              - Total number of Black cards (B >= 0)
            %   oRed           - Payoff odds for Red card (> 0)
            %   oBlack         - Payoff odds for Black card (> 0)
            %   startingWealth - Starting capital / wealth S_0 (> 0)
            %
            % OUTPUTS:
            %   finalWealth    - The invariant final wealth at step R+B
            arguments
                R (1,1) double {mustBeNonnegative, mustBeInteger}
                B (1,1) double {mustBeNonnegative, mustBeInteger}
                oRed (1,1) double {mustBePositive}
                oBlack (1,1) double {mustBePositive}
                startingWealth (1,1) double {mustBePositive}
            end

            if R == 0 && B == 0
                finalWealth = startingWealth;
                return;
            end

            % nchoosek(R+B, R) = (R+B)! / (R! * B!)
            % ln(nchoosek(R+B, R)) = gammaln(R+B+1) - gammaln(R+1) - gammaln(B+1)
            logComb = gammaln(R + B + 1) - gammaln(R + 1) - gammaln(B + 1);
            
            % ln(S_N*) = ln(S_0) + R*ln(o_Red) + B*ln(o_Black) - ln(nchoosek(R+B, R))
            logWealth = log(startingWealth) + R * log(oRed) + B * log(oBlack) - logComb;
            
            finalWealth = exp(logWealth);
        end

        function [pRed, pBlack] = computeConditionalProbs(r, b)
            % COMPUTECONDITIONALPROBS Computes conditional probabilities prior to drawing.
            %
            % Returns:
            %   pRed   - Conditional probability of Red: r / (r + b)
            %   pBlack - Conditional probability of Black: b / (r + b)
            %
            % Gracefully handles the zero remaining cards boundary case by returning 0.
            arguments
                r (1,1) double {mustBeNonnegative}
                b (1,1) double {mustBeNonnegative}
            end

            total = r + b;
            if total == 0
                pRed = 0;
                pBlack = 0;
            else
                pRed = r / total;
                pBlack = b / total;
            end
        end

        function payoff = getStepPayoff(drawnColor, r, b, oRed, oBlack, betFractions)
            % GETSTEPPAYOFF Computes the wealth multiplier factor for a single draw step.
            %
            % INPUTS:
            %   drawnColor   - Shuffled card color drawn: 1 for Red, 2 for Black
            %   r            - Remaining Red cards *prior* to the draw
            %   b            - Remaining Black cards *prior* to the draw
            %   oRed         - Payoff odds for Red card (> 0)
            %   oBlack       - Payoff odds for Black card (> 0)
            %   betFractions - 1x2 double vector containing [b_Red, b_Black]
            %
            % OUTPUTS:
            %   payoff       - Multiplier factor for the wealth update
            arguments
                drawnColor (1,1) double {mustBeMember(drawnColor, [1, 2])}
                r (1,1) double {mustBeNonnegative}
                b (1,1) double {mustBeNonnegative}
                oRed (1,1) double {mustBePositive}
                oBlack (1,1) double {mustBePositive}
                betFractions (1,2) double {mustBeNonnegative}
            end

            b_Red = betFractions(1);
            b_Black = betFractions(2);

            % Verify betting fractions do not exceed 100% of wealth
            if (b_Red + b_Black) > 1.0 + 1e-9
                error('OceanMath:Information:CardDeckBettingOptimizer:InvalidFractions', ...
                    'Sum of betting fractions must be <= 1. Got sum = %.10f.', b_Red + b_Black);
            end

            cash = 1.0 - (b_Red + b_Black);

            if drawnColor == 1
                payoff = oRed * b_Red + cash;
            else
                payoff = oBlack * b_Black + cash;
            end
        end
    end
end
