classdef GamblingSimulator < handle
    % GAMBLINGSIMULATOR Runs a multi-step betting campaign and records empirical wealth history.
    %
    % This class simulates multiple sequential horse races and records the
    % wealth trajectory of a gambler betting according to a given strategy
    % (fraction of wealth allocated to each horse). It computes both empirical
    % and theoretical log-growth rates to validate the Kelly Criterion.
    %
    % Properties:
    %   TrueProbabilities - (1 x K) double vector of true probabilities.
    %   Odds              - (1 x K) double vector of payoff odds.
    %   StrategyFractions - (1 x K) double vector of betting fractions (sum <= 1).
    %   RaceSim           - OceanMath.Information.Processes.RaceSimulator instance.
    %
    % Examples:
    %   p = [0.5, 0.3, 0.2];
    %   o = [2.0, 3.0, 5.0];
    %   b = [0.5, 0.3, 0.2]; % Kelly betting
    %   sim = OceanMath.Information.Processes.GamblingSimulator(p, o, b);
    %   res = sim.runSimulation(100, 1000);
    %
    % See also: OceanMath.Information.Processes.RaceSimulator,
    %           OceanMath.Information.Processes.KellyOptimizer

    properties
        TrueProbabilities (1,:) double {mustBePositive}
        Odds (1,:) double {mustBePositive}
        StrategyFractions (1,:) double {mustBeNonnegative}
        RaceSim OceanMath.Information.Processes.RaceSimulator
    end

    methods
        function obj = GamblingSimulator(trueProbs, odds, strategyFractions)
            % GAMBLINGSIMULATOR Construct a GamblingSimulator.
            %
            % INPUTS:
            %   trueProbs         - (1 x K) or (K x 1) vector of true winning probabilities (sum = 1).
            %   odds              - (1 x K) or (K x 1) vector of payoff odds.
            %   strategyFractions - (1 x K) or (K x 1) vector of betting fractions (sum <= 1).
            arguments
                trueProbs (1,:) double {mustBePositive, mustBeVector}
                odds (1,:) double {mustBePositive, mustBeVector}
                strategyFractions (1,:) double {mustBeNonnegative, mustBeVector}
            end

            K = length(trueProbs);
            if length(odds) ~= K || length(strategyFractions) ~= K
                error('OceanMath:Information:GamblingSimulator:SizeMismatch', ...
                    'trueProbs, odds, and strategyFractions must all have the same length. Got %d, %d, %d.', ...
                    K, length(odds), length(strategyFractions));
            end

            if sum(strategyFractions) > 1.0 + 1e-9
                error('OceanMath:Information:GamblingSimulator:InvalidFractions', ...
                    'Sum of betting fractions strategyFractions must be <= 1. Got sum = %.10f.', sum(strategyFractions));
            end

            obj.TrueProbabilities = trueProbs(:)';
            obj.Odds = odds(:)';
            obj.StrategyFractions = strategyFractions(:)';
            
            % Instantiate internal RaceSimulator
            obj.RaceSim = OceanMath.Information.Processes.RaceSimulator(obj.TrueProbabilities);
        end

        function results = runSimulation(obj, startingWealth, numTrials)
            % RUNSIMULATION Simulate a sequence of betting rounds.
            %
            % In each round t:
            %   1. Draws a winning horse w using the RaceSimulator.
            %   2. Computes the new wealth:
            %      S_t = S_{t-1} * (b_w * o_w + (1 - sum(b)))
            %
            % INPUTS:
            %   startingWealth - (1,1) double, initial capital (> 0).
            %   numTrials      - (1,1) positive integer, number of rounds.
            %
            % OUTPUTS:
            %   results - struct containing:
            %       .WealthHistory     - (1 x numTrials + 1) double, wealth at each step.
            %       .WinningHistory    - (1 x numTrials) double, index of the winner at each step.
            %       .EmpiricalGrowth   - scalar double, empirical average log2 growth rate.
            %       .TheoreticalGrowth - scalar double, analytical log2 growth rate.
            arguments
                obj (1,1) OceanMath.Information.Processes.GamblingSimulator
                startingWealth (1,1) double {mustBePositive}
                numTrials (1,1) double {mustBePositive, mustBeInteger}
            end

            wealthHistory = zeros(1, numTrials + 1);
            winningHistory = zeros(1, numTrials);
            
            wealthHistory(1) = startingWealth;
            b = obj.StrategyFractions;
            o = obj.Odds;
            cash = 1.0 - sum(b);
            
            % Draw all winners sequentially and compute wealth
            for t = 1:numTrials
                w = obj.RaceSim.drawWinner();
                winningHistory(t) = w;
                
                % S_t = S_{t-1} * (b_w * o_w + cash)
                payoffFactor = b(w) * o(w) + cash;
                wealthHistory(t+1) = wealthHistory(t) * payoffFactor;
                
                % Handle bankrupcy early to avoid negative wealth or NaN
                if wealthHistory(t+1) <= 0
                    wealthHistory(t+1:end) = 0;
                    break;
                end
            end
            
            % Empirical average log-growth rate: G = (log2(S_T) - log2(S_0)) / T
            if wealthHistory(end) > 0
                empiricalGrowth = (log2(wealthHistory(end)) - log2(startingWealth)) / numTrials;
            else
                empiricalGrowth = -Inf;
            end
            
            % Theoretical growth rate
            theoreticalGrowth = OceanMath.Information.Processes.KellyOptimizer.computeExpectedGrowth( ...
                obj.TrueProbabilities, obj.StrategyFractions, obj.Odds);
            
            results = struct( ...
                'WealthHistory', wealthHistory, ...
                'WinningHistory', winningHistory, ...
                'EmpiricalGrowth', empiricalGrowth, ...
                'TheoreticalGrowth', theoreticalGrowth);
        end
    end
end
