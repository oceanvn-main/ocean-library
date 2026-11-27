% EXP_PRISONERS_DILEMMA_DOMINANCE
%   Exercises the Game Theory framework's Step 1 building blocks end to
%   end: builds the classic Prisoner's Dilemma as a NormalFormGame and
%   runs Iterated Elimination of Strictly Dominated Strategies (IESD) to
%   confirm that two purely rational agents converge on (Defect, Defect)
%   even though (Cooperate, Cooperate) is better for both.
%
%   ENVIRONMENT:     2-player Prisoner's Dilemma, bimatrix payoff
%                     [Cooperate, Defect] x [Cooperate, Defect]
%   ALGORITHM:       OceanStats.GameTheory.Solvers.IteratedDominanceSolver
%   HYPERPARAMETERS: none -- this is a static analytic solve, not a learning run
%   ASSUMPTIONS:
%     - Standard PD payoff ordering: Temptation(5) > Reward(3) > Punishment(1) > Sucker(0)
%     - Complete information: the payoff matrix is common knowledge to both players
%     - Agents are assumed fully rational (dominance reasoning only, no learning)
%   REPRODUCIBILITY: fully deterministic (no randomness involved)
%   OUTPUTS:
%     - Round-by-round elimination log
%     - Final surviving strategy profile, confirmed to be (Defect, Defect)
%
%   Reference: standard textbook Prisoner's Dilemma;
%   Bonanno, Game Theory, Ch.1 Sec 1.1.2 (Dominated Strategies)
%
%   Package context: OceanStats.GameTheory.{Core,Solvers}

%% 
D = [2; 3];
C = [2 4];
E = zeros(4, 5);
F = ones(5, 10);
%% 


clc; clear; close all;


% =========================================================================
% === 1. GAME SETUP =======================================================
% =========================================================================
% ASSUMPTION: Temptation=5, Reward=3, Punishment=1, Sucker=0 (standard PD ordering)
PayoffA = [3 0; 5 1];   % Player 1's payoff: rows/cols = [Cooperate, Defect]
PayoffB = [3 5; 0 1];   % Player 2's payoff

game = OceanStats.GameTheory.Core.NormalFormGame( ...
    'PayoffA', PayoffA, 'PayoffB', PayoffB, ...
    'StrategyNames', {{'Cooperate', 'Defect'}, {'Cooperate', 'Defect'}});

fprintf('[Setup] Prisoner''s Dilemma: %d players, %d strategies each\n', ...
    game.numPlayers(), game.numStrategies(1));

% =========================================================================
% === 1b. DISPLAY PAYOFF MATRIX ===========================================
% =========================================================================
fprintf('\n[Payoff Matrix] (Player 1 payoff, Player 2 payoff)\n\n');

namesP1 = game.strategyNames(1);
namesP2 = game.strategyNames(2);
n1 = game.numStrategies(1);
n2 = game.numStrategies(2);

rowLabelWidth = max(cellfun(@length, namesP1)) + 2;
cellWidth = 14;

fprintf('%*s', rowLabelWidth, '');
for j = 1:n2
    fprintf('%*s', cellWidth, namesP2{j});
end
fprintf('\n');

for i = 1:n1
    fprintf('%*s', rowLabelWidth, namesP1{i});
    for j = 1:n2
        cellStr = sprintf('(%g, %g)', game.payoff(1, [i, j]), game.payoff(2, [i, j]));
        fprintf('%*s', cellWidth, cellStr);
    end
    fprintf('\n');
end
fprintf('\n');

% =========================================================================
% === 2. ITERATED ELIMINATION OF STRICTLY DOMINATED STRATEGIES ============
% =========================================================================
solver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();
result = solver.solve(game);
history = solver.getHistory();

fprintf('\n[IESD] %d elimination round(s):\n', history.NumRounds);
playerNames = {'Player 1', 'Player 2'};
for k = 1:numel(history.Log)
    entry = history.Log(k);
    names = game.strategyNames(entry.Player);
    eliminatedName = names{entry.EliminatedStrategy};
    dominatorName  = names{entry.DominatingStrategy};
    fprintf('  Round %d: %s -- "%s" eliminated (strictly dominated by "%s")\n', ...
        entry.Round, playerNames{entry.Player}, eliminatedName, dominatorName);
end

% =========================================================================
% === 3. REPORT ============================================================
% =========================================================================
namesP1 = game.strategyNames(1);
namesP2 = game.strategyNames(2);
survivingP1 = namesP1{result.SurvivingIndices{1}};
survivingP2 = namesP2{result.SurvivingIndices{2}};

fprintf('\n[Result] Surviving profile: (%s, %s)\n', survivingP1, survivingP2);

if strcmp(survivingP1, 'Defect') && strcmp(survivingP2, 'Defect')
    fprintf('[Verified] Two rational agents converge on (Defect, Defect) via pure dominance reasoning.\n');
else
    error('exp_prisoners_dilemma_dominance:UnexpectedResult', ...
        'Expected (Defect, Defect), got (%s, %s).', survivingP1, survivingP2);
end
