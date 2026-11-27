function cases = td_extra_cases()
% TD_EXTRA_CASES  Learner configurations that the Python replay JSON cannot carry (strategy objects, Blackjack, custom
% policies). Each entry: name and a function handle building a fresh TDLearner. Their results were frozen from the
% pre-refactor learner in td_extra_golden.mat (capture_td_extra_golden.m) and TDRefactorGoldenTest checks them bit for bit.
import OceanStats.RL.TD.*
Cl = 'OceanStats.RL.Env.Classic.';
cliff = @() OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
walk = @() OceanStats.RL.Env.Classic.RandomWalkEnv();
maxb = @() OceanStats.RL.Env.Classic.MaximizationBiasEnv();
bj = @() OceanStats.RL.Env.Classic.BlackjackEnv();
pol = zeros(7, 2);  pol(2:6, 1) = 0.3;  pol(2:6, 2) = 0.7;
cases = {
    'softmax q-learning',      @() TDLearner('Env', cliff(), 'Target', 'qlearning', 'Exploration', 'softmax', 'StepSize', 0.3, 'Gamma', 0.9, 'Episodes', 40, 'Seed', 31, 'Record', true)
    'decaying eps sarsa',      @() TDLearner('Env', cliff(), 'Target', 'sarsa', 'Exploration', Exploration.DecayingEpsilonGreedyExploration('Initial', 1, 'Final', 0.05), 'StepSize', 0.3, 'Gamma', 0.95, 'Episodes', 60, 'Seed', 32, 'Record', true)
    'double q softmax',        @() TDLearner('Env', maxb(), 'Target', 'qlearning', 'Double_learning', true, 'Exploration', 'softmax', 'StepSize', 0.2, 'Episodes', 80, 'Seed', 33, 'Record', true)
    'blackjack q-learning',    @() TDLearner('Env', bj(), 'Target', 'qlearning', 'StepSize', 0.05, 'Exploration', 0.2, 'Episodes', 150, 'Seed', 34, 'Record', true)
    'blackjack expected sarsa lambda', @() TDLearner('Env', bj(), 'Target', 'expectedsarsa', 'Mode', 'trace', 'Lambda', 0.5, 'StepSize', 0.05, 'Exploration', 0.2, 'Episodes', 100, 'Seed', 35, 'Record', true)
    'blackjack td0',           @() TDLearner('Env', bj(), 'Task', 'prediction', 'StepSize', 0.05, 'Episodes', 200, 'Seed', 36, 'Record', true)
    'blackjack nstep',         @() TDLearner('Env', bj(), 'Task', 'prediction', 'Mode', 'nstep', 'N', 2, 'StepSize', 0.05, 'Episodes', 120, 'Seed', 37, 'Record', true)
    'custom policy walk',      @() TDLearner('Env', walk(), 'Task', 'prediction', 'PolicyProbs', pol, 'StepSize', 0.1, 'Episodes', 100, 'Seed', 38, 'Record', true)
    'initial value optimistic', @() TDLearner('Env', cliff(), 'Target', 'qlearning', 'InitialValue', 5, 'StepSize', 0.4, 'Gamma', 0.9, 'Exploration', 0.1, 'Episodes', 30, 'Seed', 39, 'Record', true)
    'norecord trace',          @() TDLearner('Env', cliff(), 'Target', 'sarsa', 'Mode', 'trace', 'Lambda', 0.8, 'Trace', 'replacing', 'StepSize', 0.2, 'Gamma', 0.95, 'Episodes', 40, 'Seed', 40)
    };
end
