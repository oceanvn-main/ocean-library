% EXP_PARITY_GUESS
%   Plays the parity-guess game between two players and records, game by game, how player 2 learns from player 1.
%   Opens the Replay Studio so the slider moves through the games: win rates, what P2 guesses against what P1 really
%   puts, and what P2 holds (its belief or its Q table).
%
%   ENVIRONMENT:     OceanAgents.Env.Reference.ParityGuessEnv (PG1 contract): each turn both PUT a number (parity) and
%                    GUESS the other's parity; exactly one right guess wins, both right = draw, 200 turns = draw
%   PLAYERS:         P1: puts even 90 %, guesses uniformly, learns nothing.
%                    P2: puts even 70 %; guess learned by variant "fp" (fictitious play, an opponent model) or
%                    "q" (Q-learning, no model)
%   HYPERPARAMETERS: games = 800; Q: alpha 0.1, gamma 1, epsilon 0.05
%   REPRODUCIBILITY: JointRunner Seed = 5 (same run as OceanTest ParityGuessTest); recording does not change the run
%   OUTPUTS:         Results/replay/parity_<variant>.h5 and a Replay Studio per variant (layout "ParityLearning")
%   THEORY:          P2 wins p2/(1+p2), p2 = 0.1 + 0.8 g, g = P(P2 guesses even): random guess 0.333, best response
%                    (always even) 0.474 (PG1 contract section 3)
%
%   Package context: OceanAgents (Player, Runtime, Observe), OceanReplay, OceanViz.App.Replay

games = 800;
variants = "fp";                 % P2 learns by fictitious play ("q" = Q-learning is also available)
if ~exist("showStudio", "var"), showStudio = true; end

outDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'Results', 'replay');
if ~isfolder(outDir), mkdir(outDir); end

for variant = variants
    env = OceanAgents.Env.Reference.ParityGuessEnv('Roster', ["p1" "p2"]);
    F = env.Factors;
    coin = @(pEven) OceanAgents.Policies.TablePolicy(OceanAgents.Artifacts.PolicyArtifact('Kind', "ProbabilityTable", ...
        'Representation', "ColumnIndex", 'Table', repmat([pEven 1 - pEven], 3, 1), 'ActionLabels', 1:2, 'StateStart', 0));
    perceptions = @() {OceanAgents.Minds.Perception.MovesPerception(string(1:4)), OceanAgents.Minds.Perception.DigitPerception()};
    m1 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {coin(0.9), ...
        OceanAgents.Policies.RandomLegalPolicy()}), 1:4, 'Perceptions', perceptions());
    if variant == "q"
        codec = OceanAgents.Artifacts.DiscreteStateCodec(3, 'Start', 0);
        q = OceanStats.RL.TD.Values.TabularQ(true(3, 2), 0);
        m2 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {coin(0.7), ...
            OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:2, 'Epsilon', 0.05, 'StateCodec', codec)}), 1:4, ...
            'Perceptions', perceptions());
        m2.addMemory(OceanAgents.Memory.TableMemory("values", q, ["even" "odd"]));
        m2.addLearner(OceanAgents.Learners.FactorLearner(OceanAgents.Learners.OneStepQLearner(q, 1:2, ...
            'Alpha', 0.1, 'Gamma', 1, 'StateCodec', codec), F, "guess"), "q");
        learned = @() q.Q; shape = [3 2]; name = "qTable";            % rows: what P1 put last (none, even, odd); cols: guess even/odd
    else
        fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1);
        m2 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {coin(0.7), ...
            OceanAgents.Policies.FictitiousPlayPolicy(fp)}), 1:4, 'Perceptions', perceptions());
        m2.addMemory(OceanAgents.Memory.BeliefMemory("belief", fp));
        m2.addLearner(OceanAgents.Learners.FictitiousPlayLearner(fp), "fp");
        learned = @() firstBelief(fp); shape = [1 2]; name = "belief";   % P2's belief that P1 puts [even odd]
    end
    p1 = OceanAgents.Player("p1", m1);
    p2 = OceanAgents.Player("p2", m2);

    drive = @(observer, flush) playAll(env, p1, p2, observer, flush, learned, games);
    recordable = OceanAgents.Observe.ParityLearningRecordable(drive, F, games, 'LearnedShape', shape, 'LearnedName', name);
    file = fullfile(outDir, "parity_" + variant + ".h5");
    OceanReplay.Record.Recorder.run(recordable, recordable.defaultSpec(), file, 'Overwrite', true, ...
        'Title', "Parity guess: P2 learns by " + variant);
    rec = OceanReplay.Core.Recording(file);
    [~, R] = rec.series("p2Reward");
    nWin = sum(R == 1); nLoss = sum(R == -1); late = R(end-299:end);
    fprintf('%s, %d games (draws not reported)\n', variant, numel(R));
    fprintf('  all games : P2 won %d, P2 lost %d  (P2 win share of decided games %.3f)\n', nWin, nLoss, nWin / (nWin + nLoss));
    fprintf('  last 300  : P2 won %d, P2 lost %d  (P2 wins %.3f, loses %.3f of the 300 games)\n', ...
        sum(late == 1), sum(late == -1), mean(late == 1), mean(late == -1));
    if showStudio
        OceanViz.App.Replay.ReplayStudio(file);
    end
end

function out = playAll(env, p1, p2, observer, flush, learned, games)
    % One game per run(1) (identical games to run(games)); after each, record what P2 holds.
    p1.attach(observer); p2.attach(observer);
    runner = OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 5);
    out = [];
    for g = 1:games
        out = runner.run(1);
        flush(learned());
    end
end

function b = firstBelief(fp)
    c = fp.beliefs();
    b = c{1};
end
