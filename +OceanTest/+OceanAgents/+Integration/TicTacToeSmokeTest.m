classdef TicTacToeSmokeTest < OceanTest.BaseTestCase
    % TICTACTOESMOKETEST  AB2 success test O6 (user, 2026-10-01): a simple agent with an EMPTY body and a MOVES
    % observation only plays tic-tac-toe against a random-legal opponent. Not about winning: it shows the pieces work
    % end to end - legal moves only, the Laws refuse illegal moves, turns alternate, every game ends with the right
    % rewards, and the run is reproducible from its seed.

    methods (Static)
        function [rec, mind, env] = play(seed, games)
            env = OceanAgents.Env.Reference.TicTacToeEnv('Roster', ["x" "o"]);
            mind = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.RandomLegalPolicy(), 1:9);
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:9), ...
                'Sensors', {OceanAgents.Body.Sensors.MovesObservation(string(1:9))}));
            x = OceanAgents.Runtime.AgentHost("x", mind);
            o = OceanAgents.Runtime.AgentHost("o", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy()));
            rec = OceanAgents.Runtime.JointRunner(env, {x, o}, 'Seed', seed).run(games);
        end
    end

    methods (Test)
        function gamesRunToAProperEnd(tc)
            rec = tc.play(3, 200);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyEqual(rec.NumEpisodes, 200);
            R = rec.EpisodeReturns;
            tc.verifyTrue(all(ismember(R, [1 -1; -1 1; 0 0], 'rows')), 'win/loss or a draw, nothing else');
            tc.verifyTrue(all(rec.EpisodeLengths >= 5 & rec.EpisodeLengths <= 9), 'a game lasts 5 to 9 moves');
            tc.verifyEqual(R(rec.EpisodeLengths == 9 & R(:, 1) == 0, :), zeros(nnz(rec.EpisodeLengths == 9 & R(:, 1) == 0), 2));
            tc.verifyGreaterThan(nnz(R(:, 1) == 1), 0);
            tc.verifyGreaterThan(nnz(R(:, 1) == -1), 0, 'losses on the opponent''s turn reach the agent');
            again = tc.play(3, 200);
            tc.verifyEqual(again.EpisodeReturns, R, 'reproducible from the seed');
            tc.verifyEqual(again.EpisodeLengths, rec.EpisodeLengths);
        end

        function theLawsRefuseIllegalMovesAndTurnsAlternate(tc)
            env = OceanAgents.Env.Reference.TicTacToeEnv('Roster', ["x" "o"]);
            f = env.reset(1);
            tc.verifyEqual(f.Requested, [true false]);
            D = @(who, ep, fr, a) OceanAgents.Core.Decision('AgentID', who, 'EpisodeID', ep, 'FrameID', fr, 'Action', a);
            B = @(fr, d) OceanAgents.Core.ActionBatch('EpisodeID', fr.EpisodeID, 'FrameID', fr.FrameID, 'Phase', fr.Phase, ...
                'Decisions', d);
            f = env.step(B(f, D("x", f.EpisodeID, f.FrameID, 5)));
            tc.verifyEqual(f.Requested, [false true], 'turns alternate');
            tc.verifyError(@() env.step(B(f, D("o", f.EpisodeID, f.FrameID, 5))), ?MException);   % occupied cell
            tc.verifyError(@() env.step(B(f, D("x", f.EpisodeID, f.FrameID, 1))), ?MException);   % not x's turn
            tc.verifyEqual(env.Board, [0 0 0 0 1 0 0 0 0], 'refused moves changed nothing');
            tc.verifyEqual(f.Views{2}.Observation, [0 0 0 0 2 0 0 0 0], 'o sees x''s mark as the opponent''s');
            tc.verifyEqual(f.Views{1}.ActionMask, false(1, 9), 'no moves for the player who must wait');
        end
    end
end
