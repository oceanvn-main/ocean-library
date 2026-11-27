classdef ParityLearningReplayTest < OceanTest.BaseTestCase
    % PARITYLEARNINGREPLAYTEST  Recording how the parity-guess players learn: one point per game, the run unchanged
    % by the recording, the learned belief stored per game, and the ParityLearning layout opening in Replay Studio.

    methods (Static, Access = private)
        function out = drive(observer, flush)
            [p1, p2, env, fp] = OceanTest.OceanAgents.Integration.ParityGuessTest.players("fp");
            p1.attach(observer); p2.attach(observer);
            runner = OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 5);
            out = zeros(0, 2);
            for g = 1:40
                r = runner.run(1);
                out = [out; r.EpisodeReturns]; %#ok<AGROW>
                b = fp.beliefs(); flush(b{1});
            end
        end
    end

    methods (Test)
        function recordsEveryGameAndDoesNotChangeTheRun(tc)
            env = OceanAgents.Env.Reference.ParityGuessEnv('Roster', ["p1" "p2"]);
            rc = OceanAgents.Observe.ParityLearningRecordable(@OceanTest.OceanViz.ParityLearningReplayTest.drive, env.Factors, 40, ...
                'LearnedShape', [1 2], 'LearnedName', "belief");
            f = string(tempname) + ".h5"; tc.addTeardown(@() delete(f));
            returns = OceanReplay.Record.Recorder.run(rc, rc.defaultSpec(), f, 'Overwrite', true);
            [p1, p2, env2] = OceanTest.OceanAgents.Integration.ParityGuessTest.players("fp");
            plain = OceanAgents.Runtime.JointRunner(env2, {p1, p2}, 'Seed', 5).run(40);
            tc.verifyEqual(returns, plain.EpisodeReturns, 'recording changes nothing; one run(1) per game = run(40)');
            rec = OceanReplay.Core.Recording(f);
            [g, r2] = rec.series("p2Reward");
            tc.verifyEqual(g(:).', 1:40);
            tc.verifyEqual(r2(:), plain.EpisodeReturns(:, 2));
            [~, turns] = rec.series("turns");
            tc.verifyEqual(turns(:), plain.EpisodeLengths(:));
            b = double(rec.get("belief", struct('game', 40), "exact"));
            tc.verifyEqual(sum(b), 1, 'AbsTol', 1e-6, 'a belief is a distribution');
            L = OceanViz.App.Replay.Layouts.ParityLearningLayout();
            tc.verifyTrue(L.appliesTo(rec));
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close()); %#ok<NASGU>
            tc.verifyEqual(s.Layout.name(), "ParityLearning");
            s.setCoord("game", 20);
            caps = OceanTest.OceanViz.ReplayStudioTest.captions(s);
            tc.verifyFalse(any(contains(caps, "could not draw")), strjoin(caps, " | "));
        end
    end
end
