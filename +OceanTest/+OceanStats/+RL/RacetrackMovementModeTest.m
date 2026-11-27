classdef RacetrackMovementModeTest < OceanTest.BaseTestCase
    % RACETRACKMOVEMENTMODETEST  RacetrackEnv movement modes, map-declared defaults and
    %   the reachability check (2026-09-27 diagnosis of exp_compare_mc_vs_td_racetrack:
    %   d_shape_with_shortcut declares movement_mode = bidirectional, which the MATLAB
    %   environment ignored, so the finish was unreachable and every episode hit the cap).

    methods (TestClassSetup)
        function projectRoot(tc)
            tc.applyFixture(matlab.unittest.fixtures.CurrentFolderFixture(tc.LibraryRoot));  % map files are found from here
        end
    end

    methods (Test)
        function testTiledMapPropertiesAreTheDefaults(tc)
            e = OceanStats.RL.Env.Classic.RacetrackEnv("d_shape_with_shortcut");
            tc.verifyEqual(e.MovementMode, "bidirectional");
            tc.verifyEqual([e.VelocityMin e.VelocityMax], [-1 1]);            % max_velocity 2 is exclusive
            tc.verifyEqual(e.MaxEpisodeSteps, 5000);
            tc.verifyEqual(e.FailProb, 0.1);
            tc.verifyTrue(e.FinishReachable);
            tc.verifyEqual(e.MapProperties.movement_mode, 'bidirectional');
            e2 = OceanStats.RL.Env.Classic.RacetrackEnv("d_shape_with_shortcut", 'FailProb', 0, 'MaxVelocity', 3);
            tc.verifyEqual([e2.FailProb e2.VelocityMin e2.VelocityMax], [0 -2 2]);   % explicit options win
        end

        function testUnreachableFinishIsReported(tc)
            % a small fixed map whose finish lies DOWN and LEFT of the start (not an editable map file)
            map = ["#####"; "#S..#"; "###.#"; "#F..#"; "#####"];
            f = @() OceanStats.RL.Env.Classic.RacetrackEnv(map, 'MovementMode', "forward-only");
            e = tc.verifyWarning(f, 'RacetrackEnv:FinishUnreachable');
            tc.verifyFalse(e.FinishReachable);
            e2 = OceanStats.RL.Env.Classic.RacetrackEnv(map, 'MovementMode', "bidirectional");
            tc.verifyTrue(e2.FinishReachable);
        end

        function testBidirectionalDynamics(tc)
            e = OceanStats.RL.Env.Classic.RacetrackEnv("d_shape_with_shortcut", 'FailProb', 0);
            e.reset('Seed', 1);
            e.set_state([3 10 -1 0]);                                         % open track, moving left
            s = e.step(5);                                                    % no acceleration
            tc.verifyEqual(s, [2 10 -1 0]);                                   % moved LEFT
            e.set_state([3 10 0 -1]);
            s = e.step(5);
            tc.verifyEqual(s, [3 9 0 -1]);                                    % moved DOWN
            e.set_state([3 10 1 0]);                                          % (1,0) + (-1,0) -> (0,0) off the start line:
            s = e.step(2);                                                    % keep the previous velocity (Python twin)
            tc.verifyEqual(s(3:4), [1 0]);
            tc.verifyTrue(all(ismember(e.get_valid_actions([3 10 1 1]), 1:9)));
            tc.verifyGreaterThan(e.state_to_index([3 10 -1 -1]), 0);          % negative velocities are states
        end

        function testSuttonBartoTracksUnchanged(tc)
            t1 = OceanStats.RL.Env.Classic.RacetrackEnv("Track1");
            tc.verifyEqual(t1.MovementMode, "forward-only");
            tc.verifyEqual([t1.VelocityMin t1.VelocityMax], [0 4]);
            tc.verifyTrue(t1.FinishReachable);
            tc.verifyTrue(isnan(t1.MaxEpisodeSteps));
            % Known map defect (not changed here): Track2's finish block is walled in on every
            % side, so it cannot be reached in either mode. See Planning note / diagnosis.
            t2 = tc.verifyWarning(@() OceanStats.RL.Env.Classic.RacetrackEnv("Track2"), 'RacetrackEnv:FinishUnreachable');
            tc.verifyFalse(t2.FinishReachable);
        end
    end
end
