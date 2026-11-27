classdef TestRacetrackEnv < matlab.unittest.TestCase
    % TESTRACETRACKENV Unit test suite for RacetrackEnv class.
    %
    % Verifies state LUT indexing, action filtering, acceleration noise,
    % path collision traversal, and reset/step behavior for Exercise 5.12.

    methods (Test)
        function testConstructorTrack1(testCase)
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1");
            testCase.verifyEqual(env.Height, 32);
            testCase.verifyEqual(env.Width, 17);
            testCase.verifyGreaterThan(env.TotalStates, 100);
            testCase.verifyGreaterThan(size(env.StartCells, 1), 0);
            testCase.verifyGreaterThan(size(env.FinishCells, 1), 0);
        end

        function testConstructorTrack2(testCase)
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track2");
            testCase.verifyEqual(env.Height, 32);
            testCase.verifyEqual(env.Width, 32);
            testCase.verifyGreaterThan(env.TotalStates, 100);
        end

        function testStateToIndexLUT(testCase)
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1");

            % Test a known valid start cell state [x, y, 0, 0]
            start_cell = env.StartCells(1, :);
            s_valid = [start_cell(1), start_cell(2), 0, 0];
            idx = env.state_to_index(s_valid);
            testCase.verifyGreaterThan(idx, 0);

            % Test an out-of-bounds state
            s_invalid = [999, 999, 0, 0];
            idx_inv = env.state_to_index(s_invalid);
            testCase.verifyEqual(idx_inv, 0);
        end

        function testValidActions(testCase)
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1");

            % On start line with (0,0) velocity: action (0,0) [index 5] is valid
            start_cell = env.StartCells(1, :);
            s_start = [start_cell(1), start_cell(2), 0, 0];
            acts_start = env.get_valid_actions(s_start);
            testCase.verifyTrue(ismember(5, acts_start));

            % Non-start cell with (1,0) velocity: action (-1,0) [index 2] results in (0,0) velocity, so it should NOT be valid
            [ty, tx] = find(env.Grid == 1);
            s_track = [tx(1), ty(1), 1, 0];
            acts_track = env.get_valid_actions(s_track);
            testCase.verifyFalse(ismember(2, acts_track));
        end

        function testStepDeterministic(testCase)
            % Set FailProb = 0 to test deterministic step physics
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1", 'FailProb', 0.0);
            start_s = env.reset();

            % Action 9 is (+1, +1): velocity becomes (1, 1)
            [next_s, r, done] = env.step(9);
            testCase.verifyEqual(r, -1);
            testCase.verifyFalse(done);
            testCase.verifyEqual(next_s(3), 1);
            testCase.verifyEqual(next_s(4), 1);
            testCase.verifyEqual(next_s(1), start_s(1) + 1);
            testCase.verifyEqual(next_s(2), start_s(2) + 1);
        end

        function testCrashReset(testCase)
            % Force state near a wall and drive into it
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1", 'FailProb', 0.0);
            env.reset();

            % Place car at (x=12, y=26) with velocity (1, 0)
            % Action 5 is (0, 0) -> velocity remains (1, 0), target position (13, 26)
            % Cell (13, 26) is off-track wall ('#'), so path hits wall and crashes
            env.set_state([12, 26, 1, 0]);
            [next_s, r, done] = env.step(5);

            testCase.verifyEqual(r, -1);
            testCase.verifyFalse(done);  % Episode continues on crash!
            % Next state should be reset to a start line cell with (0,0) velocity
            testCase.verifyEqual(next_s(3), 0);
            testCase.verifyEqual(next_s(4), 0);
            testCase.verifyTrue(ismember(next_s(1:2), env.StartCells, 'rows'));
        end

        function testFinishLineCrossing(testCase)
            env = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1", 'FailProb', 0.0);
            env.reset();

            % Place car at track cell (x=14, y=30) moving right with velocity (1, 0)
            % Finish line is at x=15..17 (cols 15..17 are 'F')
            env.set_state([14, 30, 1, 0]);
            % Action 8 is (+1, 0) -> velocity becomes (2, 0), moves right from x=14 to x=16 (Finish line)
            [~, r, done] = env.step(8);

            testCase.verifyEqual(r, -1);
            testCase.verifyTrue(done);  % Episode terminates!
        end

        function test3PillarContract(testCase)
            meta = OceanStats.RL.MonteCarlo.RacetrackEnv.getMetadata();
            testCase.verifyEqual(meta.ClassName, "OceanStats.RL.MonteCarlo.RacetrackEnv");

            spec = OceanStats.RL.MonteCarlo.RacetrackEnv.getPipelineSpec();
            testCase.verifyEqual(spec.ComponentName, "RacetrackEnv");
        end
    end
end
