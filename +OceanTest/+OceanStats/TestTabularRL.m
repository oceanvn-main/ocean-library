classdef TestTabularRL < OceanTest.BaseTestCase
    % TESTTABULARRL Unit test suite for OceanStats.RL Tabular RL package.
    %
    % Tests interface contract, Poisson distribution normalization, state indexing,
    % action constraints, free shuttle discount, parking penalty, and DP solver convergence.
    
    methods (Test)
        function testStateSpaceGeneration(testCase)
            env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 20);
            states = env.get_all_states();
            
            testCase.verifyEqual(size(states), [441, 2]);
            testCase.verifyEqual(min(states(:)), 0);
            testCase.verifyEqual(max(states(:)), 20);
        end
        
        function testStateIndexRoundTrip(testCase)
            env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 10);
            states = env.get_all_states();
            
            for k = 1:size(states, 1)
                s = states(k, :);
                idx = env.state_to_index(s);
                s_rec = env.index_to_state(idx);
                testCase.verifyEqual(s_rec, s);
            end
        end
        
        function testPoissonCacheNormalization(testCase)
            env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 15);
            
            % Check that transition probability columns sum to 1.0
            sum1 = sum(env.Loc1TransitionProb, 1);
            sum2 = sum(env.Loc2TransitionProb, 1);
            
            testCase.assertMatrixEquals(sum1, ones(1, 16), 1e-5, "Loc 1 transition probabilities do not sum to 1.");
            testCase.assertMatrixEquals(sum2, ones(1, 16), 1e-5, "Loc 2 transition probabilities do not sum to 1.");
        end
        
        function testValidActions(testCase)
            env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 20, 'MaxMove', 5);
            
            % State [10, 10]: full move range -5 to 5
            act1 = env.get_valid_actions([10, 10]);
            testCase.verifyEqual(act1, -5:5);
            
            % State [2, 1]: cannot move more than 2 from loc 1, cannot move more than 1 from loc 2 (-1 to 2)
            act2 = env.get_valid_actions([2, 1]);
            testCase.verifyEqual(act2, -1:2);
        end
        
        function testFreeShuttleDiscount(testCase)
            % Environment WITH free shuttle
            env_shuttle = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MoveCost', 2, 'FreeShuttle', true, 'ParkingPenalty', 0);
            % Environment WITHOUT free shuttle
            env_noshuttle = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MoveCost', 2, 'FreeShuttle', false, 'ParkingPenalty', 0);
            
            % Action 2 (move 2 cars 1->2): shuttle costs 2*(2-1) = $2. Noshuttle costs 2*2 = $4.
            [~, p1, r1] = env_shuttle.get_transitions([5, 5], 2);
            [~, p2, r2] = env_noshuttle.get_transitions([5, 5], 2);
            
            diff_reward = r1(1) - r2(1);
            testCase.verifyEqual(diff_reward, 2, 'AbsTol', 1e-6);
        end
        
        function testParkingPenalty(testCase)
            % Environment with parking penalty threshold at 10 cars
            env_pen = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('ParkingThreshold', 10, 'ParkingPenalty', 4, 'FreeShuttle', false);
            % Environment with no parking penalty
            env_nopen = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('ParkingThreshold', 10, 'ParkingPenalty', 0, 'FreeShuttle', false);
            
            % State [15, 5], action 0 -> overnight [15, 5] (loc 1 > 10, loc 2 <= 10) -> 1 penalty ($4)
            [~, ~, r_pen1] = env_pen.get_transitions([15, 5], 0);
            [~, ~, r_nopen1] = env_nopen.get_transitions([15, 5], 0);
            testCase.verifyEqual(r_nopen1(1) - r_pen1(1), 4, 'AbsTol', 1e-6);
            
            % State [15, 15], action 0 -> overnight [15, 15] (both > 10) -> 2 penalties ($8)
            [~, ~, r_pen2] = env_pen.get_transitions([15, 15], 0);
            [~, ~, r_nopen2] = env_nopen.get_transitions([15, 15], 0);
            testCase.verifyEqual(r_nopen2(1) - r_pen2(1), 8, 'AbsTol', 1e-6);
        end
        
        function testSolverPolicyIterationConvergence(testCase)
            % Small test environment for fast Policy Iteration verification
            env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 5, 'MaxMove', 2);
            solver = OceanStats.RL.DP.PolicyIterationSolver(env, 'DiscountFactor', 0.9, 'Theta', 1e-3);
            
            [V_mat, pi_mat, history] = solver.run(20);
            
            testCase.verifyTrue(history.PolicySteps > 0 && history.PolicySteps <= 10);
            testCase.verifyEqual(size(V_mat), [6, 6]);
            testCase.verifyEqual(size(pi_mat), [6, 6]);
            testCase.verifyTrue(all(V_mat(:) >= 0));
        end
    end
end
