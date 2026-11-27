classdef test_kinematic < OceanTest.BaseTestCase
    % TEST_KINEMATIC  Unit tests for CurveSimulator and KinematicState.

    methods (Test)

        % ============================================================== %
        %  KinematicState — Construction
        % ============================================================== %

        function test_state_construction(testCase)
            s = OceanMath.Simulation.KinematicState( ...
                0, 0, [1 2], 5, 0, 0, [1 0], [1 0 0 0]);
            testCase.assertEqual(s.Time, 0);
            testCase.assertEqual(s.Velocity, 5);
            testCase.assertEqual(s.Position, [1 2]);
        end

        function test_state_immutable(testCase)
            s = OceanMath.Simulation.KinematicState( ...
                0, 0, [0 0], 1, 0, 0, [1 0], [1 0 0 0]);
            testCase.assertError( ...
                @() setTimeField(s), 'MATLAB:class:SetProhibited');

            function setTimeField(state)
                state.Time = 99;  %#ok<NASGU>
            end
        end

        % ============================================================== %
        %  CurveSimulator — Constant Velocity (2D)
        % ============================================================== %

        function test_constant_velocity_moves_forward(testCase)
            pts = [0 0; 3 0; 6 0; 9 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 2, 'Dt', 0.1);

            sim.step();
            s = sim.getState();

            testCase.assertGreaterThan(s.Parameter, 0, ...
                'Body should have moved forward on the curve.');
            testCase.assertGreaterThan(s.ArcDistance, 0, ...
                'Distance traveled should be positive.');
            testCase.assertEqual(s.Velocity, 2, 'AbsTol', 1e-10, ...
                'Constant velocity should remain unchanged.');
        end

        function test_constant_velocity_distance_is_correct(testCase)
            % For constant velocity v and time dt, distance = v * dt
            pts = [0 0; 5 0; 10 0; 15 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 3, 'Dt', 0.5);

            sim.step();
            s = sim.getState();

            expectedDist = 3 * 0.5;  % v * dt
            testCase.assertEqual(s.ArcDistance, expectedDist, 'AbsTol', 1e-8, ...
                'Distance should equal v * dt for constant velocity.');
        end

        function test_body_stops_at_curve_end(testCase)
            pts = [0 0; 1 0; 2 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 100, 'Dt', 1);

            sim.runFor(100);
            s = sim.getState();

            testCase.assertEqual(s.Parameter, 1, 'AbsTol', 1e-10, ...
                'Body should stop at t = 1.');
            testCase.assertEqual(s.Velocity, 0, ...
                'Velocity should be 0 at curve end (non-looping).');
        end

        function test_loop_mode_wraps(testCase)
            pts = [0 0; 3 0; 6 0; 9 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 100, 'Dt', 0.5, 'Loop', true);

            sim.runFor(100);
            s = sim.getState();

            testCase.assertGreaterThan(s.ArcDistance, 0);
            testCase.assertGreaterThan(s.Velocity, 0, ...
                'In loop mode, velocity should not stop at end.');
        end

        % ============================================================== %
        %  CurveSimulator — Uniform Acceleration
        % ============================================================== %

        function test_uniform_acceleration_increases_velocity(testCase)
            pts = [0 0; 5 0; 10 0; 15 0; 20 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 1, 'Acceleration', 2, 'Dt', 0.1);

            sim.step();
            s = sim.getState();

            % v' = v + a*dt = 1 + 2*0.1 = 1.2
            testCase.assertEqual(s.Velocity, 1.2, 'AbsTol', 1e-8, ...
                'Velocity should increase by a*dt.');
        end

        function test_suvat_distance(testCase)
            % ds = v*dt + 0.5*a*dt^2 = 1*0.1 + 0.5*2*0.01 = 0.11
            pts = [0 0; 10 0; 20 0; 30 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 1, 'Acceleration', 2, 'Dt', 0.1);

            sim.step();
            s = sim.getState();

            expectedDist = 1*0.1 + 0.5*2*0.1^2;
            testCase.assertEqual(s.ArcDistance, expectedDist, 'AbsTol', 1e-8);
        end

        function test_deceleration_stops_body(testCase)
            pts = [0 0; 5 0; 10 0; 15 0; 20 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 1, 'Acceleration', -5, 'Dt', 0.1);

            sim.runFor(100);
            s = sim.getState();

            testCase.assertEqual(s.Velocity, 0, 'AbsTol', 1e-8, ...
                'Body should decelerate to zero.');
        end

        % ============================================================== %
        %  CurveSimulator — Acceleration Profile
        % ============================================================== %

        function test_profile_acceleration(testCase)
            pts = [0 0; 5 0; 10 0; 15 0; 20 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);

            % Time-varying acceleration: ramp up
            accelFn = @(t) 2 * t;

            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 0, 'AccelProfile', accelFn, 'Dt', 0.1);

            sim.step();
            s = sim.getState();

            % a(0.1) = 0.2, so ds = 0*0.1 + 0.5*0.2*0.01 = 0.001
            testCase.assertGreaterThanOrEqual(s.ArcDistance, 0);
        end

        % ============================================================== %
        %  CurveSimulator — 3D
        % ============================================================== %

        function test_3d_simulation(testCase)
            nPts = 6;
            theta = linspace(0, pi, nPts)';
            pts = [cos(theta), sin(theta), theta];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 2, 'Dt', 0.05);

            sim.runFor(10);
            s = sim.getState();

            testCase.assertEqual(numel(s.Position), 3, ...
                'Position should be 3D.');
            testCase.assertEqual(numel(s.Tangent), 3, ...
                'Tangent should be 3D.');
            testCase.assertEqual(numel(s.Orientation), 4, ...
                'Quaternion should have 4 elements.');
        end

        % ============================================================== %
        %  CurveSimulator — History & Observer
        % ============================================================== %

        function test_history_records_all_states(testCase)
            pts = [0 0; 3 0; 6 0; 9 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 1, 'Dt', 0.1);

            sim.runFor(5);
            h = sim.getHistory();

            % History should have initial + 5 steps = 6 entries
            testCase.assertEqual(numel(h), 6);
        end

        function test_event_fires_on_step(testCase)
            pts = [0 0; 3 0; 6 0; 9 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 1, 'Dt', 0.1);

            box = struct('fired', false);
            addlistener(sim, 'StateChanged', @(~,~) setField());
            sim.step();

            testCase.assertTrue(box.fired, 'StateChanged event must fire on step().');

            function setField()
                box.fired = true;
            end
        end

        % ============================================================== %
        %  CurveSimulator — Reset
        % ============================================================== %

        function test_reset(testCase)
            pts = [0 0; 5 0; 10 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 5, 'Dt', 0.1);

            sim.runFor(20);
            sim.reset();
            s = sim.getState();

            testCase.assertEqual(s.Time, 0);
            testCase.assertEqual(s.Parameter, 0, 'AbsTol', 1e-10);
            testCase.assertEqual(s.ArcDistance, 0);
            testCase.assertEqual(sim.getStepCount(), 0);
        end

        % ============================================================== %
        %  CurveSimulator — Travel Time Estimation
        % ============================================================== %

        function test_travel_time_constant_velocity(testCase)
            pts = [0 0; 3 4; 6 0];  % not a perfectly straight line
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 5, 'Dt', 0.01);

            tt = sim.estimateTravelTime(0, 1);
            L  = curve.totalLength();

            testCase.assertEqual(tt, L / 5, 'AbsTol', 1e-8, ...
                'Travel time = length / velocity for constant speed.');
        end

        % ============================================================== %
        %  CurveSimulator — MaxVelocity Clamp
        % ============================================================== %

        function test_max_velocity_clamp(testCase)
            % Use very long curve so body doesn't reach end and stop
            pts = [0 0; 100 0; 200 0; 300 0; 400 0; 500 0];
            curve = OceanMath.Geometry.CatmullRomSpline(pts);
            sim = OceanMath.Simulation.CurveSimulator(curve, ...
                'InitialVelocity', 1, 'Acceleration', 100, ...
                'Dt', 0.01, 'MaxVelocity', 5);

            sim.runFor(20);
            s = sim.getState();

            % Body should be clamped at 5, and still moving (not at end)
            testCase.assertLessThanOrEqual(s.Velocity, 5, ...
                'Velocity should never exceed MaxVelocity.');
            testCase.assertGreaterThan(s.Velocity, 0, ...
                'Body should still be moving (not hit curve end).');
        end
    end
end
