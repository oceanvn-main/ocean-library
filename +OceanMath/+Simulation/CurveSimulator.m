classdef CurveSimulator < handle
    % CURVESIMULATOR  Physics engine for moving a body along an IParametricCurve.
    %
    % Given a parametric curve and initial conditions (velocity, acceleration),
    % this simulator advances the body's position using Newtonian kinematics:
    %   s(t + delta t) = s(t) + s'(t) delta t + 1/2 s''(t) delta t^2 + ... (Taylor series expansion). Then subtract s(t) to get ds.
    %   ds = v * dt + 0.5 * a * dt^2    (distance increment)
    %   v' = v + a * dt                 (velocity update) . a = dv/dt. Rearrange to dv = a * dt. Then v_new = v_old + dv.
    %
    %Mapping to curve parameter tau (Graphics problem)
    %   tau' = tau + ds / L              (curve parameter update, L = total arc length).  dTau/ds = 1 / (ds/dTau) = 1 / ||r'(tau)||. -> dTau = ds * (1 / ||r'(tau)||). Approx as dTau = ds * (1/L) = ds / L
    %
    % The simulator tracks all physical quantities through a KinematicState
    % value object and fires a 'StateChanged' event after each step(),
    % following the same Observer pattern as MarkovSimulator.
    %
    % Modes of Operation:
    %   CONSTANT_VELOCITY - Body moves at fixed speed (a = 0).
    %                       Useful for shortest-path travel time simulation.
    %   ACCELERATING      - Body obeys v' = v + a*dt continuously.
    %                       Useful for realistic transit/rollercoaster sim.
    %   PROFILE           - Acceleration is a user-supplied function a(t).
    %                       Useful for custom throttle/brake profiles.
    %
    % Usage:
    %   curve = OceanMath.Geometry.CatmullRomSpline(pts);
    %   sim   = OceanMath.Simulation.CurveSimulator(curve, ...
    %               'InitialVelocity', 5, 'Acceleration', 0, 'Dt', 0.01);
    %   for i = 1:500; sim.step(); end
    %   states = sim.getHistory();
    %
    % See also: OceanMath.Simulation.KinematicState
    %           OceanMath.Geometry.IParametricCurve
    %           OceanMath.Geometry.FrenetFrame

    % ------------------------------------------------------------------ %
    events
        % STATECHANGED  Fired after each step() with old and new KinematicState.
        StateChanged
    end

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Curve_ : the parametric curve to traverse
        Curve_

        % Frenet_ : FrenetFrame instance for orientation computation
        Frenet_

        % Physics state
        CurrentState_   % KinematicState value object
        TotalArcLength_ (1,1) double  % L = curve.totalLength() cached

        % Simulation parameters
        Dt_              (1,1) double  % time step size (seconds)
        AccelMode_       string        % 'constant', 'uniform', or 'profile'
        AccelValue_      (1,1) double  % constant acceleration (for 'uniform')
        AccelFn_                       % function handle a(t) (for 'profile')

        % History buffer
        History_         % cell array of KinematicState objects
        StepCount_       (1,1) double = 0

        % Options
        Loop_            (1,1) logical = false  % wrap around at t=1?
        ClampVelocity_   (1,1) double = Inf     % max speed cap
    end


    % ------------------------------------------------------------------ %
    methods
        function obj = CurveSimulator(curve, options)
            % CURVESIMULATOR  Construct and configure the physics engine.
            %
            %   sim = CurveSimulator(curve, 'InitialVelocity', 5, 'Dt', 0.01)
            %
            %   Inputs:
            %     curve - OceanMath.Geometry.IParametricCurve
            %
            %   Name-Value Options:
            %     InitialVelocity - starting speed (default: 1)
            %     Acceleration    - constant accel scalar (default: 0)
            %     AccelProfile    - function handle a(time) (overrides Acceleration)
            %     Dt              - time step in seconds (default: 0.01)
            %     Loop            - wrap around at end? (default: false)
            %     MaxVelocity     - speed cap (default: Inf)
            %     StartParameter  - starting t on curve (default: 0)
            arguments
                curve
                options.InitialVelocity (1,1) double = 1
                options.Acceleration    (1,1) double = 0
                options.AccelProfile    = []
                options.Dt              (1,1) double {mustBePositive} = 0.01
                options.Loop            (1,1) logical = false
                options.MaxVelocity     (1,1) double = Inf
                options.StartParameter  (1,1) double {mustBeInRange(options.StartParameter, 0, 1)} = 0
            end

            obj.Curve_  = curve;
            obj.Frenet_ = OceanMath.Geometry.FrenetFrame(curve);
            obj.TotalArcLength_ = curve.totalLength();
            obj.Dt_     = options.Dt;
            obj.Loop_   = options.Loop;
            obj.ClampVelocity_ = options.MaxVelocity;

            % Acceleration mode
            if ~isempty(options.AccelProfile)
                obj.AccelMode_  = "profile";
                obj.AccelFn_    = options.AccelProfile; %Custom acceleration function
                obj.AccelValue_ = 0;
            elseif options.Acceleration == 0
                obj.AccelMode_  = "constant";
                obj.AccelValue_ = 0;
            else
                obj.AccelMode_  = "uniform";
                obj.AccelValue_ = options.Acceleration;
            end

            % Build initial state
            tau0 = options.StartParameter;   % curve parameter (not time)
            v0   = options.InitialVelocity;
            a0   = obj.getAcceleration_(0);
            pos  = curve.evaluate(tau0);
            T    = curve.tangent(tau0);
            normT = sqrt(sum(T.^2));
            if normT > 1e-12
                T = T / normT;
            end
            q = obj.Frenet_.quaternion(tau0);

            obj.CurrentState_ = OceanMath.Simulation.KinematicState( ...
                0, tau0, pos, v0, a0, 0, T, q);

            % Initialize history
            obj.History_ = {obj.CurrentState_};
            obj.StepCount_ = 0;
        end

        % -------------------------------------------------------------- %
        function step(obj, dtOverride)
            % STEP  Advance the simulation by one time step dt.
            arguments
                obj
                dtOverride (1,1) double = -1
            end
            
            if dtOverride > 0
                obj.Dt_ = dtOverride;
            end
            
            % Physics (t = clock time, tau = curve parameter):
            %   ds   = v * dt + 0.5 * a * dt^2   (SUVAT distance)
            %   v'   = v + a * dt                 (velocity update)
            %   tau' = tau + ds / L               (curve parameter advance)
            %
            % At each step, position, tangent, and orientation are
            % recomputed from the curve at the new parameter t'.

            oldState = obj.CurrentState_; %Snapshot before this step

            dt  = obj.Dt_; %time step
            v   = oldState.Velocity; %current speed
            a   = obj.getAcceleration_(oldState.Time + dt); %acceleration at NEW time step

            % --- SUVAT kinematics ---
            ds = v * dt + 0.5 * a * dt^2;   % arc-length increment. How far did we move?
            vNew = v + a * dt;               % new velocity. How fast are we goig now?

            % Clamp velocity
            vNew = max(0, min(vNew, obj.ClampVelocity_));

            % Prevent negative distance (body can't go backward, just stop)
            if ds < 0
                ds = 0;
                vNew = 0;
            end

            % --- Update curve parameter (tau, not time) ---
            % dTau   = ds / max(obj.TotalArcLength_, eps);  % dTau = ds/L approximation
            % tauNew = oldState.Parameter + dTau;
            % Calculate EXACT targeted geometry distance from the beginning of the curve
            targetS = oldState.ArcDistance + ds;
            
            %Handle loop
            if obj.Loop_
                targetS = mod(targetS, obj.TotalArcLength_);
            end
            
            % True Arc-Length Mathematics: Perfect, butter-smooth Train tracking
            tauNew = obj.Curve_.getParameterAtArcLength(targetS, oldState.Parameter);

            % Handle end-of-curve
            if obj.Loop_
                tauNew = mod(tauNew, 1);
            else
                tauNew = min(tauNew, 1);
                if tauNew >= 1
                    vNew = 0;  % stop at end
                end
            end

            % --- Recompute spatial quantities from the curve ---
            posNew  = obj.Curve_.evaluate(tauNew);  % Ask the curve where we are now
            tangRaw = obj.Curve_.tangent(tauNew);    % Ask the curve which way we are facing
            normT   = sqrt(sum(tangRaw.^2));
            if normT > 1e-12
                tangNew = tangRaw / normT;  % Unit tangent vector
            else
                tangNew = oldState.Tangent;
            end
            qNew = obj.Frenet_.quaternion(tauNew);   % Orientation from Frenet frame

            % --- Build new state ---
            timeNew = oldState.Time + dt;       % t  = clock time (seconds)
            distNew = oldState.ArcDistance + ds;

            newState = OceanMath.Simulation.KinematicState( ...
                timeNew, tauNew, posNew, vNew, a, distNew, tangNew, qNew); %Freeze into immutable object

            obj.CurrentState_ = newState; %this is where we are now
            obj.StepCount_ = obj.StepCount_ + 1; %increment step counter

            % Record history
            obj.History_{end+1} = newState;

            % Fire event. Observer pattern. Broadcasts the state change to any listeners.
            evtData = OceanMath.Simulation.CurveStateChangedEventArgs( ...
                oldState, newState);
            notify(obj, 'StateChanged', evtData);
        end

        % -------------------------------------------------------------- %
        function runFor(obj, nSteps)
            % RUNFOR  Step N times.
            arguments
                obj
                nSteps (1,1) double {mustBePositive, mustBeInteger}
            end
            for i = 1:nSteps
                obj.step();
                % Stop early if we hit the end
                if ~obj.Loop_ && obj.CurrentState_.Parameter >= 1
                    break;
                end
            end
        end

        function runUntil(obj, targetTime)
            % RUNUNTIL  Step until simulation clock reaches targetTime.
            arguments
                obj
                targetTime (1,1) double {mustBePositive}
            end
            while obj.CurrentState_.Time < targetTime
                obj.step();
                if ~obj.Loop_ && obj.CurrentState_.Parameter >= 1
                    break;
                end
            end
        end

        % -------------------------------------------------------------- %
        function reset(obj, options)
            % RESET  Return to initial conditions.
            arguments
                obj
                options.StartParameter  (1,1) double = 0
                options.InitialVelocity (1,1) double = 1
            end
            tau0 = options.StartParameter;   % curve parameter (not time)
            v0   = options.InitialVelocity;
            pos  = obj.Curve_.evaluate(tau0);
            T    = obj.Curve_.tangent(tau0);
            normT = sqrt(sum(T.^2));
            if normT > 1e-12; T = T / normT; end
            q = obj.Frenet_.quaternion(tau0);

            a0   = obj.getAcceleration_(0);
            
            % Finalize the true initial state
            obj.CurrentState_ = OceanMath.Simulation.KinematicState( ...
                0, tau0, pos, v0, a0, 0, T, q);
            obj.History_ = {obj.CurrentState_};
            obj.StepCount_ = 0;
        end

        % -------------------------------------------------------------- %
        function s = getState(obj)
            % GETSTATE  Return the current KinematicState.
            s = obj.CurrentState_;
        end

        function h = getHistory(obj)
            % GETHISTORY  Return all recorded KinematicState snapshots.
            h = obj.History_;
        end

        function n = getStepCount(obj)
            % GETSTEPCOUNT  Number of steps taken since last reset.
            n = obj.StepCount_;
        end

        function travelTime = estimateTravelTime(obj, tau0, tau1)
            % ESTIMATETRAVELTIME  Estimate travel time between two curve parameters.
            %   tau0, tau1 are curve parameters in [0,1], NOT clock times.
            %
            %   For constant velocity: time = arcLength / velocity
            %   For uniform acceleration: solve s = v0*t + 0.5*a*t^2
            arguments
                obj
                tau0 (1,1) double = 0
                tau1 (1,1) double = 1
            end
            segLen = obj.Curve_.arcLength(tau0, tau1);
            v = obj.CurrentState_.Velocity;

            if strcmp(obj.AccelMode_, "constant") || obj.AccelValue_ == 0
                % d = v * t  =>  t = d / v
                if v < 1e-12
                    travelTime = Inf;
                else
                    travelTime = segLen / v;
                end
            else
                % s = v0*t + 0.5*a*t^2
                % 0.5*a*t^2 + v0*t - s = 0  (quadratic)
                a = obj.AccelValue_;
                discriminant = v^2 + 2 * a * segLen;
                if discriminant < 0
                    travelTime = Inf;  % can't reach
                else
                    travelTime = (-v + sqrt(discriminant)) / a;
                    if travelTime < 0
                        travelTime = Inf;
                    end
                end
            end
        end

        function info = describe(obj)
            % DESCRIBE  Print summary.
            s = obj.CurrentState_;
            info = sprintf(['CurveSimulator:\n' ...
                '  Curve:        %s\n' ...
                '  Mode:         %s\n' ...
                '  dt:           %.4f s\n' ...
                '  Total Length:  %.4f\n' ...
                '  Steps:        %d\n' ...
                '  Current t:    %.4f\n' ...
                '  Current v:    %.4f\n' ...
                '  Distance:     %.4f\n'], ...
                class(obj.Curve_), obj.AccelMode_, obj.Dt_, ...
                obj.TotalArcLength_, obj.StepCount_, ...
                s.Parameter, s.Velocity, s.ArcDistance);
            fprintf('%s', info);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function a = getAcceleration_(obj, time)
            % GETACCELERATION_  Return acceleration at a given simulation time.
            switch obj.AccelMode_
                case "constant"
                    a = 0;
                case "uniform"
                    a = obj.AccelValue_;
                case "profile"
                    a = obj.AccelFn_(time);
                otherwise
                    a = 0;
            end
        end
    end
end
