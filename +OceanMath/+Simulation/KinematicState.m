classdef KinematicState
    % KINEMATICSTATE  Immutable value object describing a body's physical state on a curve.
    %
    % This lightweight struct-like class holds the complete kinematic
    % snapshot at a single instant. It is the payload carried by
    % CurveSimulator's StateChanged event.
    %
    % Properties:
    %   Time         - Simulation clock (seconds)
    %   Parameter    - Curve parameter t in [0, 1] (where on the curve)
    %   Position     - [1 x d] spatial coordinate [x, y, z]
    %   Velocity     - Scalar speed along the curve (units/s)
    %   Acceleration - Scalar acceleration along the curve (units/s^2)
    %   ArcDistance   - Cumulative distance traveled (units)
    %   Tangent      - [1 x d] unit tangent vector (direction of motion)
    %   Orientation  - [1 x 4] quaternion [w,x,y,z] (from FrenetFrame)
    %
    % Design:
    %   - Value semantics (NOT handle): safe to store in arrays/history.
    %   - All properties are SetAccess = immutable — snapshot is frozen.
    %   - Works for both 2D and 3D curves.
    %
    % See also: OceanMath.Simulation.CurveSimulator

    % ------------------------------------------------------------------ %
    properties (SetAccess = immutable)
        Time          (1,1) double    % simulation clock (seconds)
        Parameter     (1,1) double    % curve parameter t in [0, 1]
        Position      (1,:) double    % [x, y, z] spatial coordinate
        Velocity      (1,1) double    % scalar speed (units/s)
        Acceleration  (1,1) double    % scalar acceleration (units/s^2)
        ArcDistance   (1,1) double    % cumulative distance traveled
        Tangent       (1,:) double    % unit tangent direction
        Orientation   (1,4) double    % quaternion [w, x, y, z]
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = KinematicState(time, param, pos, vel, accel, dist, tang, orient)
            % KINEMATICSTATE  Construct a frozen state snapshot.
            %
            %   state = KinematicState(time, t, pos, vel, accel, dist, tangent, quat)
            arguments
                time    (1,1) double
                param   (1,1) double
                pos     (1,:) double
                vel     (1,1) double
                accel   (1,1) double
                dist    (1,1) double
                tang    (1,:) double
                orient  (1,4) double = [1 0 0 0]
            end
            obj.Time         = time;
            obj.Parameter    = param;
            obj.Position     = pos;
            obj.Velocity     = vel;
            obj.Acceleration = accel;
            obj.ArcDistance   = dist;
            obj.Tangent      = tang;
            obj.Orientation  = orient;
        end

        function disp(obj)
            % DISP  Pretty-print the kinematic state.
            fprintf(['  KinematicState:\n' ...
                '    Time:         %.4f s\n' ...
                '    Parameter:    %.4f\n' ...
                '    Position:     [%s]\n' ...
                '    Velocity:     %.4f units/s\n' ...
                '    Acceleration: %.4f units/s^2\n' ...
                '    ArcDistance:   %.4f units\n'], ...
                obj.Time, obj.Parameter, ...
                strjoin(arrayfun(@(x) sprintf('%.3f', x), obj.Position, 'UniformOutput', false), ', '), ...
                obj.Velocity, obj.Acceleration, obj.ArcDistance);
        end
    end
end
