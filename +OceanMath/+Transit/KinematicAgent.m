classdef KinematicAgent < handle
    properties (SetAccess = private)
        Network_
        Splines_
        Speed_
        Dt_
        
        CurrentSegIdx_ = 1
        State_ % KinematicState
    end
    
    methods
        function obj = KinematicAgent(network, lineCode, options)
            arguments
                network
                lineCode   char
                options.Speed double = 10
                options.Dt    double = 0.02
            end
            obj.Network_ = network;
            obj.Speed_ = options.Speed;
            obj.Dt_ = options.Dt;
            
            line = network.getLine(lineCode);
            route = line.getRoute();
            
            nSeg = numel(route) - 1;
            obj.Splines_ = cell(1, nSeg);
            for i = 1:nSeg
                edgeIdx = network.findEdgeIndex(route(i), route(i+1));
                obj.Splines_{i} = network.getSpline(edgeIdx);
            end
            
            % Initial State
            tau0 = 0;
            v0 = options.Speed;
            pos = obj.Splines_{1}.evaluate(tau0);
            tang = obj.Splines_{1}.tangent(tau0);
            tang = tang / norm(tang);
            frenet = OceanMath.Geometry.FrenetFrame(obj.Splines_{1});
            q = frenet.quaternion(tau0);
            
            obj.State_ = OceanMath.Simulation.KinematicState(0, tau0, pos, v0, 0, 0, tang, q);
        end
        
        function step(obj)
            % Gravity Physics Step
            if obj.CurrentSegIdx_ > numel(obj.Splines_)
                return; % Arrived
            end
            
            spline = obj.Splines_{obj.CurrentSegIdx_};
            st = obj.State_;
            
            dt = obj.Dt_;
            v = st.Velocity;
            
            % 1. Compute Gravity Acceleration
            % T_z is the vertical component of the unit tangent vector
            T_z = 0;
            if length(st.Tangent) >= 3
                T_z = st.Tangent(3);
            end
            g = 45.81; % Magnified gravity for visual thrills
            a = -g * T_z;
            
            % Add friction
            friction = 0.1 * v;
            a = a - friction;
            
            % Motorized chain-lift: min speed going uphill
            minSpeed = 15.0;
            
            % 2. SUVAT integration
            ds = v * dt + 0.5 * a * dt^2;
            vNew = v + a * dt;
            
            if vNew < minSpeed
                vNew = minSpeed; % Chain lift engages
            end
            
            % Prevent negative distance (rollback disabled)
            if ds < 0
                ds = 0;
                vNew = 0;
            end
            
            % 3. Parameter Advance
            L = spline.totalLength();
            dTau = ds / L;
            tauNew = st.Parameter + dTau;
            
            timeNew = st.Time + dt;
            distNew = st.ArcDistance + ds;
            
            % Check for segment transition
            if tauNew >= 1.0
                obj.CurrentSegIdx_ = obj.CurrentSegIdx_ + 1;
                if obj.CurrentSegIdx_ > numel(obj.Splines_)
                    % End of track
                    obj.State_ = OceanMath.Simulation.KinematicState(timeNew, 1.0, st.Position, 0, 0, distNew, st.Tangent, st.Orientation);
                    return;
                end
                
                % Roll over to next spline
                spline = obj.Splines_{obj.CurrentSegIdx_};
                tauNew = tauNew - 1.0;
            end
            
            % Re-evaluate kinematics on current spline
            posNew = spline.evaluate(tauNew);
            tangRaw = spline.tangent(tauNew);
            tangNew = tangRaw / norm(tangRaw);
            
            frenet = OceanMath.Geometry.FrenetFrame(spline);
            qNew = frenet.quaternion(tauNew);
            
            obj.State_ = OceanMath.Simulation.KinematicState(timeNew, tauNew, posNew, vNew, a, distNew, tangNew, qNew);
        end
        
        function st = getState(obj)
            st = obj.State_;
        end
        
        function tf = isArrived(obj)
            tf = obj.CurrentSegIdx_ > numel(obj.Splines_);
        end
        
        function spl = getActiveSpline(obj)
            if obj.isArrived()
                spl = [];
            else
                spl = obj.Splines_{obj.CurrentSegIdx_};
            end
        end
        
        function p = getProgress(obj)
            if obj.isArrived()
                p = 1.0;
            else
                p = obj.State_.Parameter;
            end
        end
        
        function name = getAgentState(obj)
             if obj.isArrived()
                 name = 'dwelling'; % For visual border flash
             else
                 name = 'running';
             end
        end
    end
end
