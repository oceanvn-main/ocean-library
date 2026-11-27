classdef Participants < handle
    % PARTICIPANTS  Who is due when (S1 step S1.2b-1): the registered participants of a world and the trigger each one
    % declared. GOD (OceanAgents.World.Timeline) only holds and moves time; it does not know who takes part. Each
    % participant registers itself with a trigger, and the world asks this registry which participants are due at t and
    % when the next one is. Participants are identified by name only: a door, a device or any other part registers the
    % same way, and time never knows which kind it is.
    %
    %   P = OceanAgents.World.Participants();
    %   P.register("mover", "every-tick");  P.register("spawner", "every-k", 5);  P.register("counter", "on-event");
    %   P.due(10)          % [true true false]   (every-k: mod(t, K) == 0; on-event / never: never due by the clock)
    %   P.nextDue(10)      % 11                  smallest t > 10 at which ANY participant is due (Inf if none ever is)
    %
    % Triggers: "every-tick" due at every t; "every-k" due at t with mod(t, K) == 0 (so at t = 0 too); "on-event" never
    % due by the clock (it runs only when something is delivered to it); "never" never due.
    % nextDue is exact (closed form, no scan): every-tick -> after + 1; every-k -> K * (floor(after / K) + 1).

    properties (SetAccess = private)
        Names string = strings(1, 0)        % participants in registration order
        Triggers string = strings(1, 0)     % trigger of each participant
        K double = zeros(1, 0)              % period of each participant (1 unless "every-k")
    end

    properties (Constant)
        TriggerKinds = ["every-tick" "every-k" "on-event" "never"]
    end

    methods
        function register(obj, name, trigger, k)
            % REGISTER  Add a participant with its trigger (and period K for "every-k").
            arguments
                obj
                name (1,1) string
                trigger (1,1) string
                k = 1
            end
            if strlength(name) == 0, bad_('BadAssembly', 'A participant needs a non-empty name.'); end
            if any(obj.Names == name), bad_('BadAssembly', 'Participant names must be unique ("%s" twice).', name); end
            if ~any(trigger == obj.TriggerKinds)
                bad_('BadAssembly', 'Participant %s: trigger must be one of %s.', name, strjoin(obj.TriggerKinds, ', '));
            end
            if trigger == "every-k"
                if ~isTick_(k) || k < 1
                    bad_('BadAssembly', 'Participant %s: every-k needs a positive integer K.', name);
                end
            else
                k = 1;
            end
            obj.Names(end + 1) = name;
            obj.Triggers(end + 1) = trigger;
            obj.K(end + 1) = double(k);
        end

        function tf = due(obj, t)
            % DUE  1 x nParticipants logical: which participants are due at instant t by their trigger.
            arguments
                obj
                t (1,1) double
            end
            if ~isTick_(t), bad_('BadValue', 'due(t) needs a non-negative integer tick, got %g.', t); end
            tf = obj.Triggers == "every-tick" | (obj.Triggers == "every-k" & mod(t, obj.K) == 0);
        end

        function t = nextDue(obj, after)
            % NEXTDUE  Smallest integer t > after at which any participant is due; Inf if none ever is.
            % after may be -1 (so nextDue(-1) is the first due instant, possibly 0).
            arguments
                obj
                after (1,1) double
            end
            if ~(isfinite(after) && after == fix(after) && after >= -1)
                bad_('BadValue', 'nextDue(after) needs an integer >= -1, got %g.', after);
            end
            t = Inf;
            if any(obj.Triggers == "every-tick")
                t = after + 1;
                return
            end
            k = obj.K(obj.Triggers == "every-k");
            if ~isempty(k)
                t = min(k .* (floor(after ./ k) + 1));   % next multiple of each K strictly after 'after'
            end
        end

        function n = count(obj)
            n = numel(obj.Names);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('%d registered participants; every-k due at mod(t,K) = 0; time never knows their kind', ...
                numel(obj.Names))};
            meta.IsSymbolic = false;
            meta.Reference = "S1_Rules_Out_Of_Doors_Contract.md section 4 (S1.2b-1)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "Participants";
            spec.Inputs = struct('Name', "t", 'Domain', "World-Time", 'Dtype', "double", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "due", 'Domain', "Participants", 'Dtype', "logical", 'Shape', "1 x n");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "participants");
            spec.ComposesFrom = {};
        end
    end
end

function tf = isTick_(t)
tf = isnumeric(t) && isscalar(t) && isfinite(t) && t >= 0 && t == fix(t);
end

function bad_(reason, fmt, varargin)
error(sprintf('OceanAgents:World:%s', reason), fmt, varargin{:});
end
