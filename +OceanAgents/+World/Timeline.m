classdef Timeline < handle
    % TIMELINE  The world's clock ("GOD", design U3; S1 step S1.2b-1): it holds the current instant and the phase order
    % of an instant, and moves time forward - nothing else. It knows no participants: not doors, devices or agents, and
    % no triggers (who is due when lives in OceanAgents.World.Participants). It holds no rules, commands, events or
    % component data, and nothing asks it for the time in order to poll (U4): components are functions of time, and the
    % World moves the clock. Time has nothing to do with doors (user, 2026-10-12).
    %
    %   tl = OceanAgents.World.Timeline(["resolve" "act" "settle" "project"]);
    %   tl.setNow(10)          % World only: the clock moves forward, never back (tl.reset() returns it to 0)
    %   tl.phaseIndex("act")   % 2

    properties (SetAccess = private)
        Now double = 0                  % current instant (set by the World through setNow/reset)
        Phases string = strings(1, 0)   % phase order within an instant
    end

    methods
        function obj = Timeline(phases)
            arguments
                phases string
            end
            phases = reshape(phases, 1, []);
            if isempty(phases) || any(strlength(phases) == 0) || numel(unique(phases)) ~= numel(phases)
                bad_('BadAssembly', 'Phases must be a non-empty list of unique names.');
            end
            obj.Phases = phases;
        end

        function k = phaseIndex(obj, phase)
            % PHASEINDEX  Position of a phase name in the phase order (error if unknown).
            arguments
                obj
                phase (1,1) string
            end
            k = find(obj.Phases == string(phase), 1);
            if isempty(k), bad_('BadValue', 'Unknown phase "%s".', string(phase)); end
        end

        function setNow(obj, t)
            % SETNOW  World only: move the clock to instant t (t >= Now).
            arguments
                obj
                t (1,1) double
            end
            if ~isTick_(t) || t < obj.Now
                bad_('Retroactive', 'The clock moves forward only: Now = %g, requested %g.', obj.Now, t);
            end
            obj.Now = t;
        end

        function reset(obj)
            % RESET  World only (episode reset): Now = 0.
            obj.Now = 0;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('clock only (U3): %d phases; knows no participants', numel(obj.Phases))};
            meta.IsSymbolic = false;
            meta.Reference = "runtime/container component (W1_Kernel_Contract.md section 5.4)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "Timeline";
            spec.Inputs = struct('Name', {}, 'Domain', {}, 'Dtype', {}, 'Shape', {});
            spec.Outputs = struct('Name', "now", 'Domain', "World-Time", 'Dtype', "double", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "timeline");
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
