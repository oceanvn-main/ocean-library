classdef (Abstract) IComponent < handle
    % ICOMPONENT  An active part that holds its own data and changes it only in response to a signal: it VERIFIES the
    % signal, then ACTS (design D10; AB1_Signal_Body_Contract.md section 4). A refused signal gets a refusal reply and
    % changes nothing. There are no setters for a component's state: requests, never overwrites.
    %
    %   replies = component.receive(signal)          % sealed template: type check -> verify -> act
    %   t = component.nextChange(after)              % next self-change (Inf = none), for the calendar time binding
    %   component.useBinding("formula")              % configuration: one of TimeBindings (design D13)
    %
    % Subclasses set ID, Accepts, Emits and TimeBindings in their constructor and implement verify and act as
    % protected methods. importState is a restore path for checkpoints between runs, not a way to drive a component.

    properties (SetAccess = protected)
        ID string = ""
        Accepts = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
        Emits = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
        TimeBindings string = "calendar"        % supported: "formula" | "clock" | "calendar" | "array"
        Binding string = "calendar"             % chosen (default: the first supported, the cheapest exact one)
    end

    properties (Constant)
        AllBindings = ["formula" "clock" "calendar" "array"]
    end

    methods (Abstract, Access = protected)
        [ok, reason] = verify(obj, signal)       % only receive() calls these: verify can never be skipped
        out = act(obj, signal)
    end

    methods (Sealed)
        function out = receive(obj, signal)
            % RECEIVE  Type check, then verify, then act. Returns a row of Signals (replies and emitted signals).
            if ~any(arrayfun(@(a) a.accepts(signal.Type), obj.Accepts))
                out = OceanAgents.Minds.Signals.Signal.refusal(sprintf('%s does not accept %s', obj.ID, ...
                    signal.Type.describe()));
                return
            end
            [ok, reason] = obj.verify(signal);
            if ~ok
                out = OceanAgents.Minds.Signals.Signal.refusal(reason);
                return
            end
            out = obj.act(signal);
            if isempty(out), out = OceanAgents.Minds.Signals.Signal.empty(1, 0); end
            out = reshape(out, 1, []);
        end

        function useBinding(obj, binding)
            % USEBINDING  Choose how this component is bound to time (configuration, design D13).
            binding = string(binding);
            if ~any(binding == obj.TimeBindings)
                error('OceanAgents:Components:BadBinding', '%s supports the time bindings [%s], not "%s".', ...
                    obj.ID, strjoin(obj.TimeBindings, ', '), binding);
            end
            obj.Binding = binding;
        end
    end

    methods (Sealed)
        function k = nextTick(obj, after)
            % NEXTTICK  The first integer tick at or after nextChange(after), strictly after `after`. A predicted
            % time within floating-point noise of a tick (relative 1e-9) is that tick, so a calendar that jumps to
            % nextTick samples exactly the ticks a stepper would act on (AB1 invariant I8; critic AB1 finding 5).
            t = obj.nextChange(after);
            k = ceil(t - 1e-9 * max(1, abs(t)));
            if k <= after, k = floor(after) + 1; end
        end
    end

    methods
        function t = nextChange(~, ~)
            % NEXTCHANGE  Time of the next change this component makes by itself (Inf: none).
            t = Inf;
        end

        function s = exportState(~)
            s = struct();
        end

        function importState(~, ~)
        end

        function d = describe(obj)
            d = struct('ID', obj.ID, 'Class', string(class(obj)), ...
                'Accepts', strjoin(arrayfun(@(a) a.describe(), obj.Accepts), "; "), ...
                'Emits', strjoin(arrayfun(@(a) a.describe(), obj.Emits), "; "), 'Binding', obj.Binding);
        end
    end

    methods (Access = protected)
        function setDeclarations_(obj, id, accepts, emits, bindings)
            % SETDECLARATIONS_  For subclass constructors only.
            obj.ID = string(id);
            obj.Accepts = reshape(accepts, 1, []);
            if isempty(emits), emits = OceanAgents.Minds.Signals.SignalType.empty(1, 0); end
            obj.Emits = reshape(emits, 1, []);
            bindings = reshape(string(bindings), 1, []);
            if ~all(ismember(bindings, obj.AllBindings))
                error('OceanAgents:Components:BadBinding', '%s: unknown time binding.', id);
            end
            obj.TimeBindings = bindings;
            obj.Binding = bindings(1);
        end
    end
end
