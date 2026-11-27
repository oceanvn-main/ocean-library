classdef ThresholdConverter < OceanAgents.Components.IConverter
    % THRESHOLDCONVERTER  Number readings -> flag events, with hysteresis: fires true when the reading reaches Up while
    % low, false when it falls to Down while high (Down < Up, so a reading hovering at one level cannot chatter).
    % Like a neuron's threshold followed by a refractory margin (Research_Admission_and_Medium_Interfaces.md 3.1).
    %
    %   c = OceanAgents.Components.Converters.ThresholdConverter(2.5, 1.5);          % sense role, readings at rate 1
    %   replies = c.receive(reading)                       % reading: a number stream sample
    %   c.receive(waveSignal);  t = c.nextChange(after)    % calendar binding: exact time of the next switch
    %
    % Skipping equals stepping: sampling only at ceil(nextChange) gives the same events, at the same integer ticks,
    % as sampling every tick (AB1 invariant I8).

    properties (SetAccess = private)
        Up double
        Down double
        High (1,1) logical = false
    end

    properties (Access = private)
        Wave_ = []
    end

    methods
        function obj = ThresholdConverter(up, down, options)
            arguments
                up (1,1) double {mustBeFinite}
                down (1,1) double {mustBeFinite}
                options.Role (1,1) string = "sense"
                options.Rate (1,1) double {mustBePositive} = 1
                options.ID (1,1) string = "threshold"
            end
            if ~(down < up)
                error('OceanAgents:Signals:BadValue', 'ThresholdConverter: Down (%g) must be below Up (%g).', down, up);
            end
            T = @(varargin) OceanAgents.Minds.Signals.SignalType(varargin{:});
            stream = T('Shape', "stream", 'ValueKind', "number", 'Role', options.Role, 'Rate', options.Rate);
            wave = T('Shape', "wave", 'ValueKind', "number", 'Role', options.Role);
            out = T('Shape', "event", 'ValueKind', "flag", 'Role', options.Role);
            obj.setDeclarations_(options.ID, [stream wave], out, ["calendar" "clock"]);
            obj.InType = stream;
            obj.OutType = out;
            obj.Up = up;
            obj.Down = down;
        end

        function t = nextChange(obj, after)
            % NEXTCHANGE  Exact time the watched wave next reaches the level that would switch the state.
            t = Inf;
            if isempty(obj.Wave_), return; end
            if obj.High
                t = obj.Wave_.nextReach(obj.Down, after, "down");
            else
                t = obj.Wave_.nextReach(obj.Up, after, "up");
            end
        end

        function s = exportState(obj)
            s = struct('High', obj.High);
        end

        function importState(obj, s)
            obj.High = logical(s.High);
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(~, ~)
            ok = true;  reason = "";
        end

        function out = act(obj, signal)
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
            if signal.Type.Shape == "wave"
                obj.Wave_ = signal.Value;                 % the flow to watch; nothing happens until a reading
                return
            end
            v = signal.Value;
            if ~obj.High && v >= obj.Up
                obj.High = true;
                out = OceanAgents.Minds.Signals.Signal.make(obj.OutType, true);
            elseif obj.High && v <= obj.Down
                obj.High = false;
                out = OceanAgents.Minds.Signals.Signal.make(obj.OutType, false);
            end
        end
    end
end
