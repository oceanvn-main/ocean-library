classdef Event
    % EVENT  Something that happened, emitted by a rule (or a due prediction, or injected from outside) and routed by
    % type to the rules that consume it. A rule never sees an event emitted in its own phase at the same instant
    % (W1_Kernel_Contract.md section 7.2). Payloads are value data only.
    %
    %   e = OceanAgents.World.Event.make("CliffFall", struct('Penalty', -100), 'SourceEntity', 12);   % inside a rule
    % The World stamps Time, Phase, SourceDoor and Seq. Contract: W1_Kernel_Contract.md section 4.3.

    properties (SetAccess = immutable)
        Type string = ""
        Time double = NaN
        Phase string = ""
        SourceDoor string = ""
        SourceEntity double = 0
        Seq double = 0
        Payload struct = struct()
    end

    methods
        function obj = Event(options)
            arguments
                options.Type
                options.Time = NaN
                options.Phase = ""
                options.SourceDoor = ""
                options.SourceEntity = 0
                options.Seq = 0
                options.Payload = struct()
            end
            bad = @(msg) error('OceanAgents:World:BadValue', 'Event: %s', msg);
            if ~isfield(options, 'Type'), bad('Type is required.'); end
            obj.Type = string(options.Type);
            if ~isscalar(obj.Type) || strlength(obj.Type) == 0, bad('Type must be a non-empty string.'); end
            t = options.Time;
            if ~(isnumeric(t) && isscalar(t) && (isnan(t) || (t >= 0 && t == fix(t) && isfinite(t))))
                bad('Time must be a non-negative integer tick (NaN until stamped).');
            end
            obj.Time = double(t);
            obj.Phase = string(options.Phase);
            obj.SourceDoor = string(options.SourceDoor);
            se = options.SourceEntity;
            if ~(isnumeric(se) && isscalar(se) && se >= 0 && se == fix(se)), bad('SourceEntity must be an entity ID or 0.'); end
            obj.SourceEntity = double(se);
            sq = options.Seq;
            if ~(isnumeric(sq) && isscalar(sq) && sq >= 0 && sq == fix(sq)), bad('Seq must be a non-negative integer.'); end
            obj.Seq = double(sq);
            if ~isstruct(options.Payload) || ~isscalar(options.Payload), bad('Payload must be a scalar struct.'); end
            OceanAgents.Core.assertValueData(options.Payload, "Event.Payload");
            obj.Payload = options.Payload;
        end

        function e = stamp(obj, time, phase, door, seq)
            % STAMP  Copy with the World-assigned coordinates (called by the World only).
            e = OceanAgents.World.Event('Type', obj.Type, 'Time', time, 'Phase', phase, 'SourceDoor', door, ...
                'SourceEntity', obj.SourceEntity, 'Seq', seq, 'Payload', obj.Payload);
        end
    end

    methods (Static)
        function e = make(type, payload, options)
            % MAKE  What a rule calls: type, payload and optionally the entity the event is about.
            arguments
                type
                payload = struct()
                options.SourceEntity = 0
            end
            e = OceanAgents.World.Event('Type', type, 'Payload', payload, 'SourceEntity', options.SourceEntity);
        end
    end
end
