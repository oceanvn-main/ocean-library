classdef (Abstract) IPerception < handle
    % IPERCEPTION  The mind's side of an observation: it accepts one type of observation and turns it into what the
    % mind's policy and learners use (AB2_Observation_Contract.md section 6). At join every Required perception must
    % find a sensor of the body whose observation type it accepts (by type, never by name; AB2 I11).
    %
    %   Use  "state"    : becomes the policy's observation (e.g. the cell index)
    %        "mask"     : becomes the set of moves the Laws permit now
    %        "behavior" : tells the mind whether a body behavior is running

    properties (SetAccess = protected)
        ID string = ""
        Use string = "state"
        Accepts = OceanAgents.Minds.Signals.SignalType()
        Required logical = true
    end

    methods (Abstract)
        x = perceive(obj, signal)                % signal: the observation as delivered (Type and Value)
    end

    methods
        function tf = canRead(obj, offered)
            % CANREAD  Whether an observation of type offered can be perceived (default: Accepts accepts it).
            tf = obj.Accepts.accepts(offered);
        end
    end
end
