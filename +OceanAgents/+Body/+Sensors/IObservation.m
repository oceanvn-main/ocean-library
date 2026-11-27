classdef (Abstract) IObservation < OceanAgents.Components.IComponent
    % IOBSERVATION  A sensor carried by a body: it turns what the world shows (the view) or the body's own state into
    % one typed observation signal for the mind (design D20, D21; AB2_Observation_Contract.md section 4). Like every
    % component it verifies, then acts; an observation that does not match its declared type is refused.
    %
    %   s = sensor.observe(view, body)     % the observation signal (OutType); throws Body:BadObservation if refused
    %
    % Subclasses set Modality and OutType, implement read_ (what to read) and act (build the observation).

    properties (SetAccess = protected)
        Modality string = ""
        OutType = OceanAgents.Minds.Signals.SignalType()
    end

    methods (Sealed)
        function s = observe(obj, view, body)
            try
                r = obj.read_(view, body);
            catch err
                error('OceanAgents:Body:BadObservation', 'Sensor "%s" could not read: %s', obj.ID, err.message);
            end
            out = obj.act(struct('Value', r));        % the reading stays inside the sensor: no signal needed
            if isempty(out), out = OceanAgents.Minds.Signals.Signal.empty(1, 0); end
            if isempty(out) || out(1).isRefusal()
                reason = "no observation";
                if ~isempty(out), reason = out(1).Value.Reason; end
                error('OceanAgents:Body:BadObservation', 'Sensor "%s" could not observe: %s', obj.ID, reason);
            end
            s = out(1);
        end
    end

    methods (Static)
        function t = readingType()
            % READINGTYPE  What a sensor reads (a record, internal to the body). Cached.
            persistent cached
            if isempty(cached), cached = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "sense"); end
            t = cached;
        end
    end

    methods (Abstract, Access = protected)
        r = read_(obj, view, body)               % a scalar struct (value data)
    end

    methods (Access = protected)
        function init_(obj, id, modality, outType)
            obj.setDeclarations_(id, OceanAgents.Body.Sensors.IObservation.readingType(), outType, "calendar");
            obj.Modality = string(modality);
            obj.OutType = outType;
        end

        function [ok, reason] = verify(~, ~)
            ok = true;  reason = "";
        end

        function out = emit_(obj, value)
            % EMIT_  The observation signal, or a refusal when value does not fit OutType.
            try
                out = OceanAgents.Minds.Signals.Signal.make(obj.OutType, value);
            catch err
                if ~strcmp(err.identifier, 'OceanAgents:Signals:BadValue'), rethrow(err); end
                out = OceanAgents.Minds.Signals.Signal.refusal(err.message);
            end
        end
    end
end
