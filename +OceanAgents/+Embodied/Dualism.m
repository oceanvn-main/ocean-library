classdef Dualism < handle
    % DUALISM  The pairing of one mind with one body; it exists only while both are present (design D8). A placeholder
    % for future work: today it only checks the pair and joins them (AB1 section 8).
    %
    %   d = OceanAgents.Embodied.Dualism.form(mind, body);
    %   d2 = OceanAgents.Embodied.Dualism.form(mind, otherBody);    % a body swap, between episodes only
    %   delta = d.Signature.diff(d2.Signature);                      % what the mind will find different
    %
    % A pairing is valid when the body accepts every command the mind will send; the body may offer more.

    properties (SetAccess = private)
        Mind
        Body
        Signature
    end

    methods (Access = private)
        function obj = Dualism(mind, body, signature)
            obj.Mind = mind;
            obj.Body = body;
            obj.Signature = signature;
        end
    end

    methods (Static)
        function d = form(mind, body)
            arguments
                mind (1,1) OceanAgents.Minds.SignalMind
                body (1,1) OceanAgents.Body.IBody
            end
            if mind.InEpisode
                error('OceanAgents:Embodied:MidEpisode', 'A mind can change body only between episodes.');
            end
            sig = body.probe();
            if ~sig.accepts(mind.CommandType)
                missing = setdiff(mind.CommandType.Labels, sig.commandLabels(), 'stable');
                error('OceanAgents:Embodied:Incompatible', ...
                    'Body "%s" cannot carry out every command of this mind (missing: %s).', body.ID, ...
                    strjoin(missing, ", "));
            end
            mind.attachBody(body);
            d = OceanAgents.Embodied.Dualism(mind, body, sig);
        end
    end
end
