classdef PolicySignalMind < OceanAgents.Minds.SignalMind
    % POLICYSIGNALMIND  Any existing policy as a signal mind that sees the world only through observations (AB2): the
    % policy is shown a view built from the mind's perceptions, never the world's view. With RandomLegalPolicy and a
    % moves observation only, it is the simple tic-tac-toe player of AB2 success test O6.
    %
    %   mind = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.RandomLegalPolicy(), 1:9);
    %   OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:9), ...
    %       'Sensors', {OceanAgents.Body.Sensors.MovesObservation(string(1:9))}));

    properties (SetAccess = private)
        Policy
        Labels string
    end

    properties (Access = private)
        RawLabels_
    end

    methods
        function obj = PolicySignalMind(policy, labels, options)
            arguments
                policy (1,1) OceanAgents.Core.IPolicy
                labels
                options.Perceptions = []
            end
            obj.Policy = policy;
            obj.addPolicy("main", policy);
            obj.RawLabels_ = labels;
            obj.Labels = reshape(string(labels), 1, []);
            obj.CommandType = OceanAgents.Body.IBody.moveType(obj.Labels);
            perceptions = options.Perceptions;
            if isempty(perceptions)
                % what the Laws permit and any running behavior; a policy that needs a state is given a
                % DigitPerception (or another state perception) explicitly
                perceptions = {OceanAgents.Minds.Perception.MovesPerception(obj.Labels), ...
                    OceanAgents.Minds.Perception.BehaviorPerception()};
            end
            for k = 1:numel(perceptions), obj.addPerception(perceptions{k}); end
        end

        function d = describe(obj)
            d = struct('Name', "PolicySignalMind", 'Profile', "signal-mind", ...
                'Components', struct('Policy', obj.Policy.describe()));
        end

        function checkCompatible(obj, obsSpace, actSpace, role)
            obj.Policy.checkCompatible(obsSpace, actSpace, role);
        end

        function beginEpisode(obj, view)
            beginEpisode@OceanAgents.Minds.SignalMind(obj, view);
            obj.beginPolicies_(view);                            % bookkeeping only, never the world (AB2 I10)
        end

        function dec = decide(obj, view, stream)
            dec = obj.decideWith_(view, stream, obj.Labels, obj.RawLabels_);
        end

        function observe(~, ~)
            % OBSERVE  The world's feedback is not passed to the policy (AB2 I10); the mind learns through learn().
        end

        function endEpisode(obj)
            obj.endPolicies_();
            endEpisode@OceanAgents.Minds.SignalMind(obj);
        end

        function s = exportState(obj)
            s = struct('Name', "PolicySignalMind", 'Policy', obj.Policy.exportState());
        end

        function importState(obj, s)
            if ~isfield(s, 'Name') || s.Name ~= "PolicySignalMind"
                error('OceanAgents:Policies:Incompatible', 'State is not a PolicySignalMind state.');
            end
            obj.Policy.importState(s.Policy);
        end
    end
end
