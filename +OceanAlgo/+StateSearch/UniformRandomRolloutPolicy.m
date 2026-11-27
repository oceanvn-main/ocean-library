classdef UniformRandomRolloutPolicy < OceanAlgo.StateSearch.IRolloutPolicy
    % UNIFORMRANDOMROLLOUTPOLICY  Equiprobable choice among the legal actions. The text notes a rollout
    % algorithm can do well "even if the rollout policy is completely random" (S&B 2nd ed., Section 8.10).
    %
    %   policy = OceanAlgo.StateSearch.UniformRandomRolloutPolicy();
    methods
        function a = act(~, model, s, stream)
            arguments
                ~
                model (1,1) OceanAlgo.StateSearch.ISearchModel
                s
                stream (1,1) RandStream
            end
            actions = model.actions(s);
            if isempty(actions)
                error('OceanAlgo:StateSearch:NoLegalAction', 'The state admits no action.');
            end
            a = actions(randi(stream, numel(actions)));
        end
        function name = describe(~), name = "uniform random"; end
    end
end
