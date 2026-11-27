classdef Context < handle
    % CONTEXT  Everything a backup needs for one run, handed to it explicitly (a backup is a stateless strategy):
    % the Sampler (environment), the value function, the behaviour policy, the target / step-size / trace strategies, the
    % discount and trace parameters, the observer and the run's random stream. progress in [0,1] is the fraction of the
    % run completed (exploration schedules read it).
    properties
        sampler
        values
        policy
        target
        stepSize
        trace
        observer
        stream
        gamma (1,1) double = 1
        lambda (1,1) double = 0.9
        n (1,1) double = 1
        offline (1,1) logical = false
        progress (1,1) double = 0
        extra struct = struct()          % free-form settings for a backup that needs one the learner does not know ('Extra' option)
    end
    methods
        function p = behaviour(obj, si)
            % behaviour probabilities at state index si (1 x A)
            p = obj.policy.probabilities(obj.values, si, obj.sampler.ValidMask(si, :), obj.progress);
        end
        function a = act(obj, si)
            % choose an action index at si from the behaviour policy
            a = obj.policy.sample(obj.behaviour(si), obj.stream);
        end
        function a = draw(obj, p)
            % draw an action index from given probabilities
            a = obj.policy.sample(p, obj.stream);
        end
    end
end
