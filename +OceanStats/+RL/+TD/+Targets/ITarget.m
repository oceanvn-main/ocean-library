classdef (Abstract) ITarget < OceanStats.RL.TD.ITDComponent
    % ITARGET  Strategy: the number the estimate is moved toward after one transition.
    %
    %   y = target.compute(r, qNext, aNext, terminated, gamma, probsNext, valid)
    %
    %     r           reward R_{t+1}
    %     qNext       estimates at S_{t+1}: a (1 x A) row of action values (a scalar for state values)
    %     aNext       index of the action A_{t+1} actually selected at S_{t+1} (1 for state values)
    %     terminated  true MDP terminal (bootstrap dropped); truncation is NOT terminal
    %     gamma       discount
    %     probsNext   (1 x A) probabilities of the behaviour policy at S_{t+1}
    %     valid       (1 x A) logical mask of admissible actions at S_{t+1}
    %
    % The rule is pure: it reads its arguments and returns a number. It owns no tables, environment or
    % random numbers - the learner (OceanStats.RL.TD.TDLearner) does. Multi-step returns are separate pure
    % functions (Functional.nStepReturn / lambdaReturns) driven by the learner's modes.
    %
    % IsOffPolicy: true if the target policy differs from the behaviour policy (Q-learning). The trace
    % mode uses it to cut traces after exploratory actions (Watkins's Q(lambda)).

    properties (Abstract, Constant)
        IsOffPolicy (1,1) logical
    end

    methods (Abstract)
        y = compute(obj, r, qNext, aNext, terminated, gamma, probsNext, valid)
    end

    % Capabilities the learner reads INSTEAD of testing for concrete classes, so a new target needs no learner edit.
    methods
        function k = valueKind(~)
            % "action" (Q(s,a), control) or "state" (V(s), prediction)
            k = "action";
        end
        function tf = supportsDouble(~)
            % true if the rule is "evaluate the greedy action selected by the other table" (Double Q-learning)
            tf = false;
        end
        function b = bootstrap(obj, qNext, aNext, probsNext, valid)
            % the state-value estimate this rule bootstraps from at S_{t+1} (the tail of n-step returns)
            b = obj.compute(0, qNext, aNext, false, 1, probsNext, valid);
        end
    end
end
