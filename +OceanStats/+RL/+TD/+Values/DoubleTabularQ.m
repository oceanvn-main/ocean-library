classdef DoubleTabularQ < OceanStats.RL.TD.Values.TabularActionValues
    % DOUBLETABULARQ  Two independent action-value tables (van Hasselt 2010). Each update flips a fair coin: one table
    % selects the greedy action at the successor, the OTHER table evaluates it, and only the selecting table is updated
    % (each with its own visit counts). Behaviour acts on the average of the two tables; the final Q is that average.
    % It ignores the target's own bootstrap (it needs a target that declares supportsDouble(), i.e. a greedy target) and
    % must update BEFORE the next action is chosen, because the action is drawn from the updated average.
    properties (Constant)
        Latex = "Q_A(s,a) \leftarrow Q_A(s,a) + \alpha \left[ r + \gamma\, Q_B\!\left(s', \arg\max_{a'} Q_A(s',a')\right) - Q_A(s,a) \right]"
        Reference = "van Hasselt (2010); Sutton & Barto (2018) sec. 6.7, Double Q-learning"
    end
    properties (SetAccess = private)
        Q2 double = []
        Counts2 double = []
    end
    methods
        function obj = DoubleTabularQ(validMask, initialValue)
            arguments
                validMask logical
                initialValue (1,1) double = 0
            end
            obj@OceanStats.RL.TD.Values.TabularActionValues(validMask, initialValue);
            obj.Q2 = obj.Q;
            obj.Counts2 = zeros(size(validMask));
        end
        function s = name(~), s = "doubletabularq"; end
        function x = behaviourEstimates(obj, si), x = (obj.Q(si, :) + obj.Q2(si, :)) / 2; end
        function tf = updatesBeforeNextAction(~), tf = true; end
        function tf = requiresOneStepBackup(~), tf = true; end
        function tf = requiresGreedyTarget(~), tf = true; end
        function coin = oneStep(obj, ~, stepSize, si, ai, r, s2i, ~, term, gamma, ~, valid2, stream)
            coin = rand(stream) < 0.5;
            if coin
                A = obj.Q;  B = obj.Q2;
            else
                A = obj.Q2; B = obj.Q;
            end
            if term
                boot = 0;
            else
                q = A(s2i, :);  q(~valid2) = -Inf;
                [~, astar] = max(q);
                boot = B(s2i, astar);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, boot, term, gamma);
            if coin
                obj.Counts(si, ai) = obj.Counts(si, ai) + 1;
                alpha = stepSize.alpha(obj.Counts(si, ai));
                obj.Q(si, ai) = obj.Q(si, ai) + alpha * (y - obj.Q(si, ai));
            else
                obj.Counts2(si, ai) = obj.Counts2(si, ai) + 1;
                alpha = stepSize.alpha(obj.Counts2(si, ai));
                obj.Q2(si, ai) = obj.Q2(si, ai) + alpha * (y - obj.Q2(si, ai));
            end
        end
        function Q = finalQ(obj), Q = (obj.Q + obj.Q2) / 2; end
    end
end
