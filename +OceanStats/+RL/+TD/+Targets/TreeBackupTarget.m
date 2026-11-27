classdef TreeBackupTarget < OceanStats.RL.TD.Targets.ITarget
    % TREEBACKUPTARGET  Target calculation for n-step Tree Backup algorithm.
    %   Sutton & Barto (2018) Eq. 7.15: Tree Backup without importance sampling.
    properties (Constant)
        Latex = "G_{t:t+n} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a \mid S_{t+1}) Q(S_{t+1},a) + \gamma \pi(A_{t+1} \mid S_{t+1}) G_{t+1:t+n}"
        Reference = "Sutton & Barto (2018) Eq. 7.15, Tree Backup"
        IsOffPolicy = true
    end
    methods
        function y = compute(~, r, qNext, aNext, terminated, gamma, probsNext, valid)
            if terminated
                b = 0;
            else
                probs = probsNext(valid);
                qVals = qNext(valid);
                b = sum(probs .* qVals);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "treebackup"; end
    end
end
