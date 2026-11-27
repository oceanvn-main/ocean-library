function priority = priorityMagnitude(reward, nextQ, currentQ, terminated, gamma, validMask)
%PRIORITYMAGNITUDE Absolute TD error used by prioritized sweeping.
target = OceanStats.RL.Dyna.Functional.qLearningTarget( ...
    reward, nextQ, terminated, gamma, validMask);
if ~(isnumeric(currentQ) && isreal(currentQ) && isscalar(currentQ) && isfinite(currentQ))
    error('OceanStats:RL:Dyna:Functional:InvalidInput', ...
        'currentQ must be a finite real scalar.');
end
priority = abs(target - double(currentQ));
end
