classdef ModelOutcome
    %MODELOUTCOME Immutable outcome in a tabular model distribution.
    properties (SetAccess = immutable)
        NextStateIndex (1,1) double
        Reward (1,1) double
        Terminated (1,1) logical
        Probability (1,1) double
    end
    methods
        function obj = ModelOutcome(nextStateIndex, reward, terminated, probability)
            arguments
                nextStateIndex
                reward
                terminated
                probability
            end
            obj.NextStateIndex = integer_(nextStateIndex, true, "NextStateIndex");
            obj.Reward = scalar_(reward, "Reward");
            if ~(islogical(terminated) && isscalar(terminated)), fail_("Terminated must be a scalar logical."); end
            obj.Terminated = terminated;
            if ~obj.Terminated && obj.NextStateIndex == 0, fail_("A nonterminal outcome requires a positive NextStateIndex."); end
            obj.Probability = scalar_(probability, "Probability");
            if obj.Probability <= 0 || obj.Probability > 1, fail_("Probability must be in (0, 1]."); end
        end
    end
end
function x = scalar_(x, name)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x)), fail_(name + " must be a finite real scalar."); end
x = double(x);
end
function x = integer_(x, allowZero, name)
x = scalar_(x, name);
if x ~= fix(x) || x < double(~allowZero), fail_(name + " has an invalid integer value."); end
end
function fail_(message), error('OceanStats:RL:Dyna:Core:InvalidRecord', '%s', message); end
