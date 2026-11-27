function [stateIndex, actionIndex] = pairFromID(pairID, numStates, numActions)
%PAIRFROMID Decode a column-major linear state-action pair ID.
values = [pairID numStates numActions];
if ~(isnumeric(values) && isreal(values) && all(isfinite(values)) && all(values == fix(values)) && all(values > 0))
    fail_("PairID and dimensions must be positive integers.");
end
if pairID > numStates * numActions
    fail_("PairID exceeds the declared table dimensions.");
end
stateIndex = mod(double(pairID) - 1, double(numStates)) + 1;
actionIndex = floor((double(pairID) - 1) / double(numStates)) + 1;
end
function fail_(message), error('OceanStats:RL:Dyna:Functional:InvalidPair', '%s', message); end
