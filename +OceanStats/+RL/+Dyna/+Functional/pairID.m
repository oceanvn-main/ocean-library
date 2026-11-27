function id = pairID(stateIndex, actionIndex, numStates, numActions)
%PAIRID Column-major linear ID for a state-action pair.
values = [stateIndex actionIndex numStates numActions];
if ~(isnumeric(values) && isreal(values) && all(isfinite(values)) && all(values == fix(values)) && all(values > 0))
    fail_("Indices and dimensions must be positive integers.");
end
if stateIndex > numStates || actionIndex > numActions
    fail_("StateIndex or ActionIndex exceeds its declared dimension.");
end
id = double(stateIndex) + (double(actionIndex) - 1) * double(numStates);
end
function fail_(message), error('OceanStats:RL:Dyna:Functional:InvalidPair', '%s', message); end
