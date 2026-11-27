function report = validatePrescribedDyna(expectedFile)
%VALIDATEPRESCRIBEDDYNA Compare Ocean Dyna primitives with a Python oracle.
%   Random choices are explicit prescribed U[0,1) variates. This makes the
%   comparison independent of MATLAB/Python random-number generators.
arguments
    expectedFile (1,1) string = fullfile(fileparts(mfilename('fullpath')), ...
        'prescribed_dyna_expected.json')
end
expected = jsondecode(fileread(expectedFile));
p = expected.parameters;
mask = true(p.numStates, p.numActions);
model = OceanStats.RL.Dyna.Models.DeterministicTabularModel( ...
    p.numStates, p.numActions, mask);
q = zeros(p.numStates, p.numActions);
transitions = [1 1 0.0 2 0; 2 1 1.0 3 1; 1 2 0.2 1 0; 2 2 -0.1 1 0];
eligible = zeros(0, 1); updateIndex = 0; actualSnapshots = cell(8, 1);
selected = zeros(size(expected.selectedPairIDs));
for realStep = 1:size(transitions, 1)
    row = transitions(realStep, :); s = row(1); a = row(2);
    reward = row(3); nextState = row(4); terminated = logical(row(5));
    y = OceanStats.RL.Dyna.Functional.qLearningTarget(reward, q(nextState, :), ...
        terminated, p.gamma, mask(nextState, :));
    q(s, a) = q(s, a) + p.alpha * (y - q(s, a));
    updateIndex = updateIndex + 1; actualSnapshots{updateIndex} = q;
    transition = OceanStats.RL.Dyna.Core.Transition('StateIndex', s, ...
        'ActionIndex', a, 'Reward', reward, 'NextStateIndex', nextState, ...
        'Terminated', terminated, 'Origin', "real", 'RealStep', realStep);
    model.update(transition);
    id = OceanStats.RL.Dyna.Functional.pairID(s, a, p.numStates, p.numActions);
    if ~any(eligible == id), eligible(end + 1, 1) = id; end %#ok<AGROW>
    probabilities = ones(size(eligible)) ./ numel(eligible);
    position = OceanStats.RL.Dyna.Functional.categoricalSample( ...
        probabilities, expected.uniformDraws(realStep));
    selected(realStep) = eligible(position);
    [ps, pa] = OceanStats.RL.Dyna.Functional.pairFromID( ...
        selected(realStep), p.numStates, p.numActions);
    outcome = model.sample(ps, pa, RandStream('mrg32k3a', 'Seed', realStep));
    y = OceanStats.RL.Dyna.Functional.qLearningTarget(outcome.Reward, ...
        nextValues_(q, outcome.NextStateIndex), logical(outcome.Terminated), ...
        p.gamma, nextMask_(mask, outcome.NextStateIndex, outcome.Terminated));
    q(ps, pa) = q(ps, pa) + p.alpha * (y - q(ps, pa));
    updateIndex = updateIndex + 1; actualSnapshots{updateIndex} = q;
end
expectedSnapshots = {expected.updates.q};
assert(numel(expectedSnapshots) == numel(actualSnapshots));
for k = 1:numel(actualSnapshots)
    assert(max(abs(actualSnapshots{k} - expectedSnapshots{k}), [], 'all') < 1e-12, ...
        'OceanStats:RL:Dyna:Validation:QMismatch', 'Q mismatch at update %d.', k);
end
assert(isequal(selected(:), expected.selectedPairIDs(:)), ...
    'OceanStats:RL:Dyna:Validation:SelectionMismatch');
assert(max(abs(q - expected.finalQ), [], 'all') < 1e-12, ...
    'OceanStats:RL:Dyna:Validation:FinalQMismatch');
for k = 1:numel(expected.model)
    item = expected.model(k); snapshot = model.inspect(item.state, item.action);
    outcome = snapshot.Outcomes;
    assert(snapshot.ObservationCount == item.count && ...
        outcome.NextStateIndex == item.nextState && ...
        abs(outcome.Reward - item.reward) < 1e-12 && ...
        outcome.Terminated == item.terminated, ...
        'OceanStats:RL:Dyna:Validation:ModelMismatch');
end
bonuses = OceanStats.RL.Dyna.Functional.recencyBonus(p.kappa, [0 1 4 9 16]);
assert(max(abs(bonuses(:) - expected.recencyBonuses(:))) < 1e-12, ...
    'OceanStats:RL:Dyna:Validation:BonusMismatch');
queue = OceanStats.RL.Dyna.Planning.MaxPriorityQueue();
pushes = [4 0.5; 2 0.5; 1 0.9; 1 0.4];
for k = 1:size(pushes, 1), queue.push(pushes(k, 1), pushes(k, 2)); end
pops = zeros(numel(expected.queuePops), 2);
for k = 1:size(pops, 1), [pops(k, 1), pops(k, 2)] = queue.pop(); end
expectedPops = [[expected.queuePops.pairID]' [expected.queuePops.priority]'];
assert(max(abs(pops - expectedPops), [], 'all') < 1e-12, ...
    'OceanStats:RL:Dyna:Validation:QueueMismatch');
report = struct('Passed', true, 'NumUpdates', numel(actualSnapshots), ...
    'SelectedPairIDs', selected, 'FinalQ', q, 'ModelPairs', numel(expected.model), ...
    'QueuePops', pops, 'Reference', string(expected.reference));
end

function values = nextValues_(q, state)
if state == 0, values = zeros(1, size(q, 2)); else, values = q(state, :); end
end

function valid = nextMask_(mask, state, terminated)
if terminated, valid = false(1, size(mask, 2)); else, valid = mask(state, :); end
end
