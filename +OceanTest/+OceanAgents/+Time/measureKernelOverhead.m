function r = measureKernelOverhead(nSteps, nRepeat)
% MEASUREKERNELOVERHEAD  IK12: per-step cost of an AgentWorld driven directly versus through a KernelDomain recording
% committed history, on the same workload and machine (median of nRepeat runs). Reported, never asserted.
%
%   r = OceanTest.OceanAgents.Time.measureKernelOverhead(400, 5);
%   r.DirectMs, r.AdapterNoComponentsMs, r.AdapterRecordingMs   % milliseconds per step
%   r.RetainedBytes                                             % bytes held by the recorded states of one run
arguments
    nSteps (1,1) double = 400
    nRepeat (1,1) double = 5
end
A = OceanTest.OceanAgents.Fixtures.World.Agent.LineDoors;
mk = @() OceanAgents.World.AgentWorld(A.doors(["a1" "a2"], 'Goal', 1000, 'Limit', Inf), A.cfg(["a1" "a2"]), 'Name', "line");
modes = ["direct" "none" "recording"];
% warm-up: one untimed run of each mode so that first-call costs do not land on the first mode measured
for m = 1:numel(modes), runOnce_(mk, modes(m), 60, false); end
t = zeros(numel(modes), nRepeat);
bytes = 0;
for m = 1:numel(modes)
    for k = 1:nRepeat
        aw = mk();
        dom = [];
        if modes(m) == "none", dom = OceanAgents.Time.KernelDomain(aw, 'DomainID', "line"); end
        if modes(m) == "recording", dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos", 'DomainID', "line"); end
        f = aw.reset(7);
        rs = RandStream('mrg32k3a', 'Seed', 3);
        t0 = tic;
        for s = 1:nSteps
            if f.EpisodeEnded, f = aw.reset(7); end
            ds = OceanAgents.Core.Decision.empty(1, 0);
            for q = find(f.Requested)
                ds(end + 1) = OceanAgents.Core.Decision('AgentID', f.Roster(q), 'EpisodeID', f.EpisodeID, ...
                    'FrameID', f.FrameID, 'Action', randi(rs, 2)); %#ok<AGROW>
            end
            b = OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, 'Decisions', ds);
            f = aw.step(b);
        end
        t(m, k) = toc(t0) / nSteps * 1e3;
        if modes(m) == "recording" && k == 1
            st = dom.Log.at(0).State;
            w = whos('st');
            bytes = w.bytes * dom.Log.count();      % states have the same size; contexts are extra
        end
    end
end
r = struct('Steps', nSteps, 'Repeats', nRepeat, 'DirectMs', median(t(1, :)), ...
    'AdapterNoComponentsMs', median(t(2, :)), 'AdapterRecordingMs', median(t(3, :)), ...
    'RetainedBytes', bytes, 'Note', "RetainedBytes = commits x bytes of one stored state (contexts excluded)");
end

function runOnce_(mk, mode, nSteps, ~)
aw = mk();
if mode == "none", OceanAgents.Time.KernelDomain(aw, 'DomainID', "line"); end
if mode == "recording", OceanAgents.Time.KernelDomain(aw, 'Components', "Pos", 'DomainID', "line"); end
f = aw.reset(7);
for s = 1:nSteps
    ds = OceanAgents.Core.Decision.empty(1, 0);
    for q = find(f.Requested)
        ds(end + 1) = OceanAgents.Core.Decision('AgentID', f.Roster(q), 'EpisodeID', f.EpisodeID, ...
            'FrameID', f.FrameID, 'Action', 2); %#ok<AGROW>
    end
    f = aw.step(OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, 'Decisions', ds));
end
end
