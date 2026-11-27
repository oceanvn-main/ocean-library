function ref = referenceBetaCounts(stream, prior, quantity)
% REFERENCEBETACOUNTS  Independent plain-counting reference for a Beta-Bernoulli trust store (tests only).
% Re-implements the A2 contract rules (section 3.2 disposition table, section 3.4 counting) from scratch with string
% lists and loops; it never calls a belief store, its private helpers, or EvidenceRecord.canonicalKey.
%
%   ref = OceanTest.OceanAgents.Fixtures.Beliefs.referenceBetaCounts(stream, [a0 b0], "completion")
%
%   stream   struct array, one element per observe() call, fields:
%              Evidence  1 x K OceanAgents.Core.EvidenceRecord (read through public properties only)
%              Enabled   logical: were updates enabled for that call
%   ref      struct with fields
%              Keys (n x 1 string "subject|context"), Successes, Failures (n x 1 double)
%              Alpha = a0 + Successes, Beta = b0 + Failures, Mass = Successes + Failures
%              Dispositions (1 x total string, in delivery order), Applied (count)
seenIDs = strings(0, 1);
seenContent = strings(0, 1);
keys = strings(0, 1);
succ = zeros(0, 1);
fail = zeros(0, 1);
disps = strings(1, 0);
for c = 1:numel(stream)
    for k = 1:numel(stream(c).Evidence)
        e = stream(c).Evidence(k);
        content = join([e.EvidenceID, e.SubjectID, e.Context, e.Quantity, e.Outcome, e.SourceKind, e.SourceID, ...
            string(e.EpisodeID), string(e.Tick)], "|");
        j = find(seenIDs == e.EvidenceID, 1);
        if ~isempty(j)
            if seenContent(j) ~= content
                error('referenceBetaCounts:conflict', 'The reference stream must not contain conflicts.');
            end
            disps(end + 1) = "duplicate"; %#ok<AGROW>
            continue
        end
        seenIDs(end + 1, 1) = e.EvidenceID; %#ok<AGROW>
        seenContent(end + 1, 1) = content; %#ok<AGROW>
        if e.Quantity ~= quantity
            d = "ineligible";
        elseif e.SourceKind == "report"
            d = "report";
        elseif e.Outcome == "unknown"
            d = "unknown";
        elseif ~stream(c).Enabled
            d = "frozen";
        else
            d = "applied";
            key = e.SubjectID + "|" + e.Context;
            r = find(keys == key, 1);
            if isempty(r)
                keys(end + 1, 1) = key; %#ok<AGROW>
                succ(end + 1, 1) = 0; %#ok<AGROW>
                fail(end + 1, 1) = 0; %#ok<AGROW>
                r = numel(keys);
            end
            if e.Outcome == "success"
                succ(r) = succ(r) + 1;
            else
                fail(r) = fail(r) + 1;
            end
        end
        disps(end + 1) = d; %#ok<AGROW>
    end
end
ref = struct('Keys', keys, 'Successes', succ, 'Failures', fail, 'Alpha', prior(1) + succ, ...
    'Beta', prior(2) + fail, 'Mass', succ + fail, 'Dispositions', disps, 'Applied', nnz(disps == "applied"));
end
