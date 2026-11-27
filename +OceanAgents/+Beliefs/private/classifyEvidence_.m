function [dispositions, seen] = classifyEvidence_(evidence, seen, quantity, updatesEnabled, eligibleDisposition, maxEvidence)
% CLASSIFYEVIDENCE_  Disposition of each evidence record, in order (A2_Beliefs_Contract.md section 3.2), shared by
% every belief store so the rule cannot diverge between them. PURE: seen is a dictionary (a value type) mapping
% EvidenceID -> canonicalKey; the updated copy is returned and the caller commits it only if nothing threw.
%
%   seen ID, same key        -> "duplicate"   (nothing else happens)
%   seen ID, different key   -> error OceanAgents:Beliefs:EvidenceConflict
%   otherwise the ID is marked seen (whatever follows), then the first matching rule wins:
%   Quantity ~= quantity     -> "ineligible"
%   SourceKind == "report"   -> "report"
%   Outcome == "unknown"     -> "unknown"
%   updates disabled         -> "frozen"
%   else                     -> eligibleDisposition ("applied" for an updating store, "recorded-only" for a fixed one)
%
% More than maxEvidence seen IDs raises OceanAgents:Beliefs:CapacityExceeded (nothing is evicted).
dispositions = strings(1, numel(evidence));
for k = 1:numel(evidence)
    e = evidence(k);
    id = e.EvidenceID;
    key = e.canonicalKey();
    if isKey(seen, id)
        if seen(id) == key
            dispositions(k) = "duplicate";
            continue
        end
        error('OceanAgents:Beliefs:EvidenceConflict', ...
            'EvidenceID "%s" was already seen with different content (seen: %s; now: %s).', id, ...
            strrep(seen(id), char(31), '|'), strrep(key, char(31), '|'));
    end
    seen(id) = key;
    if numEntries(seen) > maxEvidence
        error('OceanAgents:Beliefs:CapacityExceeded', ...
            'Seen-evidence capacity MaxEvidence = %d exceeded by "%s"; nothing is evicted.', maxEvidence, id);
    end
    if e.Quantity ~= quantity
        dispositions(k) = "ineligible";
    elseif e.SourceKind == "report"
        dispositions(k) = "report";
    elseif e.Outcome == "unknown"
        dispositions(k) = "unknown";
    elseif ~updatesEnabled
        dispositions(k) = "frozen";
    else
        dispositions(k) = eligibleDisposition;
    end
end
end
