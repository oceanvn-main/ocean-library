function evidence = evidenceList_(evidence)
% EVIDENCELIST_  Accept [] or a vector of OceanAgents.Core.EvidenceRecord; return it as a 1 x K row.
if isempty(evidence)
    evidence = OceanAgents.Core.EvidenceRecord.empty(1, 0);
    return
end
if ~isa(evidence, 'OceanAgents.Core.EvidenceRecord') || ~isvector(evidence)
    error('OceanAgents:Core:InvalidRecord', 'observe expects a vector of OceanAgents.Core.EvidenceRecord, got %s.', ...
        class(evidence));
end
evidence = reshape(evidence, 1, []);
end
