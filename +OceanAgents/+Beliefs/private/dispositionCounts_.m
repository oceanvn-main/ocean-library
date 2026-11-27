function n = dispositionCounts_(n, dispositions)
% DISPOSITIONCOUNTS_  Add a batch of dispositions to the counts struct (fields Applied, Duplicate, Unknown, Report,
% Ineligible, Frozen, RecordedOnly). With no input arguments, return the all-zero counts struct.
if nargin == 0
    n = struct('Applied', 0, 'Duplicate', 0, 'Unknown', 0, 'Report', 0, 'Ineligible', 0, 'Frozen', 0, 'RecordedOnly', 0);
    return
end
names = ["applied" "duplicate" "unknown" "report" "ineligible" "frozen" "recorded-only"];
fields = ["Applied" "Duplicate" "Unknown" "Report" "Ineligible" "Frozen" "RecordedOnly"];
for k = 1:numel(names)
    n.(fields(k)) = n.(fields(k)) + nnz(dispositions == names(k));
end
end
