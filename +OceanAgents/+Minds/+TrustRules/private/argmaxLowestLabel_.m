function k = argmaxLowestLabel_(scores, labels)
% ARGMAXLOWESTLABEL_  Index of the maximum score; exact ties go to the candidate with the LOWEST label
% (deterministic, no random tie-break; A2 contract section 4.1, H9).
best = max(scores);
tied = find(scores == best);
[~, j] = min(labels(tied));
k = tied(j);
end
