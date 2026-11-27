function requireFields_(options, names, recordName)
% REQUIREFIELDS_  Name-value options without a default are absent from the struct when not passed.
missing = names(~isfield(options, names));
if ~isempty(missing)
    fail_('InvalidRecord', '%s: missing required field(s): %s.', recordName, strjoin(missing, ', '));
end
end
