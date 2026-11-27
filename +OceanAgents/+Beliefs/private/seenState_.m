function varargout = seenState_(varargin)
% SEENSTATE_  Convert the seen-evidence dictionary to and from exportable value data (two string columns).
%   [ids, keys] = seenState_(seen)            export
%   seen        = seenState_(ids, keys, max)  import (validated; errors OceanAgents:Beliefs:BadImport)
%   seen        = seenState_()                empty dictionary
if nargin == 0
    varargout{1} = dictionary(string.empty(0, 1), string.empty(0, 1));
elseif nargin == 1
    seen = varargin{1};
    if numEntries(seen) == 0
        varargout = {strings(0, 1), strings(0, 1)};
    else
        varargout = {reshape(keys(seen), [], 1), reshape(values(seen), [], 1)};
    end
else
    [ids, ks, maxEvidence] = varargin{:};
    if ~(isstring(ids) && isstring(ks) && numel(ids) == numel(ks)) || any(ismissing(ids)) || any(ismissing(ks)) || ...
            any(ids == "") || numel(unique(ids)) ~= numel(ids) || numel(ids) > maxEvidence
        error('OceanAgents:Beliefs:BadImport', ...
            'Seen evidence must be two equal-length string lists of unique non-empty IDs and keys, at most MaxEvidence.');
    end
    seen = dictionary(string.empty(0, 1), string.empty(0, 1));
    if ~isempty(ids)
        seen(reshape(ids, [], 1)) = reshape(ks, [], 1);
    end
    varargout{1} = seen;
end
end
