function fail_(reason, fmt, varargin)
% FAIL_  Throw an OceanAgents:Core:<reason> error (private helper for the Core records).
error(sprintf('OceanAgents:Core:%s', reason), fmt, varargin{:});
end
