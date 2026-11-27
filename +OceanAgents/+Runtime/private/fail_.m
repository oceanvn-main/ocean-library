function fail_(reason, fmt, varargin)
% FAIL_  Throw an OceanAgents:Runtime:<reason> error (private helper for the runtime classes).
error(sprintf('OceanAgents:Runtime:%s', reason), fmt, varargin{:});
end
