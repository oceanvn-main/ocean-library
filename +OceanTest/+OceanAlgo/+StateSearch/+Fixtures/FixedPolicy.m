classdef FixedPolicy < OceanAlgo.StateSearch.IRolloutPolicy
    % FIXEDPOLICY  Deterministic rollout policy pi(s) from a vector (test fixture).
    properties (SetAccess = immutable)
        Pi double
    end
    methods
        function obj = FixedPolicy(pi), obj.Pi = pi; end
        function a = act(obj, ~, s, ~), a = obj.Pi(s); end
    end
end
