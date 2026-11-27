classdef NotAToy < handle
    % NOTATOY  Has every ToyModule method but does not implement the ToyModule interface: a door that admits
    % ToyModule must refuse it with OceanAgents:Guard:NotAdmissible (admission is by interface + axioms).
    methods
        function c = capacity(~), c = 4; end
        function reset(~, ~), end
        function x = sample(~), x = [0.1 0.2 0.3]; end
        function y = apply(~, x), y = x; end
    end
end
