classdef (Abstract) IDecisionProcess < handle
    % IDECISIONPROCESS  An agent's mind: turns the evidence it is given into one primitive decision at a time.
    % The substitution point for different kinds of agent (reactive policy, belief-based, deliberative); swapping
    % it never changes the environment. It returns only the requested one-step decision and never steps a world.
    %
    %   mind.beginEpisode(view);  d = mind.decide(view, stream);  mind.observe(feedback);  mind.endEpisode();
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 3.

    methods (Abstract)
        d = describe(obj)                 % struct: Profile, Components
        dec = decide(obj, view, stream)
    end

    methods
        function checkCompatible(~, ~, ~, ~), end
        function beginEpisode(~, ~), end
        function observe(~, ~), end
        function endEpisode(~), end
        function s = exportState(~), s = struct(); end
        function importState(~, ~), end
    end
end
