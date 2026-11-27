classdef (Abstract) ILearner < handle
    % ILEARNER  Fits an agent's own trainable state from its own Transition records at declared boundaries.
    % A learner never steps or resets an environment and never sees another agent's data.
    %
    %   learner.consume(t)          accept one private Transition (stores it; applies nothing yet)
    %   n = learner.update(b)       apply updates due at boundary b ("step" | "episode"); n = #parameter updates
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 3.

    methods (Abstract)
        d = describe(obj)                 % struct: Name, Version, Gamma, Boundary
        consume(obj, transition)
        n = update(obj, boundary)
    end

    methods
        function checkCompatible(~, ~)
            % CHECKCOMPATIBLE  Throw OceanAgents:Learners:Incompatible if the policy description does not fit.
        end
        function s = exportState(~), s = struct(); end
        function importState(~, ~), end
    end
end
