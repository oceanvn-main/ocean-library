classdef (Abstract) IValueFunction < OceanStats.RL.TD.ITDComponent
    % IVALUEFUNCTION  What is being learned: the table (or, later, the function approximator) of state values V(s) or
    % action values Q(s,a), together with the visit counts that drive the step size. A backup (the update scheme) talks only
    % to this interface, so tabular tables, a double estimator or a future approximator are interchangeable.
    %
    %   k = vf.kind()                              "state" (V, prediction) or "action" (Q, control)
    %   x = vf.estimates(si)                       own estimates at state si: a 1 x A row (Q) or a scalar (V)
    %   x = vf.behaviourEstimates(si)              what the behaviour policy acts on (a double estimator averages its tables)
    %   x = vf.get(si, ai)                         the entry V(si) / Q(si,ai)
    %   n = vf.touch(si, ai)                       count one more visit of the entry; returns the new count
    %   vf.update(si, ai, alpha, y)                x <- x + alpha (y - x)
    %   T = vf.zerosLike();  vf.applyIncrement(dT) whole-table accumulators (eligibility traces, offline updates)
    %   coin = vf.oneStep(target, stepSize, si, ai, r, s2i, a2, term, gamma, p2, valid2, stream)
    %                                              one complete one-step update from a transition
    %   tf = vf.updatesBeforeNextAction()          true if the update must precede the choice of the next action
    %   v = vf.stateValues(idx)                    V at several states (state-value functions only)
    %   Q = vf.finalQ()  /  V = vf.finalV()        the learned table at the end of a run (action / state values)
    %   tf = vf.requiresOneStepBackup()            true if only a backup that supportsUpdateFirstValues can drive it
    %   tf = vf.requiresGreedyTarget()             true if it ignores the target's own bootstrap and needs a target that
    %                                              declares supportsDouble() (Double Q-learning)
    %
    % SCOPE: the interface is TABULAR for now: the trace and offline backups accumulate whole tables (zerosLike,
    % applyIncrement, touch counts). A function approximator would implement the same methods on its parameter vector
    % (zerosLike = zero trace, applyIncrement = weight update); only the one-step, n-step and lambda-return backups
    % are meaningful for one that has no per-entry counts.
    %
    % An estimator that must draw its own random numbers (Double Q-learning's coin) does so from the given stream, in
    % oneStep, so the order of random draws stays under the backup's control.
    methods (Abstract)
        k = kind(obj)
        x = estimates(obj, si)
        x = behaviourEstimates(obj, si)
        x = get(obj, si, ai)
        n = touch(obj, si, ai)
        update(obj, si, ai, alpha, y)
        T = zerosLike(obj)
        applyIncrement(obj, dT)
        coin = oneStep(obj, target, stepSize, si, ai, r, s2i, a2, term, gamma, p2, valid2, stream)
    end
    methods
        function tf = updatesBeforeNextAction(~), tf = false; end
        function tf = requiresOneStepBackup(~), tf = false; end
        function tf = requiresGreedyTarget(~), tf = false; end
        function Q = finalQ(obj) %#ok<STOUT>
            error('OceanStats:RL:TD:IValueFunction:NotActionValues', '%s does not hold action values.', class(obj));
        end
        function V = finalV(obj) %#ok<STOUT>
            error('OceanStats:RL:TD:IValueFunction:NotStateValues', '%s does not hold state values.', class(obj));
        end
        function v = stateValues(obj, ~) %#ok<STOUT>
            error('OceanStats:RL:TD:IValueFunction:NotStateValues', '%s does not hold state values.', class(obj));
        end
    end
    methods (Access = protected)
        function a = assumptions(~)
            a = {'the estimates being learned: a table of state values V(s) or action values Q(s,a) (tabular interface)'};
        end
    end
end
