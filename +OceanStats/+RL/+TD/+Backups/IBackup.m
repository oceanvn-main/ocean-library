classdef (Abstract) IBackup < OceanStats.RL.TD.ITDComponent
    % IBACKUP  The update scheme ("Mode"): how one episode is rolled out and how the estimates are moved toward their
    % targets. One-step, n-step, eligibility-trace and lambda-return schemes are four files; a new scheme (tree backup,
    % true online TD(lambda), ...) is ONE new file in this package - nothing else is edited.
    %
    %   [ret, L] = backup.runEpisode(ctx)       run one episode; returns its undiscounted return and length
    %
    % A backup is a stateless strategy: everything it needs is in ctx (OceanStats.RL.TD.Context). It declares what it
    % supports, so the learner validates a combination from capabilities, not from class names:
    %   supportsValueKind(kind)        can it learn "state" (prediction) / "action" (control) values?
    %   allowsOffPolicyTarget(kind)    may an off-policy target (Q-learning) be combined with it?
    %   supportsOffline(kind)          is the 'Offline' option meaningful for it?
    %   supportsUpdateFirstValues()    may it drive a value function that flips its coin and updates BEFORE the next action
    %                                  is chosen (a double estimator)?
    methods (Abstract)
        [ret, L] = runEpisode(obj, ctx)
    end
    methods (Access = protected)
        function a = assumptions(~)
            a = {'stateless strategy: rolls one episode out through the Context and updates the value function'};
        end
    end
    methods
        function spec = getPipelineSpec(obj)
            spec = getPipelineSpec@OceanStats.RL.TD.ITDComponent(obj);
            spec.Inputs = struct('Name', "context", 'Domain', "RL-Context", 'Dtype', "object", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "return,length", 'Domain', "RL-Episode", 'Dtype', "double", 'Shape', "1 x 2");
        end
        function tf = supportsValueKind(~, ~), tf = true; end
        function tf = allowsOffPolicyTarget(~, ~), tf = true; end
        function tf = supportsOffline(~, ~), tf = false; end
        function tf = supportsUpdateFirstValues(~), tf = false; end
    end
end
