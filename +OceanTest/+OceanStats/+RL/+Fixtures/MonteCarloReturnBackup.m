classdef MonteCarloReturnBackup < OceanStats.RL.TD.Backups.IBackup
    % MONTECARLORETURNBACKUP  TEST FIXTURE for the extension rule: a brand-new update scheme written as ONE file, in a
    % foreign package, registered at run time. Constant-alpha every-visit Monte Carlo prediction: roll the episode out, then
    % move every visited state toward its full return (no bootstrap). It edits nothing in TDLearner or the other backups.
    properties (Constant)
        Latex = "V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]"
        Reference = "test fixture (constant-alpha every-visit Monte Carlo prediction)"
    end
    methods
        function s = name(~), s = "mcreturn"; end
        function tf = supportsValueKind(~, kind), tf = kind == "state"; end
        function [ret, L] = runEpisode(~, ctx)
            sm = ctx.sampler;  vf = ctx.values;
            si = sm.start();
            states = si;  rewards = zeros(0, 1);  ret = 0;  L = 0;
            while true
                ai = ctx.act(si);
                [s2i, r, term, trunc, done] = sm.step(ai);
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; trunc = true; end
                rewards(end+1, 1) = r; %#ok<AGROW>
                states(end+1, 1) = s2i; %#ok<AGROW>
                if done, break; end
                si = s2i;
            end
            for t = 1:numel(rewards)
                G = OceanStats.RL.TD.Functional.nStepReturn(rewards(t:end), 0, ctx.gamma);
                cnt = vf.touch(states(t), 1);
                vf.update(states(t), 1, ctx.stepSize.alpha(cnt), G);
            end
            ctx.observer.finish(term, trunc);
        end
    end
end
