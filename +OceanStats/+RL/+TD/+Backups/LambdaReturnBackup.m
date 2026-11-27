classdef LambdaReturnBackup < OceanStats.RL.TD.Backups.IBackup
    % LAMBDARETURNBACKUP  Forward-view OFFLINE lambda-return algorithm for state values: the episode is rolled out with the
    % values frozen, every state's lambda-return is computed, and the increments are applied together at the episode's end.
    % With accumulating traces and a constant step size it equals the offline backward view (TraceBackup, Offline) exactly.
    properties (Constant)
        Latex = "G_t^{\lambda} = (1-\lambda)\sum_{n\ge1} \lambda^{n-1} G_{t:t+n}"
        Reference = "Sutton & Barto (2018) eq. 12.2, 12.17 (the offline lambda-return algorithm)"
    end
    methods
        function s = name(~), s = "lambdareturn"; end
        function tf = supportsValueKind(~, kind), tf = kind == "state"; end

        function [ret, L] = runEpisode(~, ctx)
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
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
            nextV = vf.stateValues(states(2:end));                  % V frozen during the episode (offline)
            G = OceanStats.RL.TD.Functional.lambdaReturns(rewards, nextV, term, ctx.gamma, ctx.lambda);
            dV = vf.zerosLike();
            for t = 1:numel(rewards)
                s = states(t);
                cnt = vf.touch(s, 1);
                dV(s) = dV(s) + ctx.stepSize.alpha(cnt) * (G(t) - vf.get(s, 1));
            end
            vf.applyIncrement(dV);
            ob.finishArrays(states, zeros(numel(states), 1), rewards, term, trunc, sm.OutsideLast);
        end
    end
end
