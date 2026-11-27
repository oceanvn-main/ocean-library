classdef NStepBackup < OceanStats.RL.TD.Backups.IBackup
    % NSTEPBACKUP  Buffered n-step return: n-step TD (state values) and n-step Sarsa / Expected Sarsa (action values; the tail
    % of the return is the target's bootstrap). An off-policy target is rejected: it would need importance sampling or tree
    % backup, which are not implemented. Bootstrapping stops only on true termination; a truncated episode keeps the
    % bootstrap from its last state.
    properties (Constant)
        Latex = "G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^{k} R_{t+k+1} + \gamma^{n}\, \hat v(S_{t+n})"
        Reference = "Sutton & Barto (2018) eq. 7.1-7.2 (n-step TD, n-step Sarsa)"
    end
    methods
        function s = name(~), s = "nstep"; end
        function tf = allowsOffPolicyTarget(~, kind), tf = kind ~= "action"; end

        function [ret, L] = runEpisode(obj, ctx)
            n = ctx.n;  g = ctx.gamma;
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
            ctrl = vf.kind() == "action";
            s0 = sm.start();
            Sbuf = s0;  Abuf = zeros(0, 1);  Rbuf = zeros(0, 1);
            Abuf(1) = ctx.act(s0);
            T = Inf;  t = 0;  endTerm = false;  ret = 0;  L = 0;
            while true
                if t < T
                    [s2i, r, term, ~, done] = sm.step(Abuf(t + 1));
                    L = L + 1;  ret = ret + r;
                    if sm.capped(done, L), done = true; end
                    Rbuf(t + 1, 1) = r;  Sbuf(t + 2, 1) = s2i;
                    if done
                        T = t + 1;  endTerm = term;
                        if ~term                                    % truncated: pick A_T to bootstrap from
                            if ctrl, Abuf(t + 2, 1) = ctx.act(s2i); else, Abuf(t + 2, 1) = 1; end
                        end
                    else
                        Abuf(t + 2, 1) = ctx.act(s2i);
                    end
                end
                tau = t - n + 1;
                if tau >= 0
                    last = min(tau + n, T);
                    rew = Rbuf(tau + 1:last);
                    if tau + n < T
                        boot = obj.estimate_(ctx, Sbuf(tau + n + 1), Abuf(tau + n + 1), ctrl);
                    elseif ~endTerm
                        boot = obj.estimate_(ctx, Sbuf(T + 1), Abuf(T + 1), ctrl);
                    else
                        boot = 0;
                    end
                    G = OceanStats.RL.TD.Functional.nStepReturn(rew, boot, g);
                    cnt = vf.touch(Sbuf(tau + 1), Abuf(tau + 1));
                    vf.update(Sbuf(tau + 1), Abuf(tau + 1), ctx.stepSize.alpha(cnt), G);
                end
                if tau == T - 1, break; end
                t = t + 1;
            end
            ob.finishArrays(Sbuf, Abuf, Rbuf, endTerm, ~endTerm, sm.OutsideLast);
        end
    end

    methods (Access = private)
        function e = estimate_(~, ctx, si, ai, ctrl)
            if ctrl
                e = ctx.target.bootstrap(ctx.values.estimates(si), ai, ctx.behaviour(si), ctx.sampler.ValidMask(si, :));
            else
                e = ctx.values.estimates(si);
            end
        end
    end
end
