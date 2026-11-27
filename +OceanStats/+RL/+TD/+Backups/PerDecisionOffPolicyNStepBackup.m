classdef PerDecisionOffPolicyNStepBackup < OceanStats.RL.TD.Backups.IBackup
    % PERDECISIONOFFPOLICYNSTEPBACKUP  Off-policy n-step Sarsa with per-decision importance sampling ratios.
    %   Sutton & Barto (2018) Sec 7.3 & 7.4, Eq. 7.13 - 7.14.
    properties (Constant)
        Latex = "G_{t:t+n} = \rho_t (R_{t+1} + \gamma G_{t+1:t+n}) + (1-\rho_t) V(S_t)"
        Reference = "Sutton & Barto (2018) Eq. 7.13 - 7.14"
    end
    methods
        function s = name(~), s = "perdecisionoffpolicy"; end
        function tf = allowsOffPolicyTarget(~, ~), tf = true; end

        function [ret, L] = runEpisode(~, ctx)
            n = ctx.n;  g = ctx.gamma;
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
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
                    else
                        Abuf(t + 2, 1) = ctx.act(s2i);
                    end
                end
                tau = t - n + 1;
                if tau >= 0
                    % Compute per-decision importance sampling return G_{tau:tau+n}
                    last = min(tau + n, T);
                    if tau + n < T
                        G = vf.estimates(Sbuf(tau + n + 1));
                        G = G(Abuf(tau + n + 1));
                    elseif ~endTerm
                        G = vf.estimates(Sbuf(T + 1));
                        G = G(Abuf(T + 1));
                    else
                        G = 0;
                    end
                    for k = last-1:-1:tau
                        s_k = Sbuf(k + 1);
                        a_k = Abuf(k + 1);
                        pi_prob = ctx.targetPolicy(s_k);
                        b_prob = ctx.behaviour(s_k);
                        rho = pi_prob(a_k) / max(b_prob(a_k), 1e-12);
                        v_k = sum(pi_prob .* vf.estimates(s_k));
                        G = rho * (Rbuf(k + 1) + g * G) + (1 - rho) * v_k;
                    end
                    cnt = vf.touch(Sbuf(tau + 1), Abuf(tau + 1));
                    vf.update(Sbuf(tau + 1), Abuf(tau + 1), ctx.stepSize.alpha(cnt), G);
                end
                if tau == T - 1, break; end
                t = t + 1;
            end
            ob.finishArrays(Sbuf, Abuf, Rbuf, endTerm, ~endTerm, sm.OutsideLast);
        end
    end
end
