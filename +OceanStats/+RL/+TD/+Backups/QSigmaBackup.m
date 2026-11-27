classdef QSigmaBackup < OceanStats.RL.TD.Backups.IBackup
    % QSIGMABACKUP  Unifying n-step Q(sigma) algorithm blending sampling (sigma=1) and expectation (sigma=0).
    %   Sutton & Barto (2018) Sec 7.6, Eq 7.17.
    properties (Constant)
        Latex = "G_{t:t+n} = R_{t+1} + \gamma [\sigma_{t+1} A_{t+1} + (1-\sigma_{t+1}) \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1},a)]"
        Reference = "Sutton & Barto (2018) Sec 7.6, Q(sigma)"
    end
    properties
        Sigma (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Sigma, 1)} = 0.5
    end
    methods
        function obj = QSigmaBackup(sigma)
            if nargin >= 1, obj.Sigma = sigma; end
        end
        function s = name(~), s = "qsigma"; end
        function tf = allowsOffPolicyTarget(~, ~), tf = true; end

        function [ret, L] = runEpisode(obj, ctx)
            n = ctx.n;  g = ctx.gamma;  sig = obj.Sigma;
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
                    last = min(tau + n, T);
                    if tau + n >= T && endTerm
                        G = Rbuf(T);
                        for k = T-1:-1:tau+1
                            s_k = Sbuf(k + 1);
                            a_k = Abuf(k + 1);
                            probs_k = ctx.targetPolicy(s_k);
                            valid_k = sm.ValidMask(s_k, :);
                            q_k = vf.estimates(s_k);
                            expVal = sum(probs_k(valid_k) .* q_k(valid_k));
                            sampVal = q_k(a_k);
                            G = Rbuf(k) + g * (sig * sampVal + (1 - sig) * expVal) + g * (sig + (1 - sig) * probs_k(a_k)) * G;
                        end
                    else
                        s_last = Sbuf(last + 1);
                        probs_last = ctx.targetPolicy(s_last);
                        valid_last = sm.ValidMask(s_last, :);
                        q_last = vf.estimates(s_last);
                        G = sum(probs_last(valid_last) .* q_last(valid_last));
                        for k = last-1:-1:tau+1
                            s_k = Sbuf(k + 1);
                            a_k = Abuf(k + 1);
                            probs_k = ctx.targetPolicy(s_k);
                            valid_k = sm.ValidMask(s_k, :);
                            q_k = vf.estimates(s_k);
                            expVal = sum(probs_k(valid_k) .* q_k(valid_k));
                            G = Rbuf(k) + g * (sig * q_k(a_k) + (1 - sig) * expVal);
                        end
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
