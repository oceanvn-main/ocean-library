classdef TraceBackup < OceanStats.RL.TD.Backups.IBackup
    % TRACEBACKUP  Backward-view eligibility traces: TD(lambda) for state values; Sarsa(lambda) and Expected Sarsa(lambda)
    % for action values; Watkins's Q(lambda) when the target is off-policy (the traces are cut after an action that was
    % not greedy, judged on the values that chose it, i.e. before the update). The trace strategy decides how a visit is
    % recorded (accumulating, replacing, ...). 'Offline' accumulates the episode's increments and applies them at its end.
    properties (Constant)
        Latex = "e_t = \gamma\lambda\, e_{t-1} + \nabla \hat v(S_t), \quad \Delta w = \alpha\, \delta_t\, e_t"
        Reference = "Sutton & Barto (2018) ch. 12 (TD(lambda), Sarsa(lambda)); Watkins (1989) Q(lambda)"
    end
    methods
        function s = name(~), s = "trace"; end
        function tf = supportsOffline(~, kind), tf = kind == "state"; end

        function [ret, L] = runEpisode(~, ctx)
            g = ctx.gamma;  lam = ctx.lambda;
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;  tg = ctx.target;
            ctrl = vf.kind() == "action";
            E = vf.zerosLike();
            si = sm.start();
            ai = ctx.act(si);
            ret = 0;  L = 0;
            ob.begin(si, ai);
            dV = vf.zerosLike();                                       % offline accumulator (state values)
            while true
                [s2i, r, term, trunc, done] = sm.step(ai);
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; trunc = true; end
                if ctrl
                    valid2 = sm.ValidMask(s2i, :);
                    if ~term, p2 = ctx.behaviour(s2i); a2 = ctx.draw(p2); else, p2 = zeros(1, sm.NumA); a2 = 1; end
                    y = tg.compute(r, vf.estimates(s2i), a2, term, g, p2, valid2);
                    delta = y - vf.get(si, ai);
                    cut = false;                                       % Watkins: judged on the values that chose A' (before this update)
                    if tg.IsOffPolicy && ~term
                        q = vf.estimates(s2i);  q(~valid2) = -Inf;
                        cut = vf.get(s2i, a2) < max(q);
                    end
                    cnt = vf.touch(si, ai);
                    alpha = ctx.stepSize.alpha(cnt);
                    E = ctx.trace.visitWith(E, sub2ind(size(E), si, ai), alpha, g * lam);
                    vf.applyIncrement(alpha * delta * E);
                    E = g * lam * E;
                    if cut, E(:) = 0; end
                else
                    a2 = 1;
                    y = tg.compute(r, vf.estimates(s2i), 1, term, g, [], true);
                    delta = y - vf.get(si, 1);
                    cnt = vf.touch(si, 1);
                    alpha = ctx.stepSize.alpha(cnt);
                    E = ctx.trace.visitWith(E, si, alpha, g * lam);
                    if ctx.offline, dV = dV + alpha * delta * E; else, vf.applyIncrement(alpha * delta * E); end
                    E = g * lam * E;
                end
                ob.step(r, s2i, a2 * (ctrl && ~term), sm.OutsideLast);
                if done, break; end
                si = s2i;
                if ctrl, ai = a2; else, ai = ctx.act(si); end
            end
            if ~ctrl && ctx.offline, vf.applyIncrement(dV); end
            ob.finish(term, trunc);
        end
    end
end
