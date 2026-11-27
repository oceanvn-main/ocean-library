classdef NeuralSemiGradientBackup < OceanStats.RL.TD.Backups.IBackup
    % NEURALSEMIGRADIENTBACKUP  Semi-gradient TD backup for neural network function approximation.
    %   Sutton & Barto (2018) Ch. 9 & 10: Semi-gradient TD(0) and Semi-gradient Sarsa.
    properties (Constant)
        Latex = "\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ U_t - \hat{v}(S_t, \mathbf{w}) \right] \nabla \hat{v}(S_t, \mathbf{w})"
        Reference = "Sutton & Barto (2018) Ch. 9 & 10"
    end
    methods
        function s = name(~), s = "neuralsemigradient"; end
        function tf = allowsOffPolicyTarget(~, ~), tf = true; end

        function [ret, L] = runEpisode(~, ctx)
            g = ctx.gamma;  sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
            ctrl = vf.kind() == "action";
            s0 = sm.start();
            Sbuf = s0;  Abuf = zeros(0, 1);  Rbuf = zeros(0, 1);
            if ctrl, Abuf(1) = ctx.act(s0); else, Abuf(1) = 1; end
            t = 0;  endTerm = false;  ret = 0;  L = 0;
            while true
                a_t = Abuf(t + 1);
                [s2i, r, term, ~, done] = sm.step(a_t);
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; end
                Rbuf(t + 1, 1) = r;  Sbuf(t + 2, 1) = s2i;
                if done
                    endTerm = term;
                    break;
                end
                if ctrl, a_next = ctx.act(s2i); else, a_next = 1; end
                Abuf(t + 2, 1) = a_next;
                
                % Compute TD Target
                qNext = vf.estimates(s2i);
                y = ctx.target.compute(r, qNext, a_next, term, g, ctx.behaviour(s2i), sm.ValidMask(s2i, :));
                
                % Semi-gradient update step
                vf.update(Sbuf(t + 1), a_t, ctx.stepSize.alpha(1), y);
                
                t = t + 1;
            end
            ob.finishArrays(Sbuf, Abuf, Rbuf, endTerm, ~endTerm, sm.OutsideLast);
        end
    end
end
