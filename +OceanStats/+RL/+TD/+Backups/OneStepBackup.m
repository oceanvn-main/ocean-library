classdef OneStepBackup < OceanStats.RL.TD.Backups.IBackup
    % ONESTEPBACKUP  Update right after every step with the target's one-step rule: TD(0) for state values; Sarsa,
    % Expected Sarsa and Q-learning for action values (which one is the Target strategy); Double Q-learning when the value
    % function is a double estimator (which decides its own coin and update order).
    properties (Constant)
        Latex = "x \leftarrow x + \alpha \left[ y_{t:t+1} - x \right]"
        Reference = "Sutton & Barto (2018) eq. 6.2, 6.7, 6.8 (one-step TD)"
    end
    methods
        function s = name(~), s = "onestep"; end
        function tf = supportsUpdateFirstValues(~), tf = true; end

        function [ret, L] = runEpisode(obj, ctx)
            if ctx.values.kind() == "state"
                [ret, L] = obj.predict_(ctx);
            elseif ctx.values.updatesBeforeNextAction()
                [ret, L] = obj.controlUpdateFirst_(ctx);
            else
                [ret, L] = obj.control_(ctx);
            end
        end
    end

    methods (Access = private)
        function [ret, L] = control_(~, ctx)
            % the next action is drawn BEFORE the update (Sarsa needs it as the target's action)
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
            si = sm.start();
            ai = ctx.act(si);
            ret = 0;  L = 0;
            ob.begin(si, ai);
            while true
                [s2i, r, term, trunc, done] = sm.step(ai);
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; trunc = true; end
                if ~term
                    p2 = ctx.behaviour(s2i);
                    a2 = ctx.draw(p2);
                else
                    p2 = zeros(1, sm.NumA);  a2 = 1;
                end
                vf.oneStep(ctx.target, ctx.stepSize, si, ai, r, s2i, a2, term, ctx.gamma, p2, sm.ValidMask(s2i, :), ctx.stream);
                ob.step(r, s2i, a2 * ~term, sm.OutsideLast);
                if done, break; end
                si = s2i;  ai = a2;
            end
            ob.finish(term, trunc);
        end

        function [ret, L] = controlUpdateFirst_(~, ctx)
            % a double estimator flips its coin and updates first; the next action is drawn from the updated tables
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
            si = sm.start();
            ai = ctx.act(si);
            ret = 0;  L = 0;
            ob.begin(si, ai);
            while true
                [s2i, r, term, trunc, done] = sm.step(ai);
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; trunc = true; end
                coin = vf.oneStep(ctx.target, ctx.stepSize, si, ai, r, s2i, 1, term, ctx.gamma, [], sm.ValidMask(s2i, :), ctx.stream);
                ob.step(r, s2i, 0, sm.OutsideLast);
                ob.coin(coin);
                if done, break; end
                si = s2i;
                ai = ctx.act(si);
                ob.action(ai);
            end
            ob.finish(term, trunc);
        end

        function [ret, L] = predict_(~, ctx)
            sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
            si = sm.start();
            ret = 0;  L = 0;
            ob.begin(si, 1);
            while true
                ai = ctx.act(si);
                [s2i, r, term, trunc, done] = sm.step(ai);
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; trunc = true; end
                vf.oneStep(ctx.target, ctx.stepSize, si, ai, r, s2i, 1, term, ctx.gamma, [], true, ctx.stream);
                ob.step(r, s2i, ai, sm.OutsideLast);
                if done, break; end
                si = s2i;
            end
            ob.finish(term, trunc);
        end
    end
end
