classdef DQNBackup < OceanStats.RL.TD.Backups.IBackup
    % DQNBACKUP  Deep Q-Network backup with experience replay and periodic target network synchronization.
    %   Mnih et al. (2015) Human-level control through deep reinforcement learning.
    properties (Constant)
        Latex = "L(\theta) = \mathbb{E}\left[ \left( R + \gamma \max_{a'} Q(S', a'; \theta^-) - Q(S, A; \theta) \right)^2 \right]"
        Reference = "Mnih et al. (2015)"
    end
    properties
        BatchSize (1,1) double {mustBeInteger, mustBePositive} = 32
        TargetSyncInterval (1,1) double {mustBeInteger, mustBePositive} = 100
        ReplayBuffer
    end
    methods
        function obj = DQNBackup(replayBuf, batchSize, targetSyncInterval)
            if nargin >= 1, obj.ReplayBuffer = replayBuf; end
            if nargin >= 2, obj.BatchSize = batchSize; end
            if nargin >= 3, obj.TargetSyncInterval = targetSyncInterval; end
        end
        function s = name(~), s = "dqn"; end
        function tf = allowsOffPolicyTarget(~, ~), tf = true; end

        function [ret, L] = runEpisode(obj, ctx)
            g = ctx.gamma;  sm = ctx.sampler;  vf = ctx.values;  ob = ctx.observer;
            si = sm.start();
            ai = ctx.act(si);
            t = 0;  endTerm = false;  ret = 0;  L = 0;
            Sbuf = si;  Abuf = ai;  Rbuf = zeros(0, 1);
            ob.begin(si, ai);
            while true
                [s2i, r, term, ~, done] = sm.step(Abuf(t + 1));
                L = L + 1;  ret = ret + r;
                if sm.capped(done, L), done = true; end
                Rbuf(t + 1, 1) = r;  Sbuf(t + 2, 1) = s2i;

                % (1) Store transition in replay buffer
                if ~isempty(obj.ReplayBuffer)
                    obj.ReplayBuffer.store(Sbuf(t + 1), Abuf(t + 1), r, s2i, term);
                end

                % (2) Minibatch SGD: learn from a random batch of past transitions
                if ~isempty(obj.ReplayBuffer) && obj.ReplayBuffer.Size >= obj.BatchSize
                    batch = obj.ReplayBuffer.sampleBatch(obj.BatchSize, ctx.stream);
                    % batch.S : (BatchSize x 1) state indices
                    % batch.A : (BatchSize x 1) action indices
                    % batch.R : (BatchSize x 1) rewards
                    % batch.S2: (BatchSize x 1) next state indices
                    % batch.Done: (BatchSize x 1) logical terminal flags
                    for bi = 1:obj.BatchSize
                        bs  = batch.S(bi);   ba  = batch.A(bi);
                        br  = batch.R(bi);   bs2 = batch.S2(bi);
                        bterm = batch.Done(bi);
                        valid2 = sm.ValidMask(bs2, :);
                        qNext = vf.estimates(bs2);
                        y = ctx.target.compute(br, qNext, ba, bterm, g, ctx.behaviour(bs2), valid2);
                        cnt = vf.touch(bs, ba);
                        vf.update(bs, ba, ctx.stepSize.alpha(cnt), y);
                    end
                end

                % (3) Periodic target network sync (if value function supports it)
                if mod(L, obj.TargetSyncInterval) == 0 && ismethod(vf, 'syncTargetNetwork')
                    vf.syncTargetNetwork();
                end

                ob.step(r, s2i, 0, sm.OutsideLast);
                if done
                    endTerm = term;
                    break;
                end
                Abuf(t + 2, 1) = ctx.act(s2i);
                t = t + 1;
            end
            ob.finishArrays(Sbuf, Abuf, Rbuf, endTerm, ~endTerm, sm.OutsideLast);
        end
    end
end
