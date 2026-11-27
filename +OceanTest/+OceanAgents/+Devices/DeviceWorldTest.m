classdef DeviceWorldTest < OceanTest.BaseTestCase
    % DEVICEWORLDTEST  DV1 (DV1_Device_World_Contract.md): cliff walking made of devices is the recorded cliff walking.
    %   V1 every recorded seed of cliff, cliff3x5 and cliff_limit20 (both the reward device's time limit and the
    %      TimeLimit wrapper): rows [s a r done trunc s'] bit for bit against env_golden.mat
    %   V2 the exact model, terminal mask and valid mask equal the recording; value iteration equals the native one
    %   V4 the Laws hold (a forbidden cell cannot be entered; a refused step changes nothing) and the body's position
    %      moves only on the terrain's request

    methods (Static)
        function m = builders()
            D = @(varargin) OceanAgents.Devices.DeviceWorld.cliffWalking(varargin{:});
            TL = @(env, n) OceanStats.RL.Env.Wrappers.TimeLimit(env, n);
            m = containers.Map();
            m('cliff') = {{"devices", @() D()}};
            m('cliff3x5') = {{"devices", @() D('Rows', 3, 'Cols', 5)}};
            m('cliff_limit20') = {{"reward device limit", @() D('TimeLimit', 20)}, {"TimeLimit wrapper", @() TL(D(), 20)}};
        end

        function g = golden()
            here = fileparts(which('OceanTest.OceanStats.RL.EnvGoldenParityTest'));
            g = load(fullfile(here, 'env_golden.mat'));
        end
    end

    methods (Test)
        function seededTrajectoriesAreBitIdenticalV1(tc)
            g = tc.golden();
            builders = tc.builders();
            compared = 0;
            for k = 1:numel(g.golden)
                gold = g.golden(k);
                if ~isKey(builders, gold.name), continue; end
                for variant = builders(gold.name)
                    for j = 1:numel(g.seeds)
                        env = variant{1}{2}();
                        base = env.unwrapped();
                        s = env.reset('Seed', g.seeds(j));
                        rows = zeros(400, 6);
                        for t = 1:400
                            v = base.get_valid_actions(s);
                            a = v(1 + mod(t - 1, numel(v)));
                            [s2, r, term, trunc] = env.step(a);
                            done = term || trunc;
                            rows(t, 1:5) = [s, a, r, done, trunc];
                            if done, s = env.reset(); else, s = s2; end
                            rows(t, 6) = s;
                        end
                        tc.verifyTrue(isequal(rows, gold.traj{j}), sprintf('%s (%s) seed %d: first differing row %d', ...
                            gold.name, variant{1}{1}, g.seeds(j), find(any(rows ~= gold.traj{j}, 2), 1)));
                        compared = compared + 1;
                    end
                end
            end
            tc.verifyEqual(compared, 4 * numel(g.seeds));
        end

        function exactModelEqualsTheRecordingV2(tc)
            g = tc.golden();
            builders = tc.builders();
            for k = 1:numel(g.golden)
                gold = g.golden(k);
                if ~isKey(builders, gold.name), continue; end
                env = builders(gold.name);
                base = env{1}{2}().unwrapped();
                model = zeros(0, 5);
                for s = 1:base.NumStates
                    for a = base.get_valid_actions(s)
                        [n, p, r] = base.get_transitions(s, a);
                        model = [model; s a n p r]; %#ok<AGROW>
                    end
                end
                tc.verifyEqual(model, gold.model, ['model ' gold.name]);
                tc.verifyEqual(base.TerminalMask, gold.terminal, ['terminal ' gold.name]);
                tc.verifyEqual(base.ValidMask, gold.validmask, ['valid mask ' gold.name]);
            end
            vd = OceanStats.RL.DP.ValueIterationSolver(OceanAgents.Devices.DeviceWorld.cliffWalking(), 'DiscountFactor', 0.9);
            vn = OceanStats.RL.DP.ValueIterationSolver(OceanStats.RL.Env.Classic.CliffWalkingEnv(), 'DiscountFactor', 0.9);
            [Vd, pd] = vd.run();
            [Vn, pn] = vn.run();
            tc.verifyEqual(Vd, Vn);
            tc.verifyEqual(pd, pn);
        end

        function theLawsHoldV4(tc)
            % a wall: cell 26 (row 3, column 2) is forbidden; from 25 the move right is not permitted
            env = OceanAgents.Devices.DeviceWorld.cliffWalking('Forbidden', 26);
            tc.verifyFalse(env.ValidMask(25, 2));
            env.reset('Seed', 1);
            env.step(1);                                                      % 37 -> 25
            tc.verifyEqual(env.Position.Cell, 25);
            before = env.Position.exportState();
            tc.verifyError(@() env.step(2), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');
            tc.verifyEqual(env.Position.exportState(), before, 'a refused step changed nothing');
            [s, r] = env.step(1);                                             % 25 -> 13: still allowed
            tc.verifyEqual([s r], [13 -1]);
            % the position moves only on the terrain's request
            p = env.Position;
            reply = p.receive(OceanAgents.Minds.Signals.Signal.make(OceanAgents.Minds.Signals.SignalType('ValueKind', ...
                "record", 'Role', "command"), struct('Op', "moveTo", 'Cell', 48)));
            tc.verifyTrue(reply.isRefusal(), 'an unsigned request to move is refused');
            tc.verifyEqual(p.Cell, 13);
            tc.verifyError(@() OceanAgents.Devices.DeviceWorld.cliffWalking('Position', ...
                OceanAgents.Components.GridPosition(48, 'TrustedMover', "someone")), 'OceanAgents:Devices:BadConfig');
        end

        function cliffRulesByHand(tc)
            env = OceanAgents.Devices.DeviceWorld.cliffWalking();
            tc.verifyError(@() env.step(1), 'OceanStats:RL:Env:TabularEnvironment:NotReset');
            tc.verifyEqual(env.reset('Seed', 3), 37);
            [s, r, term] = env.step(2);                                       % into the cliff
            tc.verifyEqual([s r term], [37 -100 0]);
            for a = [1, repmat(2, 1, 11), 3]
                [s, r, term, ~, info] = env.step(a);
            end
            tc.verifyEqual([s r], [48 -1]);
            tc.verifyTrue(term);
            tc.verifyEqual(info.action_mask, false(1, 4));
            tc.verifyError(@() env.step(1), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');
            limited = OceanAgents.Devices.DeviceWorld.cliffWalking('TimeLimit', 2);
            limited.reset('Seed', 1);
            limited.step(1);
            [~, ~, term, trunc, info] = limited.step(1);
            tc.verifyEqual([term trunc], [false true]);
            tc.verifyEqual(info.successor_action_mask, true(1, 4));
        end
    end
end
