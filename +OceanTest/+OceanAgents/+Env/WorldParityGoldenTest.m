classdef WorldParityGoldenTest < OceanTest.BaseTestCase
    % WORLDPARITYGOLDENTEST  L2a / L2b (W2 contract section 7): the grid worlds REBUILT FROM DOORS, seen through
    % OceanAgents.Env.WorldAsEnvironment, are the same environments as the native classes.
    %   L2a  every in-scope configuration of env_golden.mat x every recorded seed: the recording's action rule
    %        (a = v(1 + mod(t-1, numel(v))) over the valid actions; reset() after done) reproduces gold.traj
    %        [s a r done trunc s'] BIT FOR BIT. Time-limit configurations are checked both ways: the world's own
    %        'TimeLimit' and the existing OceanStats.RL.Env.Wrappers.TimeLimit around the bridge.
    %   L2b  the exact model table, terminal mask and valid mask equal gold.model / gold.terminal / gold.validmask.
    % The reference file was captured from the ORIGINAL environments (capture_env_golden.m) and is only read here.

    methods (Static)
        function m = builders()
            % name -> cell of {label, @() environment}; the 13 in-scope configurations of env_golden.mat.
            G = @(f, varargin) feval(['OceanAgents.Devices.DeviceWorld.' f], varargin{:});   % S1: worlds made of devices
            B = @(env) env;                                                  % device worlds are environments
            TL = @(env, n) OceanStats.RL.Env.Wrappers.TimeLimit(env, n);
            one = @(f, varargin) {{"world", @() B(G(f, varargin{:}))}};
            both = @(n, f, varargin) {{"world TimeLimit", @() B(G(f, varargin{:}, 'TimeLimit', n))}, ...
                {"TimeLimit wrapper", @() TL(B(G(f, varargin{:})), n)}};
            m = containers.Map();
            m('walk5') = one('randomWalk');
            m('walk19') = one('randomWalk', 'NumInner', 19, 'LeftReward', -1, 'RightReward', 1);
            m('walk4even') = one('randomWalk', 'NumInner', 4);
            m('walk5_limit4') = both(4, 'randomWalk');
            m('walk1_limit1') = both(1, 'randomWalk', 'NumInner', 1);
            m('cliff') = one('cliffWalking');
            m('cliff3x5') = one('cliffWalking', 'Rows', 3, 'Cols', 5);
            m('cliff_limit20') = both(20, 'cliffWalking');
            m('windy') = one('windy');
            m('windy_king') = one('windy', 'KingMoves', true);
            m('windy_king_stay') = one('windy', 'KingMoves', true, 'StayAction', true);
            m('windy_stoch') = one('windy', 'StochasticWind', true);
            m('windy_small') = one('windy', 'StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]);
        end

        function g = golden()
            here = fileparts(which('OceanTest.OceanStats.RL.EnvGoldenParityTest'));
            g = load(fullfile(here, 'env_golden.mat'));
        end
    end

    methods (Test)
        function inScopeConfigurationsArePresent(tc)
            g = tc.golden();
            names = string({g.golden.name});
            tc.verifyEqual(numel(names), 15);
            scope = string(tc.builders().keys());
            tc.verifyEqual(numel(scope), 13);
            tc.verifyEmpty(setdiff(scope, names));
            tc.verifyEqual(sort(setdiff(names, scope)), ["maxbias" "maxbias3"]);      % out of scope for W2
        end

        function exactModelEqualsGoldenL2b(tc)
            g = tc.golden();
            builders = tc.builders();
            for k = 1:numel(g.golden)
                gold = g.golden(k);
                if ~isKey(builders, gold.name), continue; end
                for variant = builders(gold.name)
                    env = variant{1}{2}();
                    base = env.unwrapped();
                    S = base.NumStates;  A = base.NumActions;
                    model = zeros(0, 5);
                    for s = 1:S
                        for a = base.get_valid_actions(s)
                            [n, p, r] = base.get_transitions(s, a);
                            model = [model; repmat([s a], numel(n), 1), n(:), p(:), r(:)]; %#ok<AGROW>
                        end
                    end
                    label = sprintf('%s (%s)', gold.name, variant{1}{1});
                    tc.verifyEqual(model, gold.model, ['model ' label]);
                    tc.verifyEqual(base.TerminalMask, gold.terminal, ['terminal ' label]);
                    tc.verifyEqual(base.ValidMask, gold.validmask, ['valid mask ' label]);
                    tc.verifyEqual([S A], size(gold.validmask), ['sizes ' label]);
                end
            end
        end

        function seededTrajectoriesAreBitIdenticalL2a(tc)
            g = tc.golden();
            builders = tc.builders();
            compared = 0;
            for k = 1:numel(g.golden)
                gold = g.golden(k);
                if ~isKey(builders, gold.name), continue; end
                for variant = builders(gold.name)
                    mk = variant{1}{2};
                    for j = 1:numel(g.seeds)
                        env = mk();
                        base = env.unwrapped();
                        A = base.NumActions;
                        s = env.reset('Seed', g.seeds(j));
                        rows = zeros(400, 6);
                        for t = 1:400
                            v = base.get_valid_actions(s);
                            a = v(1 + mod(t - 1, numel(v)));
                            [s2, r, term, trunc, info] = env.step(a);
                            done = term || trunc;
                            rows(t, 1:5) = [s, a, r, done, trunc];
                            if done, s = env.reset(); else, s = s2; end
                            rows(t, 6) = s;
                            tc.assertEqual(size(info.action_mask), [1 A]);
                        end
                        tc.verifyTrue(isequal(rows, gold.traj{j}), sprintf('trajectory %s (%s) seed %d: first differing row %d', ...
                            gold.name, variant{1}{1}, g.seeds(j), find(any(rows ~= gold.traj{j}, 2), 1)));
                        compared = compared + 1;
                    end
                end
            end
            tc.verifyEqual(compared, (13 + 3) * numel(g.seeds));      % 3 limit configurations are run both ways
        end

        function bridgeContract(tc)
            env = OceanAgents.Env.WorldAsEnvironment(OceanAgents.World.Doors.GridWorlds.cliffWalking());
            tc.verifyTrue(isa(env, 'OceanStats.RL.Env.IEnvironment') && isa(env, 'OceanStats.RL.Env.IModelEnvironment'));
            tc.verifyEqual(env.unwrapped(), env);
            tc.verifyError(@() env.step(1), 'OceanStats:RL:Env:TabularEnvironment:NotReset');
            tc.verifyError(@() env.reset(), 'OceanAgents:World:NoSeed');            % no global-stream fallback
            tc.verifyError(@() env.set_state(5), 'OceanAgents:Env:NotSupported');
            env.setSeed(3);
            [s, info] = env.reset();                                                  % the pending seed is used
            tc.verifyEqual(s, 37);
            tc.verifyEqual(info.action_mask, true(1, 4));
            tc.verifyError(@() env.step(0), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');
            tc.verifyError(@() env.step(5), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');
            [s2, r, term, trunc] = env.step(2);                                       % into the cliff: back to start
            tc.verifyEqual([s2 r term trunc], [37 -100 0 0]);
            for a = [1, repmat(2, 1, 11), 3]                                          % up, 11 x right, down: the goal
                [s2, r, term, ~, info] = env.step(a);
            end
            tc.verifyEqual([s2 r], [48 -1]);
            tc.verifyTrue(term);
            tc.verifyEqual(info.action_mask, false(1, 4));
            tc.verifyError(@() env.step(1), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');   % ended
            tc.verifyEqual(env.get_all_states(), (1:48)');
            tc.verifyEqual([env.state_to_index(7) env.action_to_index(3)], [7 3]);
            tc.verifyTrue(env.is_terminal(48) && ~env.is_terminal(37));
            [n, p, r] = env.get_transitions(37, 2);
            tc.verifyEqual([n p r], [37 1 -100]);
            meta = env.getMetadata();
            tc.verifyFalse(meta.IsSymbolic);
            tc.verifyEqual(env.getPipelineSpec().Kind, "NONVIRTUAL");
        end

        function perStepCostReported(tc)
            % Performance is reported, not asserted: bridge vs native on CliffWalking (same action rule as L2a).
            function t = timeIt(env, n)
                base = env.unwrapped();
                s = env.reset('Seed', 1);
                tic;
                for k = 1:n
                    v = base.get_valid_actions(s);
                    [s2, ~, term, trunc] = env.step(v(1 + mod(k - 1, numel(v))));
                    if term || trunc, s = env.reset(); else, s = s2; end
                end
                t = toc / n;
            end
            n = 2000;
            tBridge = timeIt(OceanAgents.Env.WorldAsEnvironment(OceanAgents.World.Doors.GridWorlds.cliffWalking()), n);
            tNative = timeIt(OceanStats.RL.Env.Classic.CliffWalkingEnv(), n);
            tc.log(1, sprintf('CliffWalking per step: WorldAsEnvironment %.1f us, native %.1f us (x%.0f)', ...
                1e6 * tBridge, 1e6 * tNative, tBridge / tNative));
            tc.verifyGreaterThan(tBridge, 0);
        end
    end
end
