classdef WorldRandomActionParityTest < OceanTest.BaseTestCase
    % WORLDRANDOMACTIONPARITYTEST  Native vs door-built grid worlds under RANDOM actions (critic W2 finding 1): the
    % recorded env_golden.mat action rule never leaves the wind-free columns, so wind sampling (outcome order, one
    % uniform per step) is compared here against the live native environment, including resets that continue the stream.

    properties (TestParameter)
        Config = struct( ...
            'windyStochastic', {{'StochasticWind', true}}, ...
            'windyKingStay', {{'StochasticWind', true, 'KingMoves', true, 'StayAction', true}}, ...
            'windyDeterministic', {{}}, ...
            'windyLimited', {{'StochasticWind', true, 'TimeLimit', 20}})
        Seed = {0, 3, 11}
    end

    methods (Test)
        function rowsAreIdenticalUnderRandomActions(tc, Config, Seed)
            k = find(strcmp(Config(1:2:end), 'TimeLimit'));
            nativeArgs = Config;  limit = Inf;
            if ~isempty(k), limit = Config{2 * k};  nativeArgs(2 * k - 1:2 * k) = []; end
            native = OceanStats.RL.Env.Classic.WindyGridworldEnv(nativeArgs{:});
            if isfinite(limit), native = OceanStats.RL.Env.Wrappers.TimeLimit(native, limit); end
            world = OceanAgents.Devices.DeviceWorld.windy(Config{:});
            pick = RandStream('mt19937ar', 'Seed', 100 + Seed);
            [a, ia] = native.reset('Seed', Seed);  [b, ib] = world.reset('Seed', Seed);
            tc.verifyEqual(b, a);  tc.verifyEqual(ib.action_mask, ia.action_mask);
            stochastic = 0;
            for t = 1:300
                act = randi(pick, world.NumActions);
                [a, ra, ta, ua, ia] = native.step(act);  [b, rb, tb, ub, ib] = world.step(act);
                tc.assertEqual({b, rb, tb, ub, ib.action_mask}, {a, ra, ta, ua, ia.action_mask}, sprintf('step %d', t));
                if ta || ua
                    [a, ia] = native.reset();  [b, ib] = world.reset();                  % continue the streams
                    tc.assertEqual({b, ib.action_mask}, {a, ia.action_mask}, sprintf('reset after step %d', t));
                else
                    [~, p] = world.get_transitions(a, 1);
                    stochastic = stochastic + (numel(p) > 1);
                end
            end
            if any(strcmp(Config, 'StochasticWind'))
                tc.verifyGreaterThan(stochastic, 10, 'the run must actually visit windy (multi-outcome) states');
            end
        end
    end
end
