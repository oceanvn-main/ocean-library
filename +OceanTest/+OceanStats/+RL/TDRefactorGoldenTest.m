classdef TDRefactorGoldenTest < OceanTest.BaseTestCase
    % TDREFACTORGOLDENTEST  The decomposed TDLearner must behave EXACTLY like the monolithic learner it replaced:
    %   * the 27 replay configurations (all modes, targets, Double Q-learning, Watkins, truncation, ...) regenerate a file
    %     byte-identical to td_replay_golden.json (final tables, pi, episode returns/lengths, every recorded episode with
    %     its states, actions, rewards, coin flips and termination flags) - so the random-number call order is unchanged;
    %   * 10 further configurations (softmax / decaying exploration, Blackjack control, trace and n-step prediction,
    %     custom policy, optimistic initial values, Double + softmax) equal td_extra_golden.mat bit for bit;
    %   * speed stays within +40% of the pre-refactor baseline (td_speed_baseline.mat).
    % The references were captured from the PRE-refactor learner and must never be regenerated from the new code.

    methods (Test)
        function testReplayCasesAreByteIdenticalToTheFrozenGolden(testCase)
            here = fileparts(mfilename('fullpath'));
            tmp = [tempname() '.json'];
            testCase.addTeardown(@() delete(tmp));
            evalc('OceanTest.OceanStats.RL.export_td_replay(tmp)');
            testCase.verifyTrue(strcmp(fileread(tmp), fileread(fullfile(here, 'td_replay_golden.json'))), ...
                'the regenerated replay file differs from the frozen golden');
        end

        function testExtraConfigurationsAreBitIdentical(testCase)
            here = fileparts(mfilename('fullpath'));
            g = load(fullfile(here, 'td_extra_golden.mat'));
            cases = OceanTest.OceanStats.RL.td_extra_cases();
            testCase.verifyEqual(string({g.G.name}), string(cases(:, 1))');
            for k = 1:size(cases, 1)
                [V, Q, pol, h] = cases{k, 2}().run();
                testCase.verifyEqual(V, g.G(k).V, ['V: ' cases{k, 1}]);
                testCase.verifyEqual(Q, g.G(k).Q, ['Q: ' cases{k, 1}]);
                testCase.verifyEqual(pol, g.G(k).pi, ['pi: ' cases{k, 1}]);
                testCase.verifyEqual(h, g.G(k).history, ['history: ' cases{k, 1}]);
            end
        end

        function testSpeedIsWithinFortyPercentOfTheMonolithicLearner(testCase)
            % absolute timings are machine dependent, so the baseline is scaled by a learner-independent reference workload
            % timed in the same session (td_speed_baseline.mat stores its time on the machine of the baseline)
            here = fileparts(mfilename('fullpath'));
            b = load(fullfile(here, 'td_speed_baseline.mat'));
            scale = OceanTest.OceanStats.RL.TDRefactorGoldenTest.reference_() / b.refSeconds;
            t = evalc('now = OceanTest.OceanStats.RL.benchmark_td_speed();');
            testCase.verifyEqual(now.name, b.baseline.name);
            for k = 1:numel(now.seconds)
                testCase.verifyLessThan(now.seconds(k), 1.4 * scale * b.baseline.seconds(k) + 0.05, ...
                    sprintf('%s: %.3fs vs scaled baseline %.3fs (scale %.2f) %s', now.name(k), now.seconds(k), scale * b.baseline.seconds(k), scale, t));
            end
        end
    end

    methods (Static)
        function t = reference_()
            % a fixed, learner-independent scalar workload (function calls, indexing, arithmetic): best of 3
            t = Inf;
            for rep = 1:3
                tic;  x = zeros(1, 50);  a = 0;
                for i = 1:4000000
                    x(1 + mod(i, 50)) = x(1 + mod(i, 50)) + sqrt(i) * 0.5;  a = a + x(1 + mod(i * 7, 50));
                end
                t = min(t, toc);
            end
            if a < 0, t = NaN; end                                  % keep the loop from being optimised away
        end
    end
end
