classdef TrustRulesTest < OceanTest.BaseTestCase
    % TRUSTRULESTEST  OceanAgents.Minds.TrustRules (A2 contract section 4.1): ExpectedUtilityRule H7-H9 (exact),
    % clamping, incompatibility, zero RNG use; ThompsonRule H12 (same stream state -> same choice; Beta moments of the
    % Marsaglia-Tsang sampler within a stated band; global stream untouched), its exact draw sequence, and pillars.

    methods (Static, Access = private)
        function c = h7Candidates()
            % H7: payoffs [10 -20 0]; label 1 = p1 Beta(3,2) (mean 0.6, var 0.04); 2 = p2 fixed 0.8; 3 = outside.
            c = struct('Labels', [1 2 3], 'Subjects', ["p1" "p2" "outside"], 'IsOutside', [false false true], ...
                'Estimate', [0.6 0.8 NaN], 'Variance', [0.04 NaN NaN], 'Alpha', [3 NaN NaN], 'Beta', [2 NaN NaN], ...
                'Payoff', [10 -20 0]);
        end
        function c = betaCandidates(alpha, beta)
            n = numel(alpha);
            c = struct('Labels', [1:n n + 1], 'Subjects', ["p" + (1:n) "outside"], 'IsOutside', [false(1, n) true], ...
                'Estimate', [alpha ./ (alpha + beta) NaN], ...
                'Variance', [alpha .* beta ./ ((alpha + beta).^2 .* (alpha + beta + 1)) NaN], ...
                'Alpha', [alpha NaN], 'Beta', [beta NaN], 'Payoff', [10 -20 0]);
        end
    end

    methods (Test)
        function expectedUtilityH7(tc)
            rs = RandStream('mrg32k3a', 'Seed', 1);
            before = rs.State;
            [label, info] = OceanAgents.Minds.TrustRules.ExpectedUtilityRule().choose(tc.h7Candidates(), rs);
            tc.verifyEqual(info.Scores, [-2 4 0]);           % U(p1) = 6 - 8, U(p2) = 8 - 4, U(outside) = 0
            tc.verifyEqual(label, 2);
            tc.verifyEqual(info.Probabilities, [0 1 0]);
            tc.verifyEqual(rs.State, before, 'the expected-utility rule must not draw');
        end

        function riskAdjustedH8AndClamp(tc)
            rule = OceanAgents.Minds.TrustRules.ExpectedUtilityRule('RiskAversion', 1);
            c = tc.h7Candidates();
            c.Estimate(2) = 0.5;                              % make p2 a Beta belief too: Beta(1,1), sd 1/sqrt(12)
            c.Variance(2) = 1 / 12;
            [~, info] = rule.choose(c, RandStream('mrg32k3a'));
            % H8: p~ = 0.6 - 1 * 0.2 = 0.4; U = 4 - 12 = -8 (floating point: within 1e-12)
            tc.verifyEqual(info.Scores(1), -8, 'AbsTol', 1e-12);
            % clamp: 0.5 - 1/sqrt(12) = 0.2113; and an estimate driven below 0 is clamped to 0 -> U = F
            c.Estimate(1) = 0.1;
            c.Variance(1) = 0.25;                             % 0.1 - 0.5 < 0
            [~, info] = rule.choose(c, RandStream('mrg32k3a'));
            tc.verifyEqual(info.Scores(1), -20);
            tc.verifyEqual(info.Scores(2), (0.5 - 1 / sqrt(12)) * 10 + (1 - (0.5 - 1 / sqrt(12))) * -20, 'AbsTol', 1e-12);
            % lambda > 0 on a fixed score (NaN variance) is refused, not treated as certain
            tc.verifyError(@() rule.choose(tc.h7Candidates(), RandStream('mrg32k3a')), 'OceanAgents:Minds:Incompatible');
            tc.verifyEqual(rule.describe().Requires, "variance");
            tc.verifyEmpty(OceanAgents.Minds.TrustRules.ExpectedUtilityRule().describe().Requires);
        end

        function tiesGoToTheLowestLabel(tc)
            % H9: equal expected utilities -> the lower label, whatever the candidate order.
            rule = OceanAgents.Minds.TrustRules.ExpectedUtilityRule();
            c = struct('Labels', [3 1 2], 'Subjects', ["p3" "p1" "outside"], 'IsOutside', [false false true], ...
                'Estimate', [0.8 0.8 NaN], 'Variance', NaN(1, 3), 'Alpha', NaN(1, 3), 'Beta', NaN(1, 3), ...
                'Payoff', [10 -20 0]);
            [label, info] = rule.choose(c, RandStream('mrg32k3a'));
            tc.verifyEqual(label, 1);
            tc.verifyEqual(info.Probabilities, [0 1 0]);
            % a delegate tied with the outside option: U = 0.5*10 + 0.5*(-10) = 0 = O -> lower label wins
            c2 = struct('Labels', [1 2], 'Subjects', ["p1" "outside"], 'IsOutside', [false true], ...
                'Estimate', [0.5 NaN], 'Variance', [NaN NaN], 'Alpha', [NaN NaN], 'Beta', [NaN NaN], 'Payoff', [10 -10 0]);
            tc.verifyEqual(rule.choose(c2, RandStream('mrg32k3a')), 1);
            c2.Labels = [2 1];
            tc.verifyEqual(rule.choose(c2, RandStream('mrg32k3a')), 1, 'outside has label 1 now');
        end

        function thompsonSameStateSameChoice(tc)
            % H12(a): replaying the same stream state reproduces the choice; no probabilities are invented.
            rule = OceanAgents.Minds.TrustRules.ThompsonRule();
            c = tc.betaCandidates([2 3 1], [2 1 1]);
            rs = RandStream('mrg32k3a', 'Seed', 17);
            labels = zeros(1, 40);
            states = cell(1, 40);
            for k = 1:40
                states{k} = rs.State;
                labels(k) = rule.choose(c, rs);
            end
            tc.verifyGreaterThan(numel(unique(labels)), 1, 'the draws must actually vary the choice');
            for k = [1 7 23 40]
                rs.State = states{k};
                [again, info] = rule.choose(c, rs);
                tc.verifyEqual(again, labels(k));
                tc.verifyEmpty(info.Scores);
                tc.verifyEmpty(info.Probabilities);
            end
        end

        function thompsonDrawSequence(tc)
            % Draw count/order: the choice consumes exactly one Beta draw (Gamma(alpha) then Gamma(beta)) per
            % delegate candidate in increasing label order, and nothing for the outside option.
            rule = OceanAgents.Minds.TrustRules.ThompsonRule();
            c = tc.betaCandidates([2 0.5 4], [3 0.7 1]);
            c.Labels = [4 2 1 3];                             % shuffled labels: p1->4, p2->2, p3->1, outside->3
            rs = RandStream('mrg32k3a', 'Seed', 5);
            ref = RandStream('mrg32k3a', 'Seed', 5);
            label = rule.choose(c, rs);
            S = 10; F = -20;
            p = zeros(1, 3);
            for j = [3 2 1]                                   % label order 1, 2, 4 = subjects p3, p2, p1
                p(j) = OceanAgents.Minds.TrustRules.ThompsonRule.sampleBeta(c.Alpha(j), c.Beta(j), ref);
            end
            tc.verifyEqual(rs.State, ref.State, 'same number and order of draws');
            U = [p * S + (1 - p) * F, 0];
            tc.verifyEqual(rule.LastSampledScores, U, 'AbsTol', 1e-12);
            [~, k] = max(U);
            tc.verifyEqual(label, c.Labels(k));
            % outside only: no draw at all
            only = struct('Labels', 1, 'Subjects', "outside", 'IsOutside', true, 'Estimate', NaN, 'Variance', NaN, ...
                'Alpha', NaN, 'Beta', NaN, 'Payoff', [10 -20 0]);
            before = rs.State;
            tc.verifyEqual(rule.choose(only, rs), 1);
            tc.verifyEqual(rs.State, before);
        end

        function betaSamplerMoments(tc)
            % H12(b): 20 000 draws. Band = 4 standard errors, computed from the Beta moments:
            %   SE(mean) = sqrt(Var/n);  SE(sample variance) ~ sqrt((mu4 - Var^2)/n), mu4 = kurtosis * Var^2.
            % Also exercises the shape < 1 boost with Beta(0.5, 0.5).
            n = 20000;
            for ab = [3 2; 0.5 0.5; 1 1]'
                a = ab(1); b = ab(2);
                rs = RandStream('mrg32k3a', 'Seed', 2026);
                x = zeros(1, n);
                for k = 1:n
                    x(k) = OceanAgents.Minds.TrustRules.ThompsonRule.sampleBeta(a, b, rs);
                end
                mu = a / (a + b);
                v = a * b / ((a + b)^2 * (a + b + 1));
                excess = 6 * ((a - b)^2 * (a + b + 1) - a * b * (a + b + 2)) / (a * b * (a + b + 2) * (a + b + 3));
                mu4 = (3 + excess) * v^2;
                label = sprintf('Beta(%g,%g)', a, b);
                tc.verifyLessThanOrEqual(abs(mean(x) - mu), 4 * sqrt(v / n), label);
                tc.verifyLessThanOrEqual(abs(var(x) - v), 4 * sqrt((mu4 - v^2) / n), label);
                tc.verifyTrue(all(x >= 0 & x <= 1), label);
            end
            % H1-style spot check of the numbers used: Beta(3,2) has mean 0.6 and variance 0.04 (H2)
            tc.verifyEqual(3 / 5, 0.6);
            tc.verifyEqual(3 * 2 / (25 * 6), 0.04, 'AbsTol', 1e-15);
        end

        function globalStreamUntouched(tc)
            % H12(c)
            g = RandStream.getGlobalStream();
            before = g.State;
            rule = OceanAgents.Minds.TrustRules.ThompsonRule();
            rs = RandStream('mrg32k3a', 'Seed', 3);
            for k = 1:200
                rule.choose(tc.betaCandidates([0.3 5], [0.4 2]), rs);
            end
            OceanAgents.Minds.TrustRules.ExpectedUtilityRule().choose(tc.h7Candidates(), rs);
            tc.verifyEqual(RandStream.getGlobalStream().State, before);
        end

        function thompsonNeedsBetaBeliefs(tc)
            rule = OceanAgents.Minds.TrustRules.ThompsonRule();
            tc.verifyError(@() rule.choose(tc.h7Candidates(), RandStream('mrg32k3a')), 'OceanAgents:Minds:Incompatible');
            tc.verifyEqual(rule.describe().Requires, "beta-parameters");
            tc.verifyTrue(rule.describe().UsesRandomness);
            bad = tc.h7Candidates();
            bad.Estimate = [0.6 0.8];                         % wrong length
            tc.verifyError(@() OceanAgents.Minds.TrustRules.ExpectedUtilityRule().choose(bad, RandStream('mrg32k3a')), ...
                'OceanAgents:Minds:Incompatible');
        end

        function pillars(tc)
            eu = OceanAgents.Minds.TrustRules.ExpectedUtilityRule.getMetadata();
            tc.verifyTrue(eu.IsSymbolic);
            tc.verifyEqual(eu.SymbolicLatex, "\tilde p = \mathrm{clamp}(\hat p - \lambda\sigma, 0, 1),\ U = \tilde p S + (1-\tilde p)F");
            th = OceanAgents.Minds.TrustRules.ThompsonRule.getMetadata();
            tc.verifyTrue(th.IsSymbolic);
            tc.verifyEqual(th.SymbolicLatex, "p \sim \mathrm{Beta}(\alpha,\beta),\ U = pS + (1-p)F");
            tc.verifySubstring(th.Reference, "Thompson, W. R. (1933)");
            tc.verifyTrue(any(contains(string(th.references), "Marsaglia")));
            required = {'id', 'display_name', 'short_description', 'category', 'textbook', 'pedagogy', 'equations', ...
                'Symbols', 'parameters', 'complexity', 'strengths', 'weaknesses', 'when_to_use', 'references'};
            tc.verifyTrue(all(isfield(eu, required)));
            tc.verifyTrue(all(isfield(th, required)));
            tc.verifyEqual(OceanAgents.Minds.TrustRules.ThompsonRule().getPipelineSpec().ComponentName, "ThompsonRule");
            tc.verifyEqual(OceanAgents.Minds.TrustRules.ExpectedUtilityRule().getPipelineSpec().Kind, "NONVIRTUAL");
        end
    end
end
