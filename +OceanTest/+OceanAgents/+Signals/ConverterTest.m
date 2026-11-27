classdef ConverterTest < OceanTest.BaseTestCase
    % CONVERTERTEST  AB1 section 3.6 and invariant I8: hysteresis prevents chatter on a noisy reading; skipping to the
    % predicted crossing gives the same events at the same ticks as stepping every tick; the sampler samples on time.

    methods (Static)
        function [ticks, values, visits] = runStepping(wave, up, down, horizon)
            [ticks, values, visits] = OceanTest.OceanAgents.Signals.ConverterTest.drive_(wave, up, down, horizon, "clock");
        end

        function [ticks, values, visits] = runSkipping(wave, up, down, horizon)
            [ticks, values, visits] = OceanTest.OceanAgents.Signals.ConverterTest.drive_(wave, up, down, horizon, "calendar");
        end

        function [ticks, values, visits] = drive_(wave, up, down, horizon, binding)
            c = OceanAgents.Components.Converters.ThresholdConverter(up, down);
            W = OceanAgents.Minds.Signals.SignalType('Shape', "wave", 'ValueKind', "number", 'Role', "sense");
            c.receive(OceanAgents.Minds.Signals.Signal.make(W, wave));
            c.useBinding(binding);
            [ticks, out, visits] = OceanAgents.Components.Scheduler.run(c, ...
                @(t) OceanAgents.Minds.Signals.Signal.make(c.InType, wave.valueAt(t)), horizon);
            values = logical([out.Value]);
            if isempty(out), values = false(1, 0); end
        end
    end

    methods (Test)
        function hysteresisPreventsChatter(tc)
            c = OceanAgents.Components.Converters.ThresholdConverter(1.0, 0.5);
            rs = RandStream('mrg32k3a', 'Seed', 3);
            x = 1.0 + 0.1 * randn(rs, 1, 500);                     % hovering at the upper threshold
            n = 0;
            for v = x
                n = n + numel(c.receive(OceanAgents.Minds.Signals.Signal.make(c.InType, v)));
            end
            tc.verifyEqual(n, 1, 'one switch up, never back down (noise never reaches the lower level)');
            plain = 0;  above = false;                             % the same readings without hysteresis
            for v = x
                if (v >= 1) ~= above, plain = plain + 1; above = v >= 1; end
            end
            tc.verifyGreaterThan(plain, 100);
            tc.verifyError(@() OceanAgents.Components.Converters.ThresholdConverter(1, 1), 'OceanAgents:Signals:BadValue');
        end

        function skippingEqualsStepping(tc)
            W = OceanAgents.Minds.Signals.Wave;
            cases = {W.sine(2, 1 / 37, 'Offset', 0.5, 'Phase', 0.2), W.ramp(-3, 0.11), ...
                     W.sine(1.5, 1 / 13, 'Offset', 1, 'T0', 2.5), W.ramp(5, -0.07, 'T0', 1)};
            for k = 1:numel(cases)
                [ts, vs] = tc.runStepping(cases{k}, 1.2, 0.3, 400);
                [tj, vj, visits] = tc.runSkipping(cases{k}, 1.2, 0.3, 400);
                tc.verifyEqual(tj, ts, sprintf('case %d ticks', k));
                tc.verifyEqual(vj, vs, sprintf('case %d values', k));
                tc.verifyLessThan(visits, 401, 'skipping must visit fewer ticks than stepping');
            end
        end

        function samplerSamplesOnTime(tc)
            s = OceanAgents.Components.Converters.Sampler(4);
            tc.verifyEqual(s.nextChange(0), 0.25);
            tc.verifyEqual(s.nextChange(0.25), 0.5);
            r = OceanAgents.Minds.Signals.Router();
            src = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("source"), "source");
            r.attach(s, "sampler");
            r.addRoute("in", "sampler", "sense");
            W = OceanAgents.Minds.Signals.SignalType('Shape', "wave", 'ValueKind', "number", 'Role', "sense");
            wave = OceanAgents.Minds.Signals.Wave.sine(1, 0.5);
            r.advanceTo(0.25);
            out = src.send("in", OceanAgents.Minds.Signals.Signal.make(W, wave));
            tc.verifyEqual(out.Value, wave.valueAt(0.25), 'AbsTol', 1e-15);
            tc.verifyEqual(out.Type.Shape, "stream");
            tc.verifyTrue(s.receive(OceanAgents.Minds.Signals.Signal.make(W, wave)).isRefusal());   % unstamped: no time
        end

        function skippingEqualsSteppingAtPeaks(tc)
            % critic AB1 finding 5: Up equal to a peak (or Down equal to a trough) is reached by touching
            W = OceanAgents.Minds.Signals.Wave;
            cases = {{W.sine(1, 0.25), 1, -0.5}, {W.sine(1, 1/8, 'Offset', 0.2), 1.2, -0.8}, ...
                     {W.sine(2, 1/16), 1.5, -2}, {W.constant(3), 3, 1}};
            for k = 1:numel(cases)
                c = cases{k};
                [ts, vs] = tc.runStepping(c{1}, c{2}, c{3}, 200);
                [tj, vj] = tc.runSkipping(c{1}, c{2}, c{3}, 200);
                tc.verifyEqual(tj, ts, sprintf('case %d ticks', k));
                tc.verifyEqual(vj, vs, sprintf('case %d values', k));
                tc.verifyNotEmpty(ts, sprintf('case %d must fire', k));
            end
        end

        function bindingsChangeCostNeverResults(tc)
            % AB1 invariant I7: the Scheduler reads the binding ("clock" visits every tick, "calendar" only the
            % predicted ones); the events must be identical and the calendar must visit fewer ticks
            W = OceanAgents.Minds.Signals.Wave;
            waves = {W.sine(2, 1/37, 'Offset', 0.5), W.ramp(-3, 0.11), W.sine(1, 0.25)};
            levels = [1.2 0.3; 1.2 0.3; 0.5 0];
            for k = 1:numel(waves)
                [t1, vc, nc] = tc.runStepping(waves{k}, levels(k, 1), levels(k, 2), 300);
                [tk, vk, nk] = tc.runSkipping(waves{k}, levels(k, 1), levels(k, 2), 300);
                tc.verifyEqual(tk, t1, sprintf('wave %d ticks', k));
                tc.verifyEqual(vk, vc, sprintf('wave %d values', k));
                tc.verifyEqual(nc, 301);
                tc.verifyLessThanOrEqual(nk, nc);
            end
            c = OceanAgents.Components.Converters.ThresholdConverter(1, 0);
            tc.verifyError(@() c.useBinding("array"), 'OceanAgents:Components:BadBinding');
        end

        function skippingEqualsSteppingWhenATickHitsTheLevel(tc)
            % critic AB1 round 2 blocker: a level equal to the wave's value at a tick (within rounding)
            W = OceanAgents.Minds.Signals.Wave;
            cases = {{W.sine(1, 0.25), 0.5, 0}, {W.sine(2, 1/8), sqrt(2), -sqrt(2)}, {W.sine(1, 0.25), 1, 0}, ...
                     {W.sine(1, 1/6, 'Offset', 0.3), 0.3 + sin(pi / 3), 0.3}};
            for k = 1:numel(cases)
                c = cases{k};
                [ts, vs] = tc.runStepping(c{1}, c{2}, c{3}, 300);
                [tj, vj] = tc.runSkipping(c{1}, c{2}, c{3}, 300);
                tc.verifyEqual(tj, ts, sprintf('case %d ticks', k));
                tc.verifyEqual(vj, vs, sprintf('case %d values', k));
            end
            % random sweep: rational frequencies and levels taken from the wave's own value at a tick
            rs = RandStream('mrg32k3a', 'Seed', 11);
            bad = 0;
            for n = 1:300
                f = 1 / (2 + randi(rs, 30));  A = 0.5 + 2 * rand(rs);  off = randn(rs);  ph = (randi(rs, 8) - 1) * pi / 4;
                w = W.sine(A, f, 'Offset', off, 'Phase', ph, 'T0', randi(rs, 5) - 3);
                lv = sort(w.valueAt(randi(rs, 40, 1, 2)), 'descend');
                if lv(1) - lv(2) < 1e-6, continue; end
                [ts, vs] = tc.runStepping(w, lv(1), lv(2), 120);
                [tj, vj] = tc.runSkipping(w, lv(1), lv(2), 120);
                bad = bad + ~(isequal(ts, tj) && isequal(vs, vj));
            end
            tc.verifyEqual(bad, 0, 'skip must equal step for every random wave');
        end
    end
end
