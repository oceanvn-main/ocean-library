classdef SignalsTest < OceanTest.BaseTestCase
    % SIGNALSTEST  AB1 sections 3.1-3.4: signal types match by properties only, values are checked against their type,
    % waves are evaluated and their crossings solved exactly, and only the router can stamp a sender.

    methods (Test)
        function typesMatchByPropertiesOnly(tc)
            T = @(varargin) OceanAgents.Minds.Signals.SignalType(varargin{:});
            D = OceanAgents.World.Guard.Dimension;
            moves = T('ValueKind', "label", 'Labels', ["up" "right" "down" "left"], 'Role', "command");
            tc.verifyTrue(moves.accepts(T('ValueKind', "label", 'Labels', ["up" "down"], 'Role', "command")));
            tc.verifyEqual(moves.mismatch(T('ValueKind', "label", 'Labels', ["up" "fly"], 'Role', "command")), "Labels");
            tc.verifyEqual(moves.mismatch(T('ValueKind', "label", 'Role', "command")), "Labels");    % unrestricted offer
            tc.verifyEqual(moves.mismatch(T('ValueKind', "label", 'Labels', "up", 'Role', "sense")), "Role");
            speed = T('ValueKind', "number", 'Dimension', D.L() / D.T(), 'Role', "sense");
            tc.verifyEqual(speed.mismatch(T('ValueKind', "number", 'Dimension', D.E(), 'Role', "sense")), "Dimension");
            tc.verifyEqual(speed.mismatch(T('Shape', "stream", 'ValueKind', "number", 'Rate', 2, ...
                'Dimension', D.L() / D.T(), 'Role', "sense")), "Shape");
            s1 = T('Shape', "stream", 'ValueKind', "number", 'Rate', 1, 'Role', "sense");
            tc.verifyEqual(s1.mismatch(T('Shape', "stream", 'ValueKind', "number", 'Rate', 2, 'Role', "sense")), "Rate");
            id = 'OceanAgents:Signals:BadType';
            tc.verifyError(@() T('Shape', "stream", 'ValueKind', "number"), id);                    % no rate
            tc.verifyError(@() T('ValueKind', "flag", 'Labels', "x"), id);
            tc.verifyError(@() T('ValueKind', "label", 'Dimension', D.L()), id);
            tc.verifyError(@() T('Shape', "wave", 'ValueKind', "vector", 'Width', 2), id);
            tc.verifyError(@() T('ValueKind', "number", 'Width', 3), id);
        end

        function valuesAreCheckedAgainstTheirType(tc)
            T = @(varargin) OceanAgents.Minds.Signals.SignalType(varargin{:});
            S = @(t, v) OceanAgents.Minds.Signals.Signal.make(t, v);
            id = 'OceanAgents:Signals:BadValue';
            lab = T('ValueKind', "label", 'Labels', ["a" "b"]);
            tc.verifyEqual(S(lab, 'a').Value, "a");
            tc.verifyError(@() S(lab, "c"), id);
            vec = T('ValueKind', "vector", 'Width', 3, 'Role', "sense");
            tc.verifyEqual(S(vec, [1 2 3]).Value, [1 2 3]);
            tc.verifyError(@() S(vec, [1 2]), id);
            tc.verifyError(@() S(T('ValueKind', "number"), NaN), id);
            tc.verifyEqual(S(T('ValueKind', "flag"), 1).Value, true);
            tc.verifyError(@() S(T('ValueKind', "record"), struct('h', {{@sin}})), 'OceanAgents:Core:InvalidRecord');
            wave = T('Shape', "wave", 'ValueKind', "number", 'Role', "sense");
            tc.verifyClass(S(wave, OceanAgents.Minds.Signals.Wave.constant(3)).Value, 'OceanAgents.Minds.Signals.Wave');
            tc.verifyError(@() S(wave, 3), id);
            tc.verifyFalse(S(lab, "a").isStamped());
        end

        function wavesAreEvaluated(tc)
            W = OceanAgents.Minds.Signals.Wave;
            t = 0:0.25:3;
            tc.verifyEqual(W.constant(2).valueAt(t), 2 + 0 * t);
            tc.verifyEqual(W.ramp(1, 0.5, 'T0', 1).valueAt(t), 1 + 0.5 * (t - 1), 'AbsTol', 1e-15);
            tc.verifyEqual(W.sine(2, 0.25, 'Offset', 1, 'Phase', 0.3).valueAt(t), 1 + 2 * sin(2 * pi * 0.25 * t + 0.3), ...
                'AbsTol', 1e-15);
            tc.verifyError(@() OceanAgents.Minds.Signals.Wave('Form', "square"), 'OceanAgents:Signals:BadValue');
        end

        function crossingsAreExact(tc)
            % nextCrossing against a dense grid: the first sign change after `after` brackets the analytic answer
            W = OceanAgents.Minds.Signals.Wave;
            waves = {W.ramp(-1, 0.7), W.ramp(4, -1.3, 'T0', 2), W.sine(2, 0.25, 'Offset', 1, 'Phase', 0.4), ...
                     W.sine(-1.5, 0.6, 'Offset', 0.2, 'T0', 0.3)};
            levels = [0.5 1.6 -0.4];                      % none equal to a peak (touching is not crossing)
            grid = 0:1e-4:12;
            for k = 1:numel(waves)
                v = waves{k}.valueAt(grid);
                for L = levels
                    for after = [0 1.3 4.05]
                        for dir = ["up" "down" "either"]
                            t = waves{k}.nextCrossing(L, after, dir);
                            d = diff(sign(v - L));
                            switch dir
                                case "up",   hit = d > 0;
                                case "down", hit = d < 0;
                                otherwise,   hit = d ~= 0;
                            end
                            idx = find(hit & grid(1:end-1) >= after, 1);
                            if isempty(idx)
                                tc.verifyTrue(t > grid(end) - 1e-3, sprintf('wave %d L %g after %g %s', k, L, after, dir));
                            else
                                tc.verifyEqual(t, grid(idx), 'AbsTol', 2e-4, sprintf('wave %d L %g after %g %s', k, L, after, dir));
                                tc.verifyEqual(waves{k}.valueAt(t), L, 'AbsTol', 1e-9);
                            end
                        end
                    end
                end
            end
            tc.verifyEqual(W.sine(1, 1).nextCrossing(1, 0), Inf);        % touching a peak is not a crossing
            tc.verifyEqual(W.constant(1).nextCrossing(1, 0), Inf);
            tc.verifyEqual(W.ramp(0, 1).nextCrossing(0.5, 0.5), Inf);   % strictly after
        end
    end
end
