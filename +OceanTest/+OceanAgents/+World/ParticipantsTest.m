classdef ParticipantsTest < OceanTest.BaseTestCase
    % PARTICIPANTSTEST  OceanAgents.World.Participants (S1 step S1.2b-1): who is due when, moved out of GOD (Timeline).
    % due(t) for every-tick / every-k / on-event / never, nextDue exact against a brute-force scan, registration errors.
    % The cases are the former Timeline drive tests, unchanged in their expectations.

    methods (Static, Access = private)
        function P = make(kinds, ks)
            P = OceanAgents.World.Participants();
            for n = 1:numel(kinds), P.register("door" + n, kinds(n), ks(n)); end
        end
    end

    methods (Test)
        function duePerKind(tc)
            tl = tc.make(["every-tick" "every-k" "on-event" "never"], [1 3 1 1]);
            tc.verifyEqual(tl.due(0), [true true false false]);
            tc.verifyEqual(tl.due(1), [true false false false]);
            tc.verifyEqual(tl.due(6), [true true false false]);
            tc.verifyEqual(tl.due(7), [true false false false]);
            M = cell2mat(arrayfun(@(t) tl.due(t), (0:30)', 'UniformOutput', false));
            tc.verifyFalse(any(M(:, 3:4), 'all'), 'on-event and never are never driven by the clock');
            tc.verifyEqual(find(M(:, 2))' - 1, 0:3:30);
            tc.verifyError(@() tl.due(2.5), 'OceanAgents:World:BadValue');
            tc.verifyError(@() tl.due(-1), 'OceanAgents:World:BadValue');
        end

        function nextDueMatchesBruteForce(tc)
            % Exact closed form vs scanning t = after+1, after+2, ... for several random every-k assemblies.
            rs = RandStream('mrg32k3a', 'Seed', 3);
            kinds = ["every-k" "on-event" "never"];
            for trial = 1:25
                n = randi(rs, 4);
                kk = kinds(randi(rs, 3, 1, n));
                ks = randi(rs, 9, 1, n);
                tl = tc.make(kk, ks);
                for after = [-1 0 1 7 12 29 100]
                    expected = Inf;
                    if any(kk == "every-k")
                        for t = after + 1:after + 200
                            if any(tl.due(t)), expected = t; break; end
                        end
                    end
                    tc.verifyEqual(tl.nextDue(after), expected, sprintf('trial %d after %d', trial, after));
                end
            end
            every = tc.make(["every-k" "every-tick"], [5 1]);
            tc.verifyEqual(every.nextDue(41), 42);
            tc.verifyEqual(every.nextDue(-1), 0);
            idle = tc.make(["on-event" "never"], [1 1]);
            tc.verifyEqual(idle.nextDue(0), Inf, 'on-event/never do not count');
            tc.verifyError(@() every.nextDue(1.5), 'OceanAgents:World:BadValue');
        end

        function badRegistration(tc)
            ID = 'OceanAgents:World:BadAssembly';
            P = OceanAgents.World.Participants();
            tc.verifyError(@() P.register("x", "sometimes"), ID);
            tc.verifyError(@() P.register("x", "every-k", 0), ID);
            tc.verifyError(@() P.register("x", "every-k", 2.5), ID);
            tc.verifyError(@() P.register("", "every-tick"), ID);
            P.register("door1", "every-tick");
            tc.verifyError(@() P.register("door1", "never"), ID);
            empty = OceanAgents.World.Participants();
            tc.verifyEqual(empty.nextDue(0), Inf);
            tc.verifyEqual(size(empty.due(0)), [1 0]);
        end

        function pillars(tc)
            P = OceanAgents.World.Participants();
            tc.verifyFalse(P.getMetadata().IsSymbolic);
            tc.verifyEqual(P.getPipelineSpec().ComponentName, "Participants");
        end
    end
end
