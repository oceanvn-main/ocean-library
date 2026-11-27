classdef CalendarTest < OceanTest.BaseTestCase
    % CALENDARTEST  OceanAgents.World.Calendar (W1_Kernel_Contract.md section 5.3): tie-break determinism under permuted
    % insertion, prediction replacement per (predictor, entity), cancelEntity, exact due(t, phase), nextTime,
    % retroactive errors, snapshot/restore, 1e5-prediction timing sanity, pillars.

    properties (Constant)
        Phases = ["resolve" "act" "settle" "project"]
        Doors = ["mover" "battery" "ager"]
    end

    methods (Access = private)
        function c = cal(tc)
            c = OceanAgents.World.Calendar('Phases', tc.Phases, 'Owners', tc.Doors);
        end
    end

    methods (Static, Access = private)
        function cmd = cmd(kind, target, issuerEntity, seq)
            switch kind
                case "remove", c0 = OceanAgents.World.Command.remove(target);
                case "detach", c0 = OceanAgents.World.Command.detach(target, "Charge");
            end
            cmd = c0.stamp("battery", issuerEntity, seq, [], "");
        end
    end

    methods (Test)
        function tieBreakIndependentOfInsertionOrder(tc)
            B = OceanTest.OceanAgents.Fixtures.World.Time.ItemBatch;
            b = B.random(7, 60);
            rs = RandStream('mrg32k3a', 'Seed', 11);
            ref = [];
            for trial = 1:6
                c = OceanAgents.World.Calendar();          % indices only (no names needed)
                B.schedule(c, b, randperm(rs, numel(b)));
                items = c.due(B.Time, B.Phase);
                tc.assertEqual(numel(items), numel(b));
                sig = [[items.OwnerRank]' [items.Rank]' [items.IssuerEntity]' [items.Seq]' ...
                    double([items.Kind]' == "command") [items.EntityID]'];
                if isempty(ref)
                    ref = sig;
                    % the order really is the tie-break key: rows are sorted lexicographically on
                    % (door rank, rule rank, issuer entity, seq) - time and phase are equal here
                    tc.verifyTrue(issortedrows(sig(:, 1:4)));
                else
                    tc.verifyEqual(sig, ref, sprintf('trial %d: order changed with insertion order', trial));
                end
                tc.verifyEqual(c.count(), 0);
            end
        end

        function fullKeyOrderAcrossTimeAndPhase(tc)
            c = tc.cal();
            c.schedulePrediction("ager/Pred", 4, 7, "act", "ager", "Old");
            c.schedulePrediction("battery/Pred", 3, 7, "act", "battery", "Empty", 'Rank', 2);
            c.scheduleCommand(tc.cmd("remove", 9, 1, 5), 7, "act", "battery", 'Rank', 1);
            c.scheduleCommand(tc.cmd("remove", 8, 1, 2), 7, "act", "battery", 'Rank', 1);
            c.scheduleCommand(tc.cmd("remove", 7, 0, 9), 7, "act", "battery", 'Rank', 1);
            c.schedulePrediction("mover/Pred", 1, 7, "settle", "mover", "Hit");
            c.schedulePrediction("mover/Pred", 2, 6, "project", "mover", "Hit");
            items = c.peek();
            tc.verifyEqual([items.Time], [6 7 7 7 7 7 7]);
            tc.verifyEqual([items.Phase], ["project" "act" "act" "act" "act" "act" "settle"]);
            % within (7, act): battery (rank 2) before ager (rank 3); rule rank 1 before 2; issuer 0 before 1; seq 2 < 5
            tc.verifyEqual([items(2:6).EntityID], [7 8 9 3 4]);
            tc.verifyEqual([items(2:4).Kind], repmat("command", 1, 3));
            tc.verifyEqual(items(5).PredictorKey, "battery/Pred");
            tc.verifyEqual(items(5).DueEvent, "Empty");
            tc.verifyEqual(items(5).Owner, "battery");
            tc.verifyTrue(isa(items(2).Command, 'OceanAgents.World.Command'));
            tc.verifyEqual(items(2).Command.Seq, 9);
        end

        function predictionReplacement(tc)
            c = tc.cal();
            c.schedulePrediction("battery/Pred", [1 2 3], [5 6 7], "resolve", "battery", "Empty");
            c.schedulePrediction("ager/Pred", 2, 8, "resolve", "ager", "Old");            % another key: kept
            c.schedulePrediction("battery/Pred", [2 3], [9 Inf], "resolve", "battery", "Empty");
            items = c.peek();
            tc.verifyEqual([items.EntityID], [1 2 2]);
            tc.verifyEqual([items.Time], [5 8 9]);
            tc.verifyEqual([items.PredictorKey], ["battery/Pred" "ager/Pred" "battery/Pred"]);
            tc.verifyEqual(c.count(), 3);
            % re-predicting the same time is still exactly one item
            c.schedulePrediction("battery/Pred", 1, 5, "resolve", "battery", "Empty");
            tc.verifyEqual(c.count(), 3);
            tc.verifyError(@() c.schedulePrediction("battery/Pred", [4 4], [1 2], "resolve", "battery", "Empty"), ...
                'OceanAgents:World:BadValue');
            tc.verifyError(@() c.schedulePrediction("battery/Pred", 4, 2.5, "resolve", "battery", "Empty"), ...
                'OceanAgents:World:BadValue');
            tc.verifyError(@() c.schedulePrediction("battery/Pred", [4 5], 3, "resolve", "battery", "Empty"), ...
                'OceanAgents:World:BadValue');
        end

        function cancelEntity(tc)
            c = tc.cal();
            c.schedulePrediction("battery/Pred", [1 2 3], [5 6 7], "resolve", "battery", "Empty");
            c.schedulePrediction("ager/Pred", [2 3], [8 8], "act", "ager", "Old");
            c.scheduleCommand(tc.cmd("remove", 2, 3, 1), 9, "settle", "battery");     % targets 2: dropped
            c.scheduleCommand(tc.cmd("remove", 3, 2, 2), 9, "settle", "battery");     % issued BY 2, targets 3: kept
            c.cancelEntity(2);
            items = c.peek();
            tc.verifyEqual([items.EntityID], [1 3 3 3]);
            tc.verifyEqual([items.Kind], ["prediction" "prediction" "prediction" "command"]);
            c.cancelEntity([1 3]);
            tc.verifyEqual(c.count(), 0);
            tc.verifyEqual(c.nextTime(), Inf);
        end

        function dueIsExact(tc)
            c = tc.cal();
            c.schedulePrediction("battery/Pred", [1 2 3], [5 5 6], "resolve", "battery", "Empty");
            c.schedulePrediction("ager/Pred", 1, 5, "act", "ager", "Old");
            tc.verifyEqual(c.nextTime(), 5);
            a = c.due(5, "resolve");
            tc.verifyEqual([a.EntityID], [1 2]);
            tc.verifyTrue(all([a.Phase] == "resolve") && all([a.Time] == 5));
            tc.verifyEqual(c.count(), 2);
            b = c.due(5, "act");
            tc.verifyEqual([b.EntityID b.Time], [1 5]);
            tc.verifyEqual(c.count(), 1);
            tc.verifyEqual(c.nextTime(), 6);
        end

        function dueSequenceAndNextTime(tc)
            c = tc.cal();
            tc.verifyEqual(c.nextTime(), Inf);
            c.schedulePrediction("battery/Pred", [1 2], [5 9], "act", "battery", "Empty");
            c.scheduleCommand(tc.cmd("remove", 4, 0, 1), 5, "settle", "battery");
            tc.verifyEqual(c.nextTime(), 5);
            tc.verifyEmpty(c.due(5, "resolve"));
            tc.verifyEqual([c.due(5, "act").EntityID], 1);
            tc.verifyEqual(c.nextTime(), 5);
            tc.verifyEqual(c.due(5, "settle").Kind, "command");
            tc.verifyEqual(c.nextTime(), 9);
            tc.verifyEmpty(c.due(5, "project"));
            tc.verifyEqual([c.due(9, "act").EntityID], 2);
            tc.verifyEqual(c.nextTime(), Inf);
            c.schedulePrediction("battery/Pred", 5, 20, "act", "battery", "Empty");
            c.clear();
            tc.verifyEqual([c.count() c.nextTime() c.Now c.NowPhase], [0 Inf 0 0]);
        end

        function retroactiveIsAnError(tc)
            c = tc.cal();
            c.setNow(10, 2);                                   % phases 1..2 of instant 10 released
            R = 'OceanAgents:World:Retroactive';
            tc.verifyError(@() c.schedulePrediction("b/P", 1, 9, "project", "battery", "E"), R);      % past time
            tc.verifyError(@() c.schedulePrediction("b/P", 1, 10, "act", "battery", "E"), R);         % current phase
            tc.verifyError(@() c.schedulePrediction("b/P", 1, 10, "resolve", "battery", "E"), R);     % past phase
            tc.verifyError(@() c.scheduleCommand(tc.cmd("remove", 1, 0, 1), 10, "act", "battery"), R);
            c.schedulePrediction("b/P", 1, 10, "settle", "battery", "E");                             % later phase: ok
            c.scheduleCommand(tc.cmd("remove", 1, 0, 1), 11, "resolve", "battery");                    % next instant: ok
            tc.verifyEqual(c.count(), 2);
            tc.verifyError(@() c.setNow(10, 1), R);             % the cursor never moves back
            tc.verifyError(@() c.due(9, "project"), R);
            % a failed schedule changes nothing
            tc.verifyEqual(c.count(), 2);
            % an item left behind (never released) is detected when the cursor passes it
            c2 = tc.cal();
            c2.schedulePrediction("b/P", 1, 12, "resolve", "battery", "E");
            tc.verifyError(@() c2.due(13, "resolve"), 'OceanAgents:World:InvalidState');
        end

        function namesAndIndices(tc)
            c = OceanAgents.World.Calendar();                   % no names declared: indices only
            c.schedulePrediction("x", 1, 3, 2, 1, "E");
            tc.verifyEqual(c.due(3, 2).PhaseIndex, 2);
            tc.verifyError(@() c.schedulePrediction("x", 1, 4, "act", 1, "E"), 'OceanAgents:World:BadValue');
            c2 = tc.cal();
            tc.verifyError(@() c2.schedulePrediction("x", 1, 4, "act", "nobody", "E"), 'OceanAgents:World:BadValue');
            tc.verifyError(@() c2.schedulePrediction("x", 1, 4, "noon", "mover", "E"), 'OceanAgents:World:BadValue');
            tc.verifyError(@() c2.schedulePrediction("x", 1, 4, "act", [], "E"), 'OceanAgents:World:BadValue');
            c2.schedulePrediction("x", 1, 4, 2, 3, "E");                                % indices also accepted
            it = c2.peek();
            tc.verifyEqual([it.Phase it.Owner], ["act" "ager"]);
        end

        function snapshotRestore(tc)
            c = tc.cal();
            c.schedulePrediction("battery/Pred", [1 2], [5 9], "act", "battery", "Empty");
            c.scheduleCommand(tc.cmd("detach", 4, 0, 1), 6, "settle", "battery");
            s = c.snapshot();
            before = c.peek();
            c.due(5, "act");
            c.cancelEntity(2);
            c.schedulePrediction("other", 7, 30, "act", "mover", "X");
            c.restore(s);
            tc.verifyEqual(c.peek(), before);
            tc.verifyEqual([c.Now c.NowPhase], [0 0]);
        end

        function manyPredictionsAreCheap(tc)
            % 1e5 entities: schedule all, re-predict half (replacement), release one instant, query nextTime.
            % Bound: 5 s in total (generous; vectorised storage does this in well under a second).
            n = 1e5;
            rs = RandStream('mrg32k3a', 'Seed', 1);
            ids = (1:n)';
            times = randi(rs, [1 1000], n, 1);
            c = tc.cal();
            t0 = tic;
            c.schedulePrediction("battery/Pred", ids, times, "resolve", "battery", "Empty");
            tSched = toc(t0);
            half = ids(1:2:end);
            newTimes = randi(rs, [1 1000], numel(half), 1);
            t1 = tic;
            c.schedulePrediction("battery/Pred", half, newTimes, "resolve", "battery", "Empty");
            tReplace = toc(t1);
            tc.verifyEqual(c.count(), n);
            expected = times;
            expected(1:2:end) = newTimes;
            t2 = tic;
            nt = c.nextTime();
            items = c.due(nt, "resolve");
            tDue = toc(t2);
            tc.verifyEqual(nt, min(expected));
            tc.verifyEqual(sort([items.EntityID]), find(expected == nt)');
            tc.verifyEqual(c.count(), n - nnz(expected == nt));
            total = tSched + tReplace + tDue;
            tc.log(1, sprintf('1e5 predictions: schedule %.3f s, replace half %.3f s, due+nextTime %.3f s', ...
                tSched, tReplace, tDue));
            tc.verifyLessThan(total, 5);
        end

        function pillars(tc)
            c = tc.cal();
            tc.verifyFalse(c.getMetadata().IsSymbolic);
            tc.verifyEqual(c.getPipelineSpec().ComponentName, "Calendar");
        end
    end
end
