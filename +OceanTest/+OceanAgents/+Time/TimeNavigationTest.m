classdef TimeNavigationTest < OceanTest.BaseTestCase
    % TIMENAVIGATIONTEST  The time-package utilities that sit beside the world kernel (no clock of their own):
    % TimeContext, CommitLog (history), PiecewiseFormula, QueryCache, ViewCursor, FrameCoordinator, MediaTimeMap,
    % ColumnStore. Kernel-backed capture and active registration are in KernelIntegrationTest.

    methods (Static, Access = private)
        function ctx = ctx(t, commit, varargin)
            ctx = OceanAgents.Time.TimeContext('DomainID', "w", 'Time', t, 'Commit', commit, varargin{:});
        end

        function [log, fc, cur] = chessSetup()
            % a history of eleven committed instants: pos = t, turn = mod(t, 2)
            log = OceanAgents.Time.CommitLog();
            for t = 0:10
                log.append(OceanTest.OceanAgents.Time.TimeNavigationTest.ctx(t, t), struct('pos', t, 'turn', mod(t, 2)));
            end
            fc = OceanAgents.Time.FrameCoordinator();
            fc.addSource("w", log);
            cur = OceanAgents.Time.ViewCursor("w");
        end
    end

    methods (Test)
        % ---------------- TimeContext ----------------
        function contextIdentityIsTheCommit(tc)
            a = tc.ctx(0, 1); b = tc.ctx(0, 2);
            tc.verifyFalse(a.sameCommit(b), 'same time, different commit');
            tc.verifyTrue(a.sameCommit(tc.ctx(5, 1)));
            c = a.withGeneration(4).withTime(9);
            tc.verifyEqual([c.CursorGeneration c.Time c.ActiveSetRevision], [4 9 0]);
        end

        % ---------------- PiecewiseFormula ----------------
        function rebaseKeepsHistoryAndRoots(tc)
            f = OceanAgents.Time.PiecewiseFormula(0, 100, struct('rate', -2));
            tc.verifyEqual(f.valueAt(10), 80);
            tc.verifyEqual(f.nextCrossing(0), 50);
            f.rebase(10, struct('rate', -1), 'Delta', 50);   % deposit +50 and slower drain: 80 + 50 = 130 at 10
            tc.verifyEqual(f.valueAt(5), 90);                 % old segment still exact
            tc.verifyEqual(f.valueAt(10), 130);
            tc.verifyEqual(f.valueAt(12), 128);
            tc.verifyEqual(f.nextCrossing(0), 140);           % 10 + 130/1
            tc.verifyEqual(f.segmentCount(), 2);
            tc.verifyError(@() f.valueAt(-1), 'OceanAgents:Time:NoCoverage');
            tc.verifyError(@() f.rebase(3, struct('rate', 0)), 'OceanAgents:Time:Retroactive');
            g = OceanAgents.Time.PiecewiseFormula(0, 1, struct(), 'Fcn', @(dt, v0, p) v0 + dt^2);
            tc.verifyError(@() g.nextCrossing(5), 'OceanAgents:Time:NoRootDetection');
            h = OceanAgents.Time.PiecewiseFormula(0, 5, struct('rate', 1));
            tc.verifyEqual(h.nextCrossing(0), Inf, 'a rising value never reaches a lower level');
        end

        % ---------------- CommitLog ----------------
        function logHoldsLatestCommitAndNeverInvents(tc)
            log = OceanAgents.Time.CommitLog();
            log.append(tc.ctx(0, 0), struct('pos', 0));
            log.append(tc.ctx(0, 1), struct('pos', 1));           % first step shares tick 0 with the reset
            r = log.at(0, "after");
            tc.verifyEqual(r.Context.Commit, 1);                  % same time, different identity
            tc.verifyEqual(log.at(0, "before").Status, "unavailable");
            tc.verifyEqual(log.at(-3).Status, "unavailable");
            tc.verifyError(@() log.append(tc.ctx(-1, 2), struct()), 'OceanAgents:Time:Retroactive');
        end

        function expiryAtSeekBoundaryBeforeAfter(tc)
            log = OceanAgents.Time.CommitLog();
            log.append(tc.ctx(4, 1), struct('alive', true));
            log.append(tc.ctx(5, 2), struct('alive', false));      % the expiry's commit
            tc.verifyTrue(log.at(5, "before").State.alive);
            tc.verifyFalse(log.at(5, "after").State.alive);
        end

        function logIsBounded(tc)
            log = OceanAgents.Time.CommitLog('MaxCommits', 10);
            for t = 1:100
                log.append(tc.ctx(t, t), struct('v', t));
            end
            tc.verifyEqual(log.count(), 10);
            tc.verifyEqual(log.coverage(), [91 100]);
            tc.verifyEqual(log.at(50).Status, "unavailable");     % dropped history is reported, not invented
        end

        function episodesAreNeverMixed(tc)
            log = OceanAgents.Time.CommitLog();
            for t = 0:2, log.append(tc.ctx(t, t, 'Episode', 1), struct('v', 10 + t)); end
            for t = 0:1, log.append(tc.ctx(t, 3 + t, 'Episode', 2), struct('v', 20 + t)); end
            tc.verifyEqual(log.at(1).State.v, 21, 'default: the latest episode');
            tc.verifyEqual(log.at(1, "after", 1).State.v, 11);
            tc.verifyEqual(log.at(5, "after", 2).State.v, 21);
            tc.verifyEqual(log.coverage(1), [0 2]);
            tc.verifyEqual(log.at(0, "after", 9).Status, "unavailable");
        end

        % ---------------- Cursor and frames ----------------
        function framesBatchOnceAndStayCoherent(tc)
            [log, fc, cur] = tc.chessSetup();
            fc.addView("scene", "w", ["pos" "turn"]);
            fc.addView("plot", "w", "pos");              % shares "pos" with the scene
            fc.addView("hidden", "w", "turn", 'Visible', false);
            cur.seek(6);
            b = fc.frame(cur);
            tc.verifyEqual(b.Values.w.pos, 6);
            tc.verifyEqual(b.Views.plot.pos, 6);
            tc.verifyEqual(fc.Evaluations, 2);           % pos once (shared) + turn once
            tc.verifyFalse(isfield(b.Views, 'hidden'));
            tc.verifyEqual(b.Contexts{1}.Time, 6);
            tc.verifyEqual(b.Staleness.w, 0);
            ver = log.Version; n = log.count();
            cur.seek(2.5);
            b = fc.frame(cur);
            tc.verifyEqual(b.Values.w.pos, 2);           % held commit at 2, staleness 0.5
            tc.verifyEqual(b.Staleness.w, 0.5);
            tc.verifyEqual([log.Version log.count()], [ver n], 'reading history never changes it');
        end

        function pausedUnchangedViewDoesNoWork(tc)
            [~, fc, cur] = tc.chessSetup();
            fc.addView("scene", "w", ["pos" "turn"]);
            cur.seek(4);
            fc.frame(cur);
            e0 = fc.Evaluations; f0 = fc.Frames;
            b = fc.frame(cur);
            tc.verifyFalse(b.Changed);
            tc.verifyEqual([fc.Evaluations fc.Frames], [e0 f0]);
            cur.seek(6); fc.frame(cur); e1 = fc.Evaluations;
            cur.seek(4); b = fc.frame(cur);              % an already evaluated commit is served from the cache
            tc.verifyEqual(fc.Evaluations, e1);
            tc.verifyTrue(b.Changed);
            tc.verifyGreaterThan(fc.Cache.Hits, 0);
        end

        function rapidSeekSupersedesLateResults(tc)
            [~, fc, cur] = tc.chessSetup();
            fc.addView("scene", "w", "pos");
            cur.seek(2); slow = fc.frame(cur);           % request whose I/O is "delayed"
            cur.seek(8); fresh = fc.frame(cur);
            tc.verifyFalse(fc.accept(slow, cur));         % a late result cannot overwrite the current cursor
            tc.verifyTrue(fc.accept(fresh, cur));
            tc.verifyEqual(fc.Latest.Values.w.pos, 8);
        end

        function mixedRatePanelsShowOwnTimes(tc)
            [~, fc, cur] = tc.chessSetup();
            slowLog = OceanAgents.Time.CommitLog();
            slowLog.append(OceanAgents.Time.TimeContext('DomainID', "slow", 'Time', 3), struct('v', 1));
            fc.addSource("slow", slowLog);
            fc.addView("s", "slow", "v"); fc.addView("scene", "w", "pos");
            c2 = OceanAgents.Time.ViewCursor("slow"); c2.seek(3);
            cur.seek(7);
            b = fc.frame([cur c2]);
            tc.verifyEmpty(b.CommonBoundary);              % unsynchronized: no implied common time
            tc.verifyEqual([b.Contexts{1}.Time b.Contexts{2}.Time], [7 3]);
        end

        function cursorPlaysBackwardAndClamps(tc)
            c = OceanAgents.Time.ViewCursor("w", 'Span', [0 10], 'Target', 5);
            c.play(-2); g = c.advance(1);
            tc.verifyEqual(c.Target, 3);
            tc.verifyTrue(c.isCurrent(g));
            c.advance(5);
            tc.verifyEqual(c.Target, 0);
            tc.verifyEqual(c.Rate, 0);                     % stops at the end
            g0 = c.Generation; c.advance(1);
            tc.verifyEqual(c.Generation, g0);              % paused: nothing changes
        end

        function formulaChannelEvaluatesAtResolvedTime(tc)
            [~, fc, cur] = tc.chessSetup();
            f = OceanAgents.Time.PiecewiseFormula(0, 100, struct('rate', -2));
            fc.addFormulaChannel("w", "charge", f);
            fc.addView("g", "w", "charge");
            cur.seek(5);
            b = fc.frame(cur);
            tc.verifyEqual(b.Values.w.charge, 90);
            f.rebase(8, struct('rate', 0), 'Delta', 10);  % a new segment changes the cache stamp
            cur.seek(5); cur.seek(5);
            b = fc.frame(cur);
            tc.verifyEqual(b.Values.w.charge, 90);         % history unchanged by the rebase
        end

        % ---------------- Extensions ----------------
        function reverseVariableRateVideo(tc)
            pts = [0 1 3 4 8 9 11]';
            m = OceanAgents.Time.MediaTimeMap(pts, 'Keyframes', [1 4 6], 'Offset', 10);
            tc.verifyEqual(m.frameAt(10), 1);
            tc.verifyEqual(m.frameAt(13.5), 3);             % hold: variable rate, not index/fps
            tc.verifyEqual(m.frameAt(9), 0);                 % before the first frame
            tc.verifyEqual(m.block(5), 4:5);                 % decode from the previous keyframe
            tc.verifyEqual(m.decodeStart(7), 6);
            r0 = m.Revision; m.remap('Offset', 20);
            tc.verifyEqual(m.Revision, r0 + 1);
            tc.verifyEqual(m.frameAt(21), 2);
            tc.verifyError(@() OceanAgents.Time.MediaTimeMap([1 1 2]), 'OceanAgents:Time:BadValue');
        end

        function columnBatchEqualsEagerLoopBitwise(tc)
            rs = RandStream('mrg32k3a', 'Seed', 1);
            v = rand(rs, 200, 1);
            cs = OceanAgents.Time.ColumnStore();
            for k = 1:200, cs.add(k, v(k)); end
            for k = 1:3:200, cs.setActive(k, false); end
            cs.remove(2);
            eager = v;
            for k = 1:200
                if mod(k - 1, 3) ~= 0 && k ~= 2, eager(k) = eager(k) * 1.01 + 0.5; end
            end
            n = cs.applyBatch(@(x) x * 1.01 + 0.5);
            got = nan(200, 1);
            for k = 1:200
                if k ~= 2, got(k) = cs.value(k); end
            end
            mask = true(200, 1); mask(2) = false;
            tc.verifyTrue(isequal(got(mask), eager(mask)));   % bitwise
            tc.verifyEqual(n, nnz(mod((1:200)' - 1, 3) ~= 0) - 1);
            row = cs.add(500, 1);                              % freed row is reused
            tc.verifyEqual(row, 2);
        end
    end
end
