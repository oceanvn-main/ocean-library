classdef KernelIntegrationTest < OceanTest.BaseTestCase
    % KERNELINTEGRATIONTEST  TIME_Kernel_Integration_Builder_Plan.md: kernel-backed domains (one authority, committed
    % capture, capabilities, action ownership) and active registration (inactive entities frozen, zero visits,
    % refused interactions, declared deadline policies, no retroactive work, active-set revisions).

    methods (Static, Access = private)
        function L = L()
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
        end

        function A = A()
            A = OceanTest.OceanAgents.Fixtures.World.Agent.LineDoors;
        end

        function w = batteryWorld(skip)
            % three batteries (entities 1..3): Charge 10 at rates -1, -0.5, -0.25 -> empty at 10, 20, 40
            L = OceanTest.OceanAgents.Time.KernelIntegrationTest.L();
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -1)), ...
                L.offer('Charge', struct('Value', 10, 'Rate', -0.5)), L.offer('Charge', struct('Value', 10, 'Rate', -0.25))});
            w = OceanAgents.World.World(L.battery('Init', init), 'SkipIdle', skip);
            w.reset(1);
        end

        function w = moverWorld(extra)
            % entity 1 = mover (Position [0 0], Velocity [1 0]); entities 2, 3 = batteries (empty at 10 and 20)
            L = OceanTest.OceanAgents.Time.KernelIntegrationTest.L();
            mi = @(s) L.initSpawns({L.offer('Position', [0 0], 'Velocity', [1 0])});
            bi = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -1)), ...
                L.offer('Charge', struct('Value', 10, 'Rate', -0.5))});
            doors = [L.mover('Init', mi), L.battery('Init', bi)];
            if nargin > 0, doors = [doors extra]; end
            w = OceanAgents.World.World(doors, 'SkipIdle', false);
            w.reset(1);
        end

        function d = reaper(fn)
            d = OceanAgents.World.DoorSpec('Name', "reaper", 'Resources', struct('R', 0), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Reap", 'Phase', "settle", 'Commands', "remove", 'Fn', fn), ...
                'Drive', "every-tick");
        end

        function aw = agentWorld()
            A = OceanTest.OceanAgents.Time.KernelIntegrationTest.A(); L = OceanTest.OceanAgents.Time.KernelIntegrationTest.L();
            bi = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -1))});
            doors = [A.doors(["a1" "a2"], 'Goal', 3), L.battery('Init', bi)];
            aw = OceanAgents.World.AgentWorld(doors, A.cfg(["a1" "a2"]), 'Name', "line");
        end

        function b = batch(frame, actions)
            ds = OceanAgents.Core.Decision.empty(1, 0);
            for k = find(frame.Requested)
                ds(end + 1) = OceanAgents.Core.Decision('AgentID', frame.Roster(k), 'EpisodeID', frame.EpisodeID, ...
                    'FrameID', frame.FrameID, 'Action', actions(k)); %#ok<AGROW>
            end
            b = OceanAgents.Core.ActionBatch('EpisodeID', frame.EpisodeID, 'FrameID', frame.FrameID, 'Decisions', ds);
        end

        function ran = rulesRun(reports)
            ran = strings(1, 0);
            for r = reports
                for p = r.Phases, ran = [ran p.RulesRun]; end %#ok<AGROW>
            end
        end
    end

    methods (Test)
        % ================= active registration in the kernel =================
        function fullRegistrationKeepsBehavior(tc)
            a = tc.moverWorld(); b = tc.moverWorld();
            reg = OceanAgents.Time.ActiveSet();
            rep = reg.apply(b, b.liveEntityIDs()');                 % everyone active: a no-op
            tc.verifyEmpty(rep.Deactivated);
            tc.verifyEqual(rep.Equivalence, "full");
            ra = a.run(15); rb = b.run(15);
            tc.verifyEqual(rulesRun_(ra), rulesRun_(rb));
            tc.verifyEqual(a.Store.snapshot(), b.Store.snapshot());
            tc.verifyEqual(a.Calendar.snapshot(), b.Calendar.snapshot());
            tc.verifyEqual(b.ActiveRevision, 0);
        end

        function inactiveParticipantsHoldValuesAndCostNothing(tc)
            w = tc.moverWorld();
            w.deactivate([1; 3]);
            reports = w.run(12);
            ran = tc.rulesRun(reports);
            tc.verifyFalse(any(ran == "mover/Move"), 'an inactive mover receives no rule run');
            tc.verifyEqual(w.Store.valueAt("Position", 1, w.Now), [0 0]);
            tc.verifyEqual(w.Store.valueAt("Charge", 3, w.Now), 10, 'formula value held while the world advanced');
            tc.verifyFalse(w.Registry.isAlive(2), 'the active battery still drained at its own time');
            tc.verifyTrue(w.Registry.isAlive(3));
            it = w.Calendar.peek();
            tc.verifyEmpty(it(ismember([it.EntityID], [1 3])), 'no calendar work for inactive entities');
        end

        function unregisteringOnePredictorPreservesOthers(tc)
            w = tc.batteryWorld(true);
            before = w.Calendar.peek();
            w.deactivate(2);
            after = w.Calendar.peek();
            tc.verifyEqual(sort([after([after.Kind] == "prediction").EntityID]), [1 3]);
            tc.verifyEqual(after([after.EntityID] == 1 & [after.Kind] == "prediction").Time, 10);
            tc.verifyEqual(after([after.EntityID] == 3 & [after.Kind] == "prediction").Time, 40);
            tc.verifyGreaterThan(numel(before), numel(after));
        end

        function reactivationSchedulesNoRetroactiveWork(tc)
            for skip = [true false]
                w = tc.batteryWorld(skip);
                w.deactivate(2);                         % frozen at 0: would have emptied at 20
                w.run(8);
                tc.verifyEqual(w.Store.valueAt("Charge", 2, 8), 10);
                w.activate(2);
                it = w.Calendar.peek();
                e2 = it([it.EntityID] == 2 & [it.Kind] == "prediction");
                tc.verifyEqual(e2.Time, 28, 'time functions resume from the held value: empty 20 ticks after activation');
                tc.verifyGreaterThan(e2.Time, w.Now);
                if skip, tc.verifyLessThanOrEqual(w.NextTime, 28); end
                w.run(27);
                tc.verifyTrue(w.Registry.isAlive(2));
                w.run(28);
                tc.verifyFalse(w.Registry.isAlive(2), 'removed exactly at 28');
                tc.verifyEqual(w.ActiveRevision, 2);
            end
        end

        function interactionWithInactiveIsRefusedBeforeMutation(tc)
            fn = @(ctx) OceanTest.OceanAgents.Time.KernelIntegrationTest.reapAt(ctx, 3, 2);
            w = tc.moverWorld(tc.reaper(fn));
            w.deactivate(3);
            snap = [];
            for k = 1:3
                if k == 3, snap = w.Store.snapshot(); end
                r = w.advance();
            end
            refused = [r.Phases.CommandsRefused];
            tc.verifyEqual(numel(refused), 1);
            tc.verifyEqual(refused.Reason, "inactive");
            tc.verifyTrue(w.Registry.isAlive(3));
            now3 = w.Store.snapshot();
            tc.verifyEqual(now3.Components.Charge.Base(3), snap.Components.Charge.Base(3));
        end

        function setAsideDeadlinesNeedADeclaredPolicy(tc)
            % entity 1 (mover) has a scheduled removal at 15; it is inactive from 2 to 8
            fn = @(ctx) OceanTest.OceanAgents.Time.KernelIntegrationTest.scheduleRemove(ctx, 1, 15);
            expected = struct('shift', 21, 'refuse', 15, 'drop', NaN);
            for pol = ["shift" "refuse" "drop" "undeclared"]
                w = tc.moverWorld(tc.reaper(fn));
                w.run(2);
                w.deactivate(1);
                w.run(8);
                if pol == "undeclared"
                    tc.verifyError(@() w.activate(1), 'OceanAgents:World:UndeclaredPolicy');
                    tc.verifyEqual(w.inactiveEntityIDs(), 1, 'refused before any change');
                    continue
                end
                w.activate(1, 'Deadlines', pol);
                it = w.Calendar.peek();
                cmd = it([it.Kind] == "command" & [it.EntityID] == 1);
                if pol == "drop"
                    tc.verifyEmpty(cmd);
                else
                    tc.verifyEqual(cmd.Time, expected.(char(pol)), char(pol));
                end
            end
            w = tc.moverWorld(tc.reaper(fn));
            w.run(2); w.deactivate(1); w.run(20);
            tc.verifyError(@() w.activate(1, 'Deadlines', "refuse"), 'OceanAgents:World:OverdueWork');
            tc.verifyEqual(w.inactiveEntityIDs(), 1);
        end

        function resetClearsTheInactiveSet(tc)
            w = tc.batteryWorld(true);
            w.deactivate([1 2]);
            w.reset(1);
            tc.verifyEmpty(w.inactiveEntityIDs());
            tc.verifyEqual(w.ActiveRevision, 2);
            ids = w.liveEntityIDs();                      % IDs continue across episodes
            tc.verifyError(@() w.deactivate(99), 'OceanAgents:World:BadValue');
            w.deactivate(ids(1));
            tc.verifyError(@() w.deactivate(ids(1)), 'OceanAgents:World:BadValue');
            tc.verifyError(@() w.activate(ids(2)), 'OceanAgents:World:BadValue');
        end

        % ================= ActiveSet registry =================
        function registryExpandsAndExplains(tc)
            w = tc.batteryWorld(true);
            reg = OceanAgents.Time.ActiveSet();
            reg.declare(1, 'Owner', "m1", 'DependsOn', 3, 'Binding', "formula");
            reg.declare(2, 'Owner', "m2", 'Resources', "charger");
            reg.declare(3, 'Owner', "m3", 'Resources', "charger");
            p = reg.plan(w, 2);
            tc.verifyEqual(p.Active, [2 3]);
            tc.verifyEqual(p.ExpandedBy.ID, 3);
            tc.verifyMatches(p.ExpandedBy.Reason, "shares resource");
            tc.verifyMatches(p.Equivalence, "selective-registration");
            tc.verifyEqual(sort(p.Frozen), 1);
            p = reg.plan(w, 1);                                   % 1 -> 3 -> (charger) 2
            tc.verifyEqual(p.Active, [1 2 3]);
            tc.verifyEqual(p.Equivalence, "full");
            tc.verifyEqual(w.inactiveEntityIDs(), zeros(0, 1), 'plan changes nothing');
            tc.verifyError(@() reg.apply(w, 2, 'Expand', false), 'OceanAgents:Time:DependencyNotActive');
            tc.verifyEmpty(w.inactiveEntityIDs());
            rep = reg.apply(w, 2);
            tc.verifyEqual(rep.Deactivated, 1);
            tc.verifyEqual(w.inactiveEntityIDs(), 1);
            tc.verifyEqual(reg.Revision, w.ActiveRevision);
            tc.verifyError(@() reg.declare(2, 'Owner', "other"), 'OceanAgents:Time:DuplicateOwnership');
            tc.verifyError(@() reg.plan(w, 99), 'OceanAgents:Time:UnknownEntity');
            rep = reg.apply(w, [1 2 3]);
            tc.verifyEqual(rep.Activated, 1);
        end

        % ================= kernel-backed domain: bare world =================
        function bareWorldThroughDomainEqualsDirectRun(tc)
            direct = tc.moverWorld();
            via = tc.moverWorld();
            dom = OceanAgents.Time.KernelDomain(via, 'DomainID', "bare");
            rd = direct.run(25);
            dom.advanceUntil(25);
            tc.verifyEqual(via.Store.snapshot(), direct.Store.snapshot());
            tc.verifyEqual(via.Calendar.snapshot(), direct.Calendar.snapshot());
            tc.verifyEqual(dom.Now, direct.Now, 'time is the kernel clock');
            tc.verifyEqual(dom.Visits, numel(rd));
            tc.verifyEqual(dom.Log.count(), numel(rd), 'one commit per kernel instant, none at an arbitrary target');
            dom.advanceUntil(25);
            tc.verifyEqual(dom.Log.count(), numel(rd), 'nothing more to process: no synthetic commit');
            tc.verifyError(@() dom.advanceUntil(3), 'OceanAgents:Time:Retroactive');
        end

        function capabilitiesAndUnsupportedOperationsRefuse(tc)
            dom = OceanAgents.Time.KernelDomain(tc.moverWorld());
            c = dom.capabilities();
            tc.verifyEqual([c.Phased c.Autonomous c.Restore c.MessageInjection c.ActiveRegistration], ...
                [true true false false true]);
            for f = {@() dom.handler("x", @(~, ~) 0), @() dom.schedule(5, "x"), @() dom.cancelOwner("o"), ...
                    @() dom.exportState(), @() dom.importState(struct()), @() dom.inject(1, "x")}
                tc.verifyError(f{1}, 'OceanAgents:Time:Unsupported');
            end
            tc.verifyEqual(c.Backend, "kernel");
        end

        % ================= kernel-backed domain: AgentWorld =================
        function agentWorldCommitsHaveDistinctIdentities(tc)
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos", 'DomainID', "line");
            f = aw.reset(3);
            tc.verifyEqual(dom.Log.count(), 1);
            f = aw.step(tc.batch(f, [2 2]));
            f = aw.step(tc.batch(f, [2 2]));
            tc.verifyEqual(dom.Log.count(), 3);
            r0 = dom.Log.at(0, "before"); r1 = dom.Log.at(0, "after");
            tc.verifyEqual(r0.Status, "unavailable");
            tc.verifyEqual(r1.Context.Commit, 2, 'reset (commit 1) and the first step (commit 2) share tick 0');
            tc.verifyEqual(dom.Log.at(0, "after").Context.Time, 0);
            last = dom.Log.at(1);
            tc.verifyEqual(last.Context.Commit, 3);
            tc.verifyEqual(last.Context.Time, 1);
            tc.verifyEqual(last.State.Pos, aw.researchView("Pos").Value, 'history equals the committed frame');
            tc.verifyEqual(last.State.Ended, aw.researchView("Pos").Ended);
            tc.verifyFalse(dom.HistoryGap);
            tc.verifyEqual(dom.context().Commit, 3);
            tc.verifyEqual(f.Tick, dom.Now);
        end

        function failedStepPublishesNothing(tc)
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos");
            f = aw.reset(3);
            good = tc.batch(f, [2 2]);
            bad = OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID + 7, 'Decisions', good.Decisions);
            tc.verifyError(@() aw.step(bad), ?MException);
            tc.verifyEqual(dom.Log.count(), 1);
            tc.verifyEqual(dom.Commit, 1);
            aw.step(good);
            tc.verifyEqual(dom.Log.count(), 2);
        end

        function newEpisodeKeepsHistoryPerEpisode(tc)
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos");
            f = aw.reset(3); f = aw.step(tc.batch(f, [2 2])); aw.step(tc.batch(f, [2 2]));   %#ok<NASGU>
            f = aw.reset(3); aw.step(tc.batch(f, [1 1]));
            tc.verifyEqual(dom.Log.count(), 5);
            tc.verifyEqual(dom.Log.coverage(), [0 0], 'default reads the latest episode');
            tc.verifyEqual(dom.Log.coverage(1), [0 1]);
            tc.verifyEqual(dom.Log.at(1, "after", 1).State.Pos, [2; 2]);
            tc.verifyEqual(dom.Log.at(0, "after", 2).Context.Episode, 2);
        end

        function actionDrivenDomainCannotBeAdvancedAroundItsTransaction(tc)
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos", 'DomainID', "line");
            aw.reset(3);
            tc.verifyEqual(dom.capabilities().Autonomous, false);
            tc.verifyError(@() dom.advanceUntil(5), 'OceanAgents:Time:ActionDriven');
            tc.verifyEqual(aw.NumAdvances, 0, 'refused before anything moved');
            dom.pause();
            f = aw.reset(3);
            tc.verifyError(@() dom.step(tc.batch(f, [2 2])), 'OceanAgents:Time:DomainPaused');
            tc.verifyEqual(aw.NumAdvances, 0);
            dom.resume();
            dom.step(tc.batch(f, [2 2]));
            tc.verifyEqual(aw.NumAdvances, 1);
        end

        function viewingDoesNotChangeOutcomes(tc)
            plain = tc.agentWorld(); watched = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(watched, 'Components', "Pos", 'DomainID', "line");
            fc = OceanAgents.Time.FrameCoordinator(); fc.addSource("line", dom.Log); fc.addView("scene", "line", "Pos");
            cur = OceanAgents.Time.ViewCursor("line");
            fp = plain.reset(5); fw = watched.reset(5);
            rng = RandStream('mrg32k3a', 'Seed', 2);
            for s = 1:6
                a = randi(rng, 2, 1, 2);
                fp = plain.step(tc.batch(fp, a)); fw = watched.step(tc.batch(fw, a));
                cur.seek(randi(rng, 3) - 1); fc.frame(cur); cur.seek(0); fc.frame(cur);   % repeated seeks
                tc.verifyEqual(fw.Rewards, fp.Rewards);
                tc.verifyEqual(fw.EpisodeEnded, fp.EpisodeEnded);
                tc.verifyEqual(fw.Tick, fp.Tick);
                if fp.EpisodeEnded, break, end
            end
            tc.verifyEqual(watched.researchView("Pos").Value, plain.researchView("Pos").Value);
            tc.verifyEqual(watched.NumAdvances, plain.NumAdvances);
        end

        function frameCoordinatorReadsKernelHistory(tc)
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos", 'DomainID', "line");
            f = aw.reset(3);
            for s = 1:2, f = aw.step(tc.batch(f, [2 1])); end
            fc = OceanAgents.Time.FrameCoordinator(); fc.addSource("line", dom.Log); fc.addView("scene", "line", "Pos");
            cur = OceanAgents.Time.ViewCursor("line"); cur.seek(0);
            b = fc.frame(cur);
            tc.verifyEqual(b.Contexts{1}.Commit, 2);        % tick 0 holds the reset and first step; latest commit at 0
            tc.verifyEqual(b.Values.line.Pos, dom.Log.at(0).State.Pos);
            cur.seek(1);
            b = fc.frame(cur);
            tc.verifyEqual(b.Values.line.Pos, dom.Log.at(1).State.Pos);
            tc.verifyEqual(b.Contexts{1}.ActiveSetRevision, 0);
        end

        function activeRegistrationThroughAgentWorld(tc)
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos");
            f = aw.reset(3);
            roster = aw.entityIDs();
            tc.verifyError(@() aw.deactivate(roster(1)), 'OceanAgents:World:RosterEntity');
            others = setdiff(aw.liveEntityIDs(), roster);
            tc.verifyEqual(numel(others), 1);
            reg = OceanAgents.Time.ActiveSet();
            rep = reg.apply(dom, roster);                   % battery goes inactive; agents are pinned active
            tc.verifyEqual(rep.Deactivated, others);
            f = aw.step(tc.batch(f, [2 2]));
            tc.verifyEqual(dom.context().ActiveSetRevision, 1);
            tc.verifyEqual(dom.Log.at(0).Context.ActiveSetRevision, 1);
            tc.verifyMatches(rep.Equivalence, "selective-registration");
            reg.apply(dom, [roster others]);
            tc.verifyEmpty(dom.inactiveEntityIDs());
        end

        function recordingFailureNeverBreaksTheWorld(tc)
            log = OceanTest.OceanAgents.Time.FailingCommitLog();
            log.FailFrom = 2;
            aw = tc.agentWorld();
            dom = OceanAgents.Time.KernelDomain(aw, 'Components', "Pos", 'Log', log);
            f = aw.reset(3);
            tc.verifyFalse(dom.HistoryGap);
            f = aw.step(tc.batch(f, [2 2]));                 % the world commits; recording fails
            tc.verifyTrue(dom.HistoryGap);
            tc.verifyEqual(dom.PublishError.identifier, 'Test:DiskFull');
            tc.verifyEqual(aw.NumAdvances, 1);
            tc.verifyEqual(log.count(), 1, 'the gap is stated, not silently filled');
            aw.step(tc.batch(f, [2 2]));
            tc.verifyEqual(aw.NumAdvances, 2);
        end

    end

    methods (Static)
        function out = reapAt(ctx, target, at)
            out = struct();
            if ctx.Now == at
                out = struct('Commands', OceanAgents.World.Command.remove(target));
            end
        end

        function out = scheduleRemove(ctx, target, at)
            out = struct();
            if ctx.Now == 2 || (ctx.Now == 0 && false)
                out = struct('Commands', OceanAgents.World.Command.remove(target, 'At', at));
            end
        end
    end
end

function ran = rulesRun_(reports)
ran = strings(1, 0);
for r = reports
    for p = r.Phases, ran = [ran p.RulesRun]; end %#ok<AGROW>
end
end
