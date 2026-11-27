classdef EmbodiedTrustTest < OceanTest.BaseTestCase
    % EMBODIEDTRUSTTEST  Regression tests for the AB1 critic findings: senders cannot be forged, verify cannot be
    % skipped, memories cannot be written around the writer check, refusals change nothing (component, body, memory),
    % a committed behavior cannot be replaced, the learning gate holds (host-only, committed steps only), a learner
    % must update the table its mind acts on, adding a learner is all-or-nothing, and a body belongs to one mind.

    methods (Test)
        function sendersCannotBeForged(tc)
            T = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', ["left" "right"], 'Role', "command");
            tc.verifyError(@() OceanAgents.Minds.Signals.Signal(T, "left", 1, "mind", "motor", 1), 'MATLAB:TooManyInputs');
            s = OceanAgents.Minds.Signals.Signal.make(T, "left");
            tc.verifyError(@() setSender(s), 'MATLAB:class:SetProhibited');
            ep = OceanAgents.Memory.EpisodeMemory("episodes");
            ep.bindWriter("mind");
            W = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            rec = struct('Episode', 1, 'Observation', 1, 'Action', "left", 'Reward', 0, 'Terminated', false, 'Truncated', false);
            tc.verifyError(@() ep.receive(OceanAgents.Minds.Signals.Signal.make(W, rec)), 'OceanAgents:Memory:NotWriter');
            tc.verifyError(@() ep.store(rec), 'MATLAB:class:MethodRestricted');           % no way around the check
            function setSender(x), x.Sender = "mind"; end
        end

        function aReceiverCannotForgeItsReplySender(tc)
            r = OceanAgents.Minds.Signals.Router();
            p = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("mind"), "mind");
            r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("motor"), "motor");
            r.addRoute("motor", "motor", "command");
            T = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', "left", 'Role', "command");
            replies = p.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "left"));
            tc.verifyEqual(replies.Sender, "motor");                      % always restamped by the router
        end

        function verifyCannotBeSkipped(tc)
            c = OceanTest.OceanAgents.Fixtures.Signals.CounterComponent();
            T = c.Accepts;
            tc.verifyError(@() c.act(OceanAgents.Minds.Signals.Signal.make(T, "forbidden")), 'MATLAB:class:MethodRestricted');
            tc.verifyError(@() c.verify(OceanAgents.Minds.Signals.Signal.make(T, "left")), 'MATLAB:class:MethodRestricted');
            tc.verifyTrue(c.receive(OceanAgents.Minds.Signals.Signal.make(T, "forbidden")).isRefusal());
            tc.verifyEqual(c.Count, 0);
        end

        function memoryRefusalsChangeNothing(tc)
            q = OceanStats.RL.TD.Values.TabularQ(true(3, 2), 0);
            mem = OceanAgents.Memory.TableMemory("values", q, ["a" "b"]);
            before = mem.values();
            Q = OceanAgents.Minds.Signals.SignalType.query();
            tc.verifyTrue(mem.receive(OceanAgents.Minds.Signals.Signal.make(Q, struct('Op', "set", 'State', 1))).isRefusal());
            tc.verifyEqual(mem.values(), before);
            tc.verifyFalse(any(strcmp(properties(mem), 'Table')), 'the live table is never handed out');
        end

        function bodyRefusalsChangeNothing(tc)
            moves = ["up" "right" "down" "left"];
            M = OceanAgents.Body.IBody.moveType(moves);
            B = OceanAgents.Body.GridRobotBody.behaviorType();
            S = @(t, v) OceanAgents.Minds.Signals.Signal.make(t, v);
            robot = OceanAgents.Body.GridRobotBody(moves);
            robot.command(S(B, struct('Behavior', "repeat", 'Move', "right", 'Times', 3, 'Interruptible', true)));
            r = robot.command(S(OceanAgents.Body.IBody.moveType("upleft"), "upleft"));   % refused diagonal
            tc.verifyTrue(r.isRefusal());
            tc.verifyNotEmpty(robot.Behavior, 'a refused move must not cancel the running behavior');
            tc.verifyEqual(robot.toAction(), "right");
            committed = OceanAgents.Body.GridRobotBody(moves);
            committed.command(S(B, struct('Behavior', "repeat", 'Move', "up", 'Times', 2, 'Interruptible', false)));
            r = committed.command(S(B, struct('Behavior', "repeat", 'Move', "left", 'Times', 1, 'Interruptible', true)));
            tc.verifyTrue(r.isRefusal());                                  % busy: cannot replace a committed behavior
            tc.verifyEqual([committed.toAction(), committed.toAction()], ["up" "up"]);
            committed.command(S(M, "left"));
            tc.verifyEqual(committed.toAction(), "left");
        end

        function theLearningGateHolds(tc)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            codec = OceanAgents.Artifacts.DiscreteStateCodec(48);
            q = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
            OceanAgents.Embodied.Dualism.form(mind, OceanTest.OceanAgents.Fixtures.Signals.FailingBody(string(1:4), 30));
            learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec);
            mind.addLearner(learner, "q");
            host = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1);
            tc.verifyError(@() mind.learn([]), 'MATLAB:class:MethodRestricted');           % host-only entry
            rec = OceanAgents.Runtime.SingleAgentRunner(OceanAgents.Env.NativeSingleAgentBridge( ...
                OceanStats.RL.Env.Wrappers.TimeLimit(env, 200)), host, 'Seed', 7).run(5);
            tc.verifyEqual(rec.Status, "Failed");
            tc.verifyEqual(rec.StepCalls, 29);
            tc.verifyEqual(learner.NumUpdates, 29, 'the step that failed is never learned');
        end

        function aLearnerMustUpdateTheTableItsMindActsOn(tc)
            q = OceanStats.RL.TD.Values.TabularQ(true(4, 2), 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:2);
            other = OceanAgents.Learners.OneStepQLearner(OceanStats.RL.TD.Values.TabularQ(true(4, 2), 0), 1:2, 'Alpha', 0.5);
            tc.verifyError(@() mind.addLearner(other, "other"), 'OceanAgents:Memory:NotWriter');
            tc.verifyEqual(mind.numLearners(), 0);
            mine = OceanAgents.Learners.OneStepQLearner(q, 1:2, 'Alpha', 0.5);
            mind.addLearner(mine, "q");
            tc.verifyError(@() mind.addLearner(mine, "again"), 'OceanAgents:Memory:WriterBound');
            tc.verifyEqual(mind.numLearners(), 1, 'a refused addLearner changes nothing');
        end

        function aBodyBelongsToOneMind(tc)
            body = OceanAgents.Body.EmptyBody(["1" "2"]);
            a = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            b = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            OceanAgents.Embodied.Dualism.form(a, body);
            tc.verifyError(@() OceanAgents.Embodied.Dualism.form(b, body), 'OceanAgents:Embodied:Incompatible');
            OceanAgents.Embodied.Dualism.form(a, OceanAgents.Body.EmptyBody(["1" "2"]));    % a moves on ...
            OceanAgents.Embodied.Dualism.form(b, body);                                     % ... the body is free
            tc.verifyTrue(body.isPaired());
        end

        function classesAtTheTrustBoundaryAreSealed(tc)
            for c = ["Signal" "SignalType" "Wave" "Router" "Port"]
                tc.verifyTrue(meta.class.fromName("OceanAgents.Minds.Signals." + c).Sealed, c);
            end
            tc.verifyTrue(meta.class.fromName("OceanAgents.Body.BodySignature").Sealed);
        end

        function writerIdentityIsTiedToOneRouter(tc)
            W = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            rec = struct('Episode', 1, 'Observation', 1, 'Action', "a", 'Reward', 0, 'Terminated', false, 'Truncated', false);
            ep = OceanAgents.Memory.EpisodeMemory("episodes");
            ep.bindWriter("learner");
            r1 = OceanAgents.Minds.Signals.Router();
            r1.attach(ep, "episodes");
            % a second router (another agent) cannot reach this memory at all ...
            r2 = OceanAgents.Minds.Signals.Router();
            tc.verifyError(@() r2.attach(ep, "episodes"), 'OceanAgents:Memory:WriterBound');
            % ... and a converter cannot take the writer's name on the memory's own router
            r1.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("learner"), "learner");
            tc.verifyError(@() r1.registerConverter(OceanAgents.Components.Converters.ThresholdConverter(1, 0), "learner"), ...
                'OceanAgents:Signals:BadType');
            lp = r1.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("writer2"), "writer2");
            r1.addRoute("w", "episodes", "command");
            tc.verifyError(@() lp.send("w", OceanAgents.Minds.Signals.Signal.make(W, rec)), 'OceanAgents:Memory:NotWriter');
            tc.verifyEqual(ep.receive(OceanAgents.Minds.Signals.Signal.make( ...
                OceanAgents.Minds.Signals.SignalType.query(), struct('Op', "count"))).Value.Count, 0);
        end

        function aRefusedJoinChangesNothing(tc)
            a = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            b = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            bodyA = OceanAgents.Body.GridRobotBody(["1" "2"]);
            bodyB = OceanAgents.Body.EmptyBody(["1" "2"]);
            OceanAgents.Embodied.Dualism.form(a, bodyA);
            OceanAgents.Embodied.Dualism.form(b, bodyB);
            routerBefore = b.Router;
            tc.verifyError(@() OceanAgents.Embodied.Dualism.form(b, bodyA), 'OceanAgents:Embodied:Incompatible');
            tc.verifyTrue(b.Router == routerBefore, 'the refused join left the old router in place');
            tc.verifyTrue(b.Body == bodyB);
            view = OceanAgents.Core.ActorView('AgentID', "b", 'EpisodeID', 1, 'FrameID', 1, 'Tick', 0, ...
                'Observation', 1, 'ActionLabels', 1:2, 'ActionMask', true(1, 2));
            b.beginEpisode(view);
            d = b.decide(view, RandStream('mrg32k3a', 'Seed', 1));
            tc.verifyTrue(any(d.Action == [1 2]));
            tc.verifyEqual(bodyA.Motor.Count, 0, 'the refused join never let b drive a''s body');
            b.endEpisode();
            OceanAgents.Embodied.Dualism.form(b, OceanAgents.Body.GridRobotBody(["1" "2"]));   % b may still swap
            tc.verifyClass(b.Body, 'OceanAgents.Body.GridRobotBody');
            reply = b.query("values", struct('Op', "row", 'State', 1));                     % its memory re-joined
            tc.verifyEqual(reply.Labels, ["1" "2"]);
        end

        function ownershipCannotBeFaked(tc)
            % critic AB1 round 3: the owner of a router is the mind object itself, set only by that mind
            tc.verifyError(@() OceanAgents.Minds.Signals.Router('Owner', 1), ?MException);
            r = OceanAgents.Minds.Signals.Router();
            m = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            tc.verifyError(@() r.setOwner_(m), 'MATLAB:class:MethodRestricted');
            % another mind's memory can be neither added nor reached
            c = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2, 'MemoryID', "valuesC");
            d = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            OceanAgents.Embodied.Dualism.form(c, OceanAgents.Body.EmptyBody(["1" "2"]));
            tc.verifyError(@() d.addMemory(c.Values), 'OceanAgents:Memory:WriterBound');
            tc.verifyError(@() r.attach(c.Values, "stolen"), 'OceanAgents:Memory:WriterBound');
            tc.verifyTrue(c.Values.isOwned());
        end

        function aReplayedWriteIsRefused(tc)
            W = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            rec = struct('Episode', 1, 'Observation', 1, 'Action', "a", 'Reward', 0, 'Terminated', false, 'Truncated', false);
            ep = OceanAgents.Memory.EpisodeMemory("ep");
            ep.bindWriter("learner");
            r = OceanAgents.Minds.Signals.Router();
            writer = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("learner"), "learner");
            spy = OceanTest.OceanAgents.Fixtures.Signals.RecorderComponent("spy");
            r.attach(spy, "spy");
            r.attach(ep, "ep");
            r.addRoute("toSpy", "spy", "command");
            r.addRoute("toEp", "ep", "command");
            writer.send("toSpy", OceanAgents.Minds.Signals.Signal.make(W, rec));   % the writer's genuine stamped signal ...
            tc.verifyEqual(spy.Last.Sender, "learner");
            tc.verifyError(@() ep.receive(spy.Last), 'OceanAgents:Memory:NotWriter');   % ... replayed to the memory
            writer.send("toEp", OceanAgents.Minds.Signals.Signal.make(W, rec));    % the real write still works
            n = ep.receive(OceanAgents.Minds.Signals.Signal.make(OceanAgents.Minds.Signals.SignalType.query(), ...
                struct('Op', "count")));
            tc.verifyEqual(n.Value.Count, 1);
        end

        function aPlainRouterDoesNotLockAMemoryOutOfAMind(tc)
            ep = OceanAgents.Memory.EpisodeMemory("ep");
            harness = OceanAgents.Minds.Signals.Router();
            harness.attach(ep, "ep");                                             % unowned: a plain router may hold it
            m = OceanAgents.Minds.QSignalMind(OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), 1:2);
            m.addMemory(ep);                                                      % a mind may still claim it
            OceanAgents.Embodied.Dualism.form(m, OceanAgents.Body.EmptyBody(["1" "2"]));
            tc.verifyEqual(m.query("ep", struct('Op', "count")).Count, 0);
            tc.verifyError(@() harness.attach(OceanAgents.Memory.EpisodeMemory("x"), "ep"), 'OceanAgents:Signals:BadType');
        end

        function malformedBehaviorRecordsAreRefused(tc)
            B = OceanAgents.Body.GridRobotBody.behaviorType();
            robot = OceanAgents.Body.GridRobotBody(["up" "down"]);
            S = @(v) OceanAgents.Minds.Signals.Signal.make(B, v);
            tc.verifyTrue(robot.command(S(struct('Behavior', "repeat", 'Move', "up", 'Times', Inf))).isRefusal());
            tc.verifyTrue(robot.command(S(struct('Behavior', "repeat", 'Move', "up", 'Times', 2, ...
                'Interruptible', "zz"))).isRefusal());
            tc.verifyEmpty(robot.Behavior);
        end
    end
end
