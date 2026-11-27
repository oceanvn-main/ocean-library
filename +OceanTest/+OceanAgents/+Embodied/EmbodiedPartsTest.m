classdef EmbodiedPartsTest < OceanTest.BaseTestCase
    % EMBODIEDPARTSTEST  AB1 sections 5, 6 and 8: memory (one writer, queries, label-keyed transfer), body (probe,
    % forwarding, motor refusal, behaviors start / stop / committed), signature diff and the Dualism pairing rules.

    methods (Static)
        function S = sig(type, v)
            S = OceanAgents.Minds.Signals.Signal.make(type, v);
        end
    end

    methods (Test)
        function memoryHasOneWriterAndAnswersQueries(tc)
            valid = true(3, 2);  valid(3, 2) = false;
            q = OceanStats.RL.TD.Values.TabularQ(valid, 0);
            T = q.Q;  T(1, :) = [1 5];  T(2, :) = [7 -1];  q.setTables(T, zeros(size(T)));
            mem = OceanAgents.Memory.TableMemory("values", q, ["left" "right"]);
            Q = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "query");
            a = mem.receive(tc.sig(Q, struct('Op', "best", 'State', 1)));
            tc.verifyEqual([a.Value.Label, string(a.Value.Value)], ["right" "5"]);
            a = mem.receive(tc.sig(Q, struct('Op', "value", 'State', 2, 'Label', "left")));
            tc.verifyEqual(a.Value.Value, 7);
            tc.verifyTrue(mem.receive(tc.sig(Q, struct('Op', "value", 'State', 9, 'Label', "left"))).isRefusal());
            tc.verifyTrue(mem.receive(tc.sig(Q, struct('Op', "teleport", 'State', 1))).isRefusal());
            other = OceanStats.RL.TD.Values.TabularQ(valid, 0);
            stranger = OceanAgents.Learners.OneStepQLearner(other, 1:2, 'Alpha', 0.5);
            tc.verifyError(@() mem.bindLearner(stranger, "stranger"), 'OceanAgents:Memory:NotWriter');
            mem.bindLearner(OceanAgents.Learners.OneStepQLearner(q, 1:2, 'Alpha', 0.5), "learner");
            tc.verifyError(@() mem.bindWriter("second"), 'OceanAgents:Memory:WriterBound');
            % a write signal from anyone but the bound writer is refused (routed, so the sender is trusted)
            ep = OceanAgents.Memory.EpisodeMemory("episodes");
            ep.bindWriter("mind");
            r = OceanAgents.Minds.Signals.Router();
            mindPort = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("mind"), "mind");
            spyPort = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("spy"), "spy");
            r.attach(ep, "episodes");
            r.addRoute("write", "episodes", "command");
            r.addRoute("ask", "episodes", "query");
            W = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            rec = struct('Episode', 1, 'Observation', 3, 'Action', "left", 'Reward', -1, 'Terminated', false, ...
                'Truncated', false);
            mindPort.send("write", tc.sig(W, rec));
            tc.verifyError(@() spyPort.send("write", tc.sig(W, rec)), 'OceanAgents:Memory:NotWriter');
            n = spyPort.send("ask", tc.sig(Q, struct('Op', "count")));
            tc.verifyEqual(n.Value.Count, 1);                            % anyone may read
        end

        function labelKeyedTransferAfterABodySwap(tc)
            q = OceanStats.RL.TD.Values.TabularQ(true(2, 4), 0);
            T = [1 2 3 4; 5 6 7 8];  q.setTables(T, zeros(2, 4));
            old = OceanAgents.Memory.TableMemory("values", q, ["up" "right" "down" "left"]);
            king = ["up" "right" "down" "left" "upright" "downright" "downleft" "upleft" "stay"];
            Q2 = OceanAgents.Memory.TableMemory.transfer(old, king, true(2, 9), -1);
            tc.verifyEqual(Q2.Q(:, 1:4), T);
            tc.verifyEqual(Q2.Q(:, 5:9), -ones(2, 5));
        end

        function bodiesProbeForwardAndRefuse(tc)
            moves = ["up" "right" "down" "left"];
            M = OceanAgents.Body.IBody.moveType(moves);
            empty = OceanAgents.Body.EmptyBody(moves);
            tc.verifyEqual(empty.probe().commandLabels(), moves);
            empty.command(tc.sig(M, "left"));
            tc.verifyEqual(empty.toAction(), "left");
            tc.verifyError(@() empty.toAction(), 'OceanAgents:Body:NoAction');            % nothing commanded
            robot = OceanAgents.Body.GridRobotBody(moves);
            tc.verifyEqual(robot.probe().Components, ["motor:OceanAgents.Components.Motor" ...
                "digits:OceanAgents.Body.Sensors.DigitObservation" ...
                "moves:OceanAgents.Body.Sensors.MovesObservation" "behavior:OceanAgents.Body.Sensors.BehaviorObservation"]);
            robot.command(tc.sig(M, "down"));
            tc.verifyEqual(robot.toAction(), "down");
            diag = OceanAgents.Body.IBody.moveType("upleft");
            r = robot.command(tc.sig(diag, "upleft"));
            tc.verifyTrue(r.isRefusal());                                    % a four-move motor has no diagonal
            tc.verifyEqual(robot.Motor.Count, 1);                            % the refusal changed nothing
            view = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Tick', 0, ...
                'Observation', 37, 'ActionLabels', 1:4, 'ActionMask', true(1, 4));
            s = robot.Sensors{1}.observe(view, robot);                       % its digit sensor reads the cell
            tc.verifyEqual(s.Value, 37);
        end

        function behaviorsStartOnlyBySignalAndCanBeCommitted(tc)
            moves = ["up" "right" "down" "left"];
            B = OceanAgents.Body.GridRobotBody.behaviorType();
            robot = OceanAgents.Body.GridRobotBody(moves);
            robot.command(tc.sig(B, struct('Behavior', "repeat", 'Move', "right", 'Times', 2, 'Interruptible', true)));
            tc.verifyEqual([robot.toAction(), robot.toAction()], ["right" "right"]);
            tc.verifyError(@() robot.toAction(), 'OceanAgents:Body:NoAction');           % done
            robot.command(tc.sig(B, struct('Behavior', "repeat", 'Move', "up", 'Times', 5, 'Interruptible', true)));
            robot.command(tc.sig(OceanAgents.Body.IBody.moveType(moves), "left"));       % interrupts
            tc.verifyEqual(robot.toAction(), "left");
            tc.verifyEmpty(robot.Behavior);
            robot.command(tc.sig(B, struct('Behavior', "repeat", 'Move', "up", 'Times', 3, 'Interruptible', false)));
            r = robot.command(tc.sig(OceanAgents.Body.IBody.moveType(moves), "left"));
            tc.verifyTrue(r.isRefusal());                                                % busy, committed
            stop = robot.command(tc.sig(B, struct('Behavior', "stop")));
            tc.verifyTrue(stop.isRefusal());
            tc.verifySubstring(char(stop.Value.Reason), 'NotInterruptible');
            tc.verifyEqual(robot.toAction(), "up");
            tc.verifyTrue(robot.command(tc.sig(B, struct('Behavior', "repeat", 'Move', "fly", 'Times', 1))).isRefusal());
        end

        function dualismPairsAndSwapsBetweenEpisodes(tc)
            four = ["1" "2" "3" "4"];
            king = [four "5" "6" "7" "8" "9"];
            q4 = OceanStats.RL.TD.Values.TabularQ(true(5, 4), 0);
            mind = OceanAgents.Minds.QSignalMind(q4, 1:4);
            d1 = OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.GridRobotBody(four));
            d2 = OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.GridRobotBody(king));   % the body may offer more
            delta = d1.Signature.diff(d2.Signature);
            tc.verifyEqual(delta.Added, ["5" "6" "7" "8" "9"]);
            tc.verifyEmpty(delta.Removed);
            tc.verifyTrue(mind.BodySignature.isequalTo(d2.Signature));                      % the mind feels the new body
            q9 = OceanStats.RL.TD.Values.TabularQ(true(5, 9), 0);
            kingMind = OceanAgents.Minds.QSignalMind(q9, 1:9);
            tc.verifyError(@() OceanAgents.Embodied.Dualism.form(kingMind, OceanAgents.Body.GridRobotBody(four)), ...
                'OceanAgents:Embodied:Incompatible');
            view = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Tick', 0, ...
                'Observation', 1, 'ActionLabels', 1:4, 'ActionMask', true(1, 4));
            mind.beginEpisode(view);
            tc.verifyError(@() OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(four)), ...
                'OceanAgents:Embodied:MidEpisode');
            mind.endEpisode();
            reply = mind.query("values", struct('Op', "row", 'State', 2));
            tc.verifyEqual(reply.Labels, four);
        end
    end
end
