classdef ObservationTest < OceanTest.BaseTestCase
    % OBSERVATIONTEST  AB2 (AB2_Observation_Contract.md): sensors and their types (O3, I12), the mind decides from its
    % observations only (O2, I10), a body without a needed sensor is refused (I11), and a running behavior is let
    % continue without the mind issuing anything (O4, I13).

    methods (Static)
        function v = view(obs, mask, labels)
            if nargin < 3, labels = 1:numel(mask); end
            v = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, 'Tick', 0, ...
                'Observation', obs, 'ActionLabels', labels, 'ActionMask', logical(mask));
        end
    end

    methods (Test)
        function sensorsReportTypedObservations(tc)
            body = OceanAgents.Body.EmptyBody(["a" "b" "c"]);
            v = tc.view(7, [1 0 1]);
            d = OceanAgents.Body.Sensors.DigitObservation().observe(v, body);
            tc.verifyEqual([d.Value, string(d.Type.ValueKind)], ["7" "number"]);
            m = OceanAgents.Body.Sensors.MovesObservation(["a" "b" "c"]).observe(v, body);
            tc.verifyEqual(m.Value, [true false true]);
            tc.verifyEqual(m.Type.Labels, ["a" "b" "c"]);
            tc.verifyError(@() OceanAgents.Body.Sensors.DigitObservation('Width', 2).observe(v, body), 'OceanAgents:Body:BadObservation');
            tc.verifyError(@() OceanAgents.Body.Sensors.MovesObservation(["a" "b"]).observe(v, body), 'OceanAgents:Body:BadObservation');
            st = OceanAgents.Body.Sensors.BehaviorObservation().observe(v, OceanAgents.Body.GridRobotBody(["a" "b"]));
            tc.verifyFalse(st.Value);
            tc.verifyEqual(st.Type.Labels, "running");
        end

        function visionIsAnImageOfItsDeclaredSize(tc)
            g = OceanAgents.Body.Sensors.GridRenderer(4, 12);
            s = OceanAgents.Body.Sensors.VisionObservation(g);
            img = s.observe(tc.view(37, true(1, 4)), []).Value;
            tc.verifyEqual(size(img), [4 12]);                                % 4 x 12 x 1
            tc.verifyEqual(find(img'), 37);                                   % row-major cell 37 is lit
            T = s.OutType;
            tc.verifyEqual(T.Size, [4 12 1]);
            tc.verifyError(@() OceanAgents.Minds.Signals.Signal.make(T, zeros(4, 11)), 'OceanAgents:Signals:BadValue');
            other = OceanAgents.Minds.Signals.SignalType('ValueKind', "array", 'Size', [5 12 1], 'Role', "sense");
            tc.verifyEqual(T.mismatch(other), "Size");
            ep = OceanAgents.Memory.EpisodeMemory("seen");                   % an image can be kept as experience
            ep.bindWriter("mind");
            r = OceanAgents.Minds.Signals.Router();
            p = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("mind"), "mind");
            r.attach(ep, "seen");
            r.addRoute("keep", "seen", "command");
            W = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            p.send("keep", OceanAgents.Minds.Signals.Signal.make(W, struct('Episode', 1, 'Observation', img, ...
                'Action', "1", 'Reward', 0, 'Terminated', false, 'Truncated', false)));
            got = ep.receive(OceanAgents.Minds.Signals.Signal.make(OceanAgents.Minds.Signals.SignalType.query(), ...
                struct('Op', "last", 'Count', 1)));
            tc.verifyEqual(got.Value.Records.Observation, img);
        end

        function theMindDecidesFromObservationsOnly(tc)
            % O2 / I10: the body's sensor reports fixed permitted moves; two very different world views give the
            % same decision with the same random stream
            labels = string(1:4);
            mind = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.RandomLegalPolicy(), 1:4);
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(labels, 'Sensors', ...
                {OceanTest.OceanAgents.Fixtures.Signals.FixedMovesSensor(labels, [0 0 1 0])}));
            v1 = tc.view(5, [1 1 1 1]);
            v2 = tc.view(99, [1 1 0 1]);                                      % a different world, move 3 not allowed
            mind.beginEpisode(v1);
            d1 = mind.decide(v1, RandStream('mrg32k3a', 'Seed', 2));
            d2 = mind.decide(v2, RandStream('mrg32k3a', 'Seed', 2));
            tc.verifyEqual([d1.Action d2.Action], [3 3], 'only the observed permitted move is ever chosen');
        end

        function aBodyWithoutANeededSensorIsRefused(tc)
            % I11: a Q mind needs a digit observation of the state; a moves-only body cannot carry it
            q = OceanStats.RL.TD.Values.TabularQ(true(4, 4), 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:4);
            body = OceanAgents.Body.EmptyBody(string(1:4), 'Sensors', {OceanAgents.Body.Sensors.MovesObservation(string(1:4))});
            tc.verifyError(@() OceanAgents.Embodied.Dualism.form(mind, body), 'OceanAgents:Embodied:Incompatible');
            tc.verifyFalse(body.isPaired());
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:4)));   % default sensors fit
        end

        function aRunningBehaviorIsLetContinue(tc)
            % O4 / I13: the mind sees the behavior through its observation and issues nothing while it runs
            moves = string(1:4);
            q = OceanStats.RL.TD.Values.TabularQ(true(10, 4), 0);
            T = q.Q;  T(3, :) = [0 0 0 5];  q.setTables(T, zeros(size(T)));   % move 4 is best in state 3
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0);
            robot = OceanAgents.Body.GridRobotBody(moves);
            OceanAgents.Embodied.Dualism.form(mind, robot);
            v = tc.view(3, true(1, 4));
            mind.beginEpisode(v);
            robot.command(OceanAgents.Minds.Signals.Signal.make(OceanAgents.Body.GridRobotBody.behaviorType(), ...
                struct('Behavior', "repeat", 'Move', "2", 'Times', 3, 'Interruptible', true)));
            before = robot.Motor.Count;
            acts = zeros(1, 3);
            for k = 1:3
                d = mind.decide(v, RandStream('mrg32k3a', 'Seed', k));
                acts(k) = d.Action;
            end
            tc.verifyEqual(acts, [2 2 2]);
            tc.verifyEqual(robot.Motor.Count - before, 3, 'the motor moved only for the behavior''s three steps');
            d = mind.decide(v, RandStream('mrg32k3a', 'Seed', 9));            % behavior done: the policy decides again
            tc.verifyEqual(d.Action, 4, 'the policy decides again: the greedy move');
            tc.verifyEmpty(robot.Behavior);
        end

        function theLearnerLearnsFromPerceptionsOnly(tc)
            % critic AB2 finding 1: a sensor that always reports cell 5 -> only row 5 of the table ever changes
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            codec = OceanAgents.Artifacts.DiscreteStateCodec(48);
            q = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.3, 'StateCodec', codec);
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:4), 'Sensors', ...
                {OceanTest.OceanAgents.Fixtures.Signals.FixedDigitSensor(5), OceanAgents.Body.Sensors.MovesObservation(string(1:4))}));
            learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec);
            mind.addLearner(learner, "q");
            host = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1);
            rec = OceanAgents.Runtime.SingleAgentRunner(OceanAgents.Env.NativeSingleAgentBridge( ...
                OceanStats.RL.Env.Wrappers.TimeLimit(env, 30)), host, 'Seed', 4).run(3);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyGreaterThan(learner.NumUpdates, 0);
            changed = find(any(q.Q ~= OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0).Q, 2));
            tc.verifyEqual(changed', 5, 'the learner only ever saw the perceived state 5');
        end

        function anAmbiguousPairingIsRefused(tc)
            % critic AB2 finding 3: two sensors of the same type would make the pairing a guess
            q = OceanStats.RL.TD.Values.TabularQ(true(10, 4), 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:4);
            two = OceanAgents.Body.EmptyBody(string(1:4), 'Sensors', {OceanAgents.Body.Sensors.DigitObservation('ID', "battery"), ...
                OceanAgents.Body.Sensors.DigitObservation('ID', "cell"), OceanAgents.Body.Sensors.MovesObservation(string(1:4))});
            tc.verifyError(@() OceanAgents.Embodied.Dualism.form(mind, two), 'OceanAgents:Embodied:Incompatible');
            D = OceanAgents.World.Guard.Dimension;                            % a typed battery (energy) is no longer ambiguous
            typed = OceanAgents.Body.EmptyBody(string(1:4), 'Sensors', ...
                {OceanTest.OceanAgents.Fixtures.Signals.FixedDigitSensor(9, 'ID', "battery", 'Dimension', D.E()), ...
                OceanAgents.Body.Sensors.DigitObservation('ID', "cell"), OceanAgents.Body.Sensors.MovesObservation(string(1:4))});
            OceanAgents.Embodied.Dualism.form(mind, typed);
            tc.verifyFalse(OceanAgents.Minds.Perception.BehaviorPerception().canRead( ...
                OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "sense")), 'behavior has its own type');
        end

        function aBehaviorMoveTheLawsForbidIsStopped(tc)
            % critic AB2 finding 4: check the Laws before a behavior step is used up
            labels = string(1:4);
            q = OceanStats.RL.TD.Values.TabularQ(true(10, 4), 0);
            T = q.Q;  T(:, 3) = 1;  q.setTables(T, zeros(size(T)));
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0);
            robot = OceanAgents.Body.GridRobotBody(labels, 'Sensors', {OceanAgents.Body.Sensors.DigitObservation(), ...
                OceanTest.OceanAgents.Fixtures.Signals.FixedMovesSensor(labels, [1 0 1 1]), ...
                OceanAgents.Body.Sensors.BehaviorObservation()});
            OceanAgents.Embodied.Dualism.form(mind, robot);
            v = tc.view(3, true(1, 4));
            mind.beginEpisode(v);
            B = OceanAgents.Body.GridRobotBody.behaviorType();
            robot.command(OceanAgents.Minds.Signals.Signal.make(B, struct('Behavior', "repeat", 'Move', "2", 'Times', 3)));
            d = mind.decide(v, RandStream('mrg32k3a', 'Seed', 1));
            tc.verifyEqual(d.Action, 3, 'the behavior was stopped; the policy chose a permitted move');
            tc.verifyEmpty(robot.Behavior);
            mind.endEpisode();
            mind.beginEpisode(tc.view(3, true(1, 4)));
            robot.command(OceanAgents.Minds.Signals.Signal.make(B, struct('Behavior', "repeat", 'Move', "2", 'Times', 3, ...
                'Interruptible', false)));
            v2 = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 2, 'FrameID', 7, 'Tick', 0, 'Observation', 3, ...
                'ActionLabels', 1:4, 'ActionMask', true(1, 4));
            tc.verifyError(@() mind.decide(v2, RandStream('mrg32k3a', 'Seed', 1)), 'OceanAgents:Body:BehaviorBlocked');
        end

        function aBehaviorRunEqualsTheSameMovesCommanded(tc)
            % O4: three behavior steps == the same move commanded three times (same actions, same motor work)
            labels = string(1:4);
            a = OceanAgents.Body.GridRobotBody(labels);
            b = OceanAgents.Body.GridRobotBody(labels);
            a.command(OceanAgents.Minds.Signals.Signal.make(OceanAgents.Body.GridRobotBody.behaviorType(), ...
                struct('Behavior', "repeat", 'Move', "2", 'Times', 3)));
            M = OceanAgents.Body.IBody.moveType(labels);
            seqA = strings(1, 3);  seqB = strings(1, 3);
            for k = 1:3
                seqA(k) = a.toAction();
                b.command(OceanAgents.Minds.Signals.Signal.make(M, "2"));
                seqB(k) = b.toAction();
            end
            tc.verifyEqual(seqA, seqB);
            tc.verifyEqual(a.Motor.Count, b.Motor.Count);
        end

        function ticTacToeDecisionsIgnoreTheBoard(tc)
            % O6: a moves-only agent decides the same whatever the board shows
            mind = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.RandomLegalPolicy(), 1:9);
            OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:9), ...
                'Sensors', {OceanAgents.Body.Sensors.MovesObservation(string(1:9))}));
            mask = logical([1 0 1 0 0 1 1 0 1]);
            v1 = OceanAgents.Core.ActorView('AgentID', "x", 'EpisodeID', 1, 'FrameID', 1, 'Tick', 0, ...
                'Observation', [0 1 0 2 2 0 0 1 0], 'ActionLabels', 1:9, 'ActionMask', mask);
            v2 = OceanAgents.Core.ActorView('AgentID', "x", 'EpisodeID', 1, 'FrameID', 2, 'Tick', 0, ...
                'Observation', [0 2 0 1 1 0 0 2 0], 'ActionLabels', 1:9, 'ActionMask', mask);
            mind.beginEpisode(v1);
            d1 = mind.decide(v1, RandStream('mrg32k3a', 'Seed', 6));
            d2 = mind.decide(v2, RandStream('mrg32k3a', 'Seed', 6));
            tc.verifyEqual(d1.Action, d2.Action);
            tc.verifyTrue(mask(d1.Action));
        end

        function newValueKindsAreChecked(tc)
            T = @(varargin) OceanAgents.Minds.Signals.SignalType(varargin{:});
            img = T('ValueKind', "array", 'Size', [2 3 1], 'Role', "sense");
            OceanAgents.Minds.Signals.Signal.make(img, ones(2, 3));
            tc.verifyError(@() OceanAgents.Minds.Signals.Signal.make(img, ones(3, 2)), 'OceanAgents:Signals:BadValue');
            tc.verifyError(@() T('ValueKind', "array", 'Role', "sense"), 'OceanAgents:Signals:BadType');       % no Size
            f = T('ValueKind', "flags", 'Width', 2, 'Labels', ["a" "b"], 'Role', "sense");
            tc.verifyEqual(OceanAgents.Minds.Signals.Signal.make(f, [1 0]).Value, [true false]);
            tc.verifyError(@() OceanAgents.Minds.Signals.Signal.make(f, [1 0 1]), 'OceanAgents:Signals:BadValue');
            tc.verifyError(@() T('ValueKind', "flags", 'Width', 2, 'Labels', "a", 'Role', "sense"), ...
                'OceanAgents:Signals:BadType');
            tc.verifyError(@() OceanAgents.Body.Sensors.VisionObservation(OceanAgents.Body.Sensors.GridRenderer(2, 2)).observe( ...
                tc.view(9, true(1, 2)), []), 'OceanAgents:Body:BadObservation');           % not a cell of a 2 x 2 grid
        end
    end
end
