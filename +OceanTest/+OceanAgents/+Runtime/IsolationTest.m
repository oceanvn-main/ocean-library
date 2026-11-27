classdef IsolationTest < OceanTest.BaseTestCase
    % ISOLATIONTEST  C01/C02 (agents share no mutable state), AP02, IG04: OceanAgents.Runtime.IsolationChecker
    % walks public and private properties, cells, structs, containers.Map values and function-handle workspaces,
    % reports useful paths, honours the ISpace allowlist, is cycle-safe and enforces the depth limit.
    % Contract: A0_Contract_Freeze.md sections 4.1 and 4.7.

    methods (Static, Access = private)
        function [hA, hB, mA, mB] = twoHosts()
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            mA = FixedProbabilityMind([0.5 0.5]);
            mB = FixedProbabilityMind([0.5 0.5]);
            hA = OceanAgents.Runtime.AgentHost("p1", mA, 'Learner', CountingLearner());
            hB = OceanAgents.Runtime.AgentHost("p2", mB, 'Learner', CountingLearner());
        end
    end

    methods (Test)
        function separatelyBuiltHostsShareNothing(tc)
            % C01: build two agents separately, run them together, then look for any shared handle.
            [hA, hB] = tc.twoHosts();
            env = OceanTest.OceanAgents.Fixtures.Runtime.ToyJointEnv('Rounds', 4);
            runner = OceanAgents.Runtime.JointRunner(env, {hA, hB}, 'Seed', 3);   % keeps the hosts bound
            rec = runner.run(2);
            tc.assertEqual(rec.Status, "Completed");
            [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEmpty(pa);
            tc.verifyEmpty(pb);
            tc.verifyFalse(hA.Stream == hB.Stream);
        end

        function nestedSharedHandleIsFoundWithPath(tc)
            % C02: a handle hidden in a struct in a cell in a cell of one mind, directly in the other's.
            [hA, hB, mA, mB] = tc.twoHosts();
            shared = OceanTest.OceanAgents.Fixtures.Runtime.EventLog();
            mA.Payload = {1, {struct('x', shared, 'y', 2)}};
            mB.Payload = {shared};
            [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEqual(pa, "Mind.Payload{2}{1}.x");
            tc.verifyEqual(pb, "Mind.Payload{1}");
        end

        function privatePropertiesAreWalked(tc)
            [hA, hB, mA, mB] = tc.twoHosts();
            q = containers.Map();                                   % stands in for a shared value table
            mA.Payload = {OceanTest.OceanAgents.Fixtures.Runtime.PrivateHolder(q)};
            mB.Payload = {OceanTest.OceanAgents.Fixtures.Runtime.PrivateHolder(q)};
            before = warning('query', 'MATLAB:structOnObject');
            pa = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEqual(pa, "Mind.Payload{1}.Secret");
            tc.verifyEqual(warning('query', 'MATLAB:structOnObject'), before, ...
                'the structOnObject warning state must be restored');
        end

        function mapValuesAndFunctionWorkspaces(tc)
            [hA, hB, mA, mB] = tc.twoHosts();
            shared = OceanTest.OceanAgents.Fixtures.Runtime.EventLog();
            m = containers.Map();
            m('cache') = shared;
            mA.Payload = {m};
            mB.Payload = {@() numel(shared.Events)};                 % captured in an anonymous function
            [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEqual(pa, "Mind.Payload{1}('cache')");
            tc.verifyEqual(pb, "Mind.Payload{1}@workspace.shared");
        end

        function dictionaryValuesAreWalked(tc)
            % F4: a MATLAB dictionary is a value type that struct() cannot open; its values must still be walked.
            [hA, hB, mA, mB] = tc.twoHosts();
            shared = OceanTest.OceanAgents.Fixtures.Runtime.EventLog();
            d = dictionary("k", {shared});
            mA.Payload = {d};
            mB.Payload = {d};
            [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEqual(pa, "Mind.Payload{1}('k')");
            tc.verifyEqual(pb, "Mind.Payload{1}('k')");
            mB.Payload = {dictionary("k", {OceanTest.OceanAgents.Fixtures.Runtime.EventLog()}), dictionary()};
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB), 'equal-looking, distinct handles');
        end

        function javaObjectsComparedByReference(tc)
            % F4: MATLAB == on Java objects calls equals() (two empty HashMaps are "equal"); identity must be by
            % reference. The same HashMap is reported; two distinct empty ones are not.
            tc.assumeTrue(usejava('jvm'), 'needs the JVM');
            [hA, hB, mA, mB] = tc.twoHosts();
            jm = java.util.HashMap();
            mA.Payload = {jm};
            mB.Payload = {struct('m', jm)};
            [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEqual(pa, "Mind.Payload{1}");
            tc.verifyEqual(pb, "Mind.Payload{1}.m");
            mB.Payload = {java.util.HashMap()};
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB));
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB, 'Allow', "java.util.HashMap"));
        end

        function pythonObjectsComparedByReferenceAndNotWalked(tc)
            % Gym import finding: a Python object's attribute graph has no end (ndarray.T), so it is an opaque leaf
            % compared by reference (`is`), as Java objects are. Equal but distinct lists are not shared.
            tc.assumeTrue(OceanTest.OceanAgents.Runtime.IsolationTest.hasPython(), 'needs a Python environment');
            [hA, hB, mA, mB] = tc.twoHosts();
            pl = py.list({1, 2});
            mA.Payload = {pl};
            mB.Payload = {struct('m', pl)};
            [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB);
            tc.verifyEqual(pa, "Mind.Payload{1}");
            tc.verifyEqual(pb, "Mind.Payload{1}.m");
            mB.Payload = {py.list({1, 2}), py.None, py.int(3)};
            mA.Payload = {pl, py.None, py.int(3)};
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB), 'equal content, distinct objects');
            mA.Payload = {py.complex(1, 2)};                          % .real/.imag/.conjugate: never walked
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB));
            mB.Payload = {pl};  mA.Payload = {pl};
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB, 'Allow', "py.list"));
        end

        function sharedStreamIsFound(tc)
            [hA, hB] = tc.twoHosts();
            rs = RandStream('mrg32k3a', 'Seed', 1);
            hA.attachStream(rs);
            hB.attachStream(rs);
            tc.verifyEqual(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB), "Stream");
        end

        function allowlistedSpaceIsNotReported(tc)
            % EnvSpec spaces are shared by design. (ISpace is a value class today, so a shared space is not even a
            % shared handle; the allowlist mechanism itself is exercised with a handle class below.)
            [hA, hB, mA, mB] = tc.twoHosts();
            sp = OceanStats.RL.Env.Spaces.Discrete(3);
            mA.Payload = {sp};
            mB.Payload = {struct('space', sp)};
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB));
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB, 'Allow', strings(1, 0)));
        end

        function allowlistSkipsClassAndSubclasses(tc)
            [hA, hB, mA, mB] = tc.twoHosts();
            shared = OceanTest.OceanAgents.Fixtures.Runtime.EventLog();
            mA.Payload = {shared};
            mB.Payload = {shared};
            tc.verifyEqual(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB), "Mind.Payload{1}");
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB, ...
                'Allow', "OceanTest.OceanAgents.Fixtures.Runtime.EventLog"));
            % allowlisting a superclass covers the subclass (here: everything that is a handle -> nothing walked)
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB, 'Allow', "handle"));
        end

        function cyclesAndRootIdentity(tc)
            [hA, hB, mA] = tc.twoHosts();
            mA.Payload = {mA, {mA}};                                 % mind refers to itself
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB));
            tc.verifyEqual(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hA), "<root>");
        end

        function depthLimit(tc)
            [hA, hB, mA] = tc.twoHosts();
            deep = {};
            for k = 1:60
                deep = {deep};
            end
            mA.Payload = deep;
            tc.verifyError(@() OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB), ...
                'OceanAgents:Runtime:IsolationDepth');
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hA, hB, 'MaxDepth', 100));
        end

        function pillars(tc)
            c = OceanAgents.Runtime.IsolationChecker();
            tc.verifyFalse(c.getMetadata().IsSymbolic);
            tc.verifyEqual(c.getPipelineSpec().ComponentName, "IsolationChecker");
        end
    end

    methods (Static)
        function tf = hasPython()
            try
                tf = strlength(pyenv().Version) > 0 && logical(py.operator.is_(py.None, py.None));
            catch
                tf = false;
            end
        end
    end
end
