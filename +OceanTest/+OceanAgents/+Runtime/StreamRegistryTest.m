classdef StreamRegistryTest < OceanTest.BaseTestCase
    % STREAMREGISTRYTEST  OceanAgents.Runtime.StreamRegistry: FNV-1a substream indexing, handle reuse, AP38
    % (adding a purpose never changes another purpose's draws), collisions, nextSeed range, global stream untouched.
    % Expected values: A0_Contract_Freeze.md section 4.3; FNV-1a reference vectors from the FNV specification
    % (offset basis 2166136261 for the empty string).

    methods (Test)
        function fnvReferenceVectors(tc)
            R = @OceanAgents.Runtime.StreamRegistry.fnv1a32;
            tc.verifyEqual(R(""), 2166136261);            % 0x811C9DC5 (offset basis)
            tc.verifyEqual(R("a"), 3826002220);           % 0xE40C292C
            tc.verifyEqual(R("foobar"), 3214735720);      % 0xBF9CF968
            % UTF-8: U+00E9 hashes its two UTF-8 bytes C3 A9 (513665217), not the single Latin-1 byte E9
            % (1812687940). Values from an independent (Python) FNV-1a implementation.
            tc.verifyEqual(R(string(char(233))), 513665217);
            tc.verifyEqual(OceanAgents.Runtime.StreamRegistry.indexOf("foobar"), 1 + mod(3214735720, 2^31));
        end

        function streamIsTheContractedSubstream(tc)
            reg = OceanAgents.Runtime.StreamRegistry(11);
            rs = reg.stream("env");
            k = 1 + mod(OceanAgents.Runtime.StreamRegistry.fnv1a32("env"), 2^31);
            tc.verifyEqual(rs.Type, 'mrg32k3a');
            tc.verifyEqual(double(rs.NumStreams), 2^31);
            tc.verifyEqual(double(rs.StreamIndex), k);
            tc.verifyEqual(double(rs.Seed), 11);
            ref = RandStream.create('mrg32k3a', 'Seed', 11, 'NumStreams', 2^31, 'StreamIndices', k);
            tc.verifyEqual(rand(rs, 1, 5), rand(ref, 1, 5));
            tc.verifyTrue(reg.stream("env") == rs, 'same purpose must return the same handle');
            tc.verifyFalse(reg.stream("agent:a:action") == rs);
        end

        function addingPurposeDoesNotChangeOthers(tc)
            % AP38: draws of "agent:alice:action" are the same whether or not other purposes were created first.
            regA = OceanAgents.Runtime.StreamRegistry(3);
            drawsA = rand(regA.stream("agent:alice:action"), 1, 20);
            regB = OceanAgents.Runtime.StreamRegistry(3);
            rand(regB.stream("agent:zed:action"), 1, 7);
            regB.nextSeed("env");
            drawsB = rand(regB.stream("agent:alice:action"), 1, 20);
            tc.verifyEqual(drawsB, drawsA);
            tc.verifyEqual(regB.purposes(), sort(["agent:alice:action" "agent:zed:action" "env"]));
        end

        function collisionIsAnError(tc)
            % These two purposes collide modulo 2^31 (the test checks that from the formula before relying on it).
            p1 = "purpose:168724";
            p2 = "purpose:598200";
            tc.assertEqual(OceanAgents.Runtime.StreamRegistry.indexOf(p1), OceanAgents.Runtime.StreamRegistry.indexOf(p2));
            reg = OceanAgents.Runtime.StreamRegistry(0);
            reg.stream(p1);
            tc.verifyError(@() reg.stream(p2), 'OceanAgents:Runtime:StreamCollision');
            tc.verifyTrue(isa(reg.stream(p1), 'RandStream'));      % the first owner still works
        end

        function nextSeedRangeAndDeterminism(tc)
            reg1 = OceanAgents.Runtime.StreamRegistry(5);
            reg2 = OceanAgents.Runtime.StreamRegistry(5);
            reg3 = OceanAgents.Runtime.StreamRegistry(6);
            s1 = arrayfun(@(~) reg1.nextSeed("env"), 1:50);
            s2 = arrayfun(@(~) reg2.nextSeed("env"), 1:50);
            s3 = arrayfun(@(~) reg3.nextSeed("env"), 1:50);
            tc.verifyEqual(s1, s2);
            tc.verifyNotEqual(s1, s3);
            tc.verifyTrue(all(s1 >= 0 & s1 <= 2^32 - 1 & s1 == fix(s1)));
            tc.verifyGreaterThan(numel(unique(s1)), 45);
        end

        function globalStreamUntouched(tc)
            g = RandStream.getGlobalStream();
            before = g.State;
            reg = OceanAgents.Runtime.StreamRegistry(9);
            reg.nextSeed("env");
            rand(reg.stream("agent:x:action"), 1, 3);
            tc.verifyEqual(RandStream.getGlobalStream().State, before);
        end

        function rejectsBadInput(tc)
            tc.verifyError(@() OceanAgents.Runtime.StreamRegistry(-1), 'MATLAB:validators:mustBeNonnegative');
            tc.verifyError(@() OceanAgents.Runtime.StreamRegistry(1.5), 'MATLAB:validators:mustBeInteger');
            reg = OceanAgents.Runtime.StreamRegistry(1);
            tc.verifyError(@() reg.stream(""), 'MATLAB:validators:mustBeNonzeroLengthText');
        end

        function pillars(tc)
            reg = OceanAgents.Runtime.StreamRegistry(1);
            tc.verifyFalse(reg.getMetadata().IsSymbolic);
            tc.verifyEqual(reg.getPipelineSpec().ComponentName, "StreamRegistry");
        end
    end
end
