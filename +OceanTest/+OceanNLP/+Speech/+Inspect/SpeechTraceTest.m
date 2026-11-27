classdef SpeechTraceTest < OceanTest.BaseTestCase
    % SPEECHTRACETEST  OceanNLP.Speech.Inspect.SpeechTrace: schema, lineage, budgets, no-op.

    methods (Test)
        function recordsStepsArtifactsAndLineage(tc)
            t = OceanNLP.Speech.Inspect.SpeechTrace();
            x = (1:10)';
            a = t.record("Input", "input", x);
            b = t.record("Square", "compute", x.^2, a, 'Summary', "x.^2");
            tc.verifyEqual([a b], ["artifact-1" "artifact-2"]);
            r = t.reader();
            tc.verifyClass(r, 'OceanNLP.Inspect.MemoryReader');
            steps = r.listSteps();
            tc.verifyEqual(numel(steps), 2);
            tc.verifyEqual(steps(2).InputIds, "artifact-1");
            tc.verifyEqual(steps(2).OutputIds, "artifact-2");
            tc.verifyEqual(r.listStages(), ["input" "compute"]);
            art = r.getArtifact(b);
            tc.verifyEqual(art.Value, x.^2);
            tc.verifyEqual(art.Status, "recorded");
            tc.verifyEqual(art.ByteCount, 80);
            lin = r.getLineage(a);
            tc.verifyEqual(lin.ConsumerIds, "step-2");
            tc.verifyEqual(t.BytesUsed, 160);
        end

        function perArtifactBudgetOmitsValueButKeepsMetadata(tc)
            t = OceanNLP.Speech.Inspect.SpeechTrace('MaxArtifactBytes', 100, 'MaxTotalBytes', 1e6);
            small = t.record("Small", "s", zeros(5, 1));          % 40 B
            big = t.record("Big", "s", zeros(20, 3), small);       % 480 B > 100
            r = t.reader();
            a = r.getArtifact(big);
            tc.verifyEqual(a.Status, "omitted");
            tc.verifyEmpty(a.Value);
            tc.verifyEqual(a.ByteCount, 480);
            tc.verifyEqual(a.Kind, "double");
            tc.verifySubstring(a.Summary, "20×3");
            tc.verifySubstring(a.Summary, "omitted");
            tc.verifyEqual(r.getStep("step-2").Status, "omitted");
            tc.verifyEqual(r.getStep("step-2").InputIds, small);   % lineage kept
            tc.verifyEqual(t.BytesUsed, 40);
        end

        function totalBudgetIsCumulative(tc)
            t = OceanNLP.Speech.Inspect.SpeechTrace('MaxArtifactBytes', 1e6, 'MaxTotalBytes', 100);
            t.record("A", "s", zeros(8, 1));    % 64 B, fits
            t.record("B", "s", zeros(8, 1));    % 64 B, 128 > 100 -> omitted
            t.record("C", "s", zeros(4, 1));    % 32 B, 96 <= 100 -> fits
            r = t.reader();
            tc.verifyEqual([r.getArtifact("artifact-1").Status, r.getArtifact("artifact-2").Status, ...
                r.getArtifact("artifact-3").Status], ["recorded" "omitted" "recorded"]);
            tc.verifyEqual(t.BytesUsed, 96);
        end

        function disabledTraceIsStrictNoOp(tc)
            t = OceanNLP.Speech.Inspect.SpeechTrace('Enabled', false);
            s0 = rng;
            id = t.record("A", "s", rand(3), "artifact-99");   % unknown input not even checked
            s1 = rng;
            rng(s0); rand(3); s2 = rng;                       % only the caller's rand moved the RNG
            tc.verifyEqual(id, "");
            tc.verifyEqual(s1, s2);
            tc.verifyEqual(t.numRecords(), 0);
            tc.verifyEqual(t.BytesUsed, 0);
            tc.verifyEmpty(t.reader().listSteps());
        end

        function unknownInputIdIsRejected(tc)
            t = OceanNLP.Speech.Inspect.SpeechTrace();
            tc.verifyError(@() t.record("A", "s", 1, "artifact-7"), ...
                'OceanNLP:Speech:Inspect:SpeechTrace:UnknownInput');
        end

        function byteCountIsDeep(tc)
            s = struct('a', zeros(10, 1), 'b', {{int8([1 2 3]), "xy"}});
            sw = whos('s');
            expected = 80 + 3 + OceanNLP.Speech.Inspect.SpeechTrace.byteCount("xy");
            tc.verifyEqual(OceanNLP.Speech.Inspect.SpeechTrace.byteCount(s), expected);
            tc.verifyLessThan(expected, sw.bytes);       % whos adds struct/cell headers
            w = OceanNLP.Speech.Core.Waveform(zeros(100, 1), 8000);
            tc.verifyGreaterThanOrEqual(OceanNLP.Speech.Inspect.SpeechTrace.byteCount(w), 800);
        end
    end
end
