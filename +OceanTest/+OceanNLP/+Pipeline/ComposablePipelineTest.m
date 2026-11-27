classdef ComposablePipelineTest < OceanTest.BaseTestCase
    methods (Test)
        function testStagesCanBeReorderedAndDisabled(testCase)
            p = OceanNLP.Pipeline.ComposablePipeline();
            p.add(OceanNLP.Pipeline.Stage.fromMethod("add", struct('step',@(x)x+1), "step"));
            p.add(OceanNLP.Pipeline.Stage.fromMethod("scale", struct('step',@(x)x*2), "step"));
            testCase.verifyEqual(p.run(3),8);
            p.move("scale",1);
            testCase.verifyEqual(p.run(3),7);
            p.enable("scale",false);
            testCase.verifyEqual(p.run(3),4);
        end

        function testBadStagesRejectedAtConstruction(testCase)
            F = @OceanNLP.Pipeline.Stage.fromFunction;
            M = @OceanNLP.Pipeline.Stage.fromMethod;
            C = @OceanNLP.Pipeline.Stage;
            testCase.verifyError(@() F("noinput", @() 1), 'OceanNLP:Pipeline:BadStage');
            testCase.verifyError(@() F("twoinputs", @(a,b) a+b), 'OceanNLP:Pipeline:BadStage');
            testCase.verifyError(@() F("notfn", 42), 'OceanNLP:Pipeline:BadStage');
            testCase.verifyError(@() M("nomethod", containers.Map(), "forward"), 'OceanNLP:Pipeline:BadStage');
            testCase.verifyError(@() M("nofield", struct('a',1), "a"), 'OceanNLP:Pipeline:BadStage');
            testCase.verifyError(@() F("1bad name", @(s) s), 'OceanNLP:Pipeline:BadStageName');
            testCase.verifyError(@() C("both", 'Step', @(s) s, 'Component', struct('f',@(s)s), 'Method', "f"), ...
                'OceanNLP:Pipeline:BadStage');
            % varargin handles and objects with a real method are accepted
            testCase.verifyClass(F("vararg", @(varargin) varargin{1}), 'OceanNLP.Pipeline.Stage');
            tok = OceanNLP.Tokenize.RegexTokenizer();
            testCase.verifyClass(M("tokenize", tok, "forward"), 'OceanNLP.Pipeline.Stage');
        end

        function testPipelineRejectsDuplicatesPositionsAndNonStages(testCase)
            p = OceanNLP.Pipeline.ComposablePipeline();
            p.add(OceanNLP.Pipeline.Stage.fromFunction("a", @(s) s));
            testCase.verifyError(@() p.add(OceanNLP.Pipeline.Stage.fromFunction("a", @(s) s)), ...
                'OceanNLP:Pipeline:DuplicateName');
            testCase.verifyError(@() p.add(OceanNLP.Pipeline.Stage.fromFunction("b", @(s) s), 'Position', 0), ...
                'OceanNLP:Pipeline:BadPosition');
            testCase.verifyError(@() p.add(OceanNLP.Pipeline.Stage.fromFunction("b", @(s) s), 'Position', 1.5), ...
                'OceanNLP:Pipeline:BadPosition');
            testCase.verifyError(@() p.add(@(s) s), 'OceanNLP:Pipeline:BadStage');
            p.add(OceanNLP.Pipeline.Stage.fromFunction("b", @(s) s), 'Position', 1);
            testCase.verifyEqual(p.names(), ["b" "a"]);
            testCase.verifyError(@() p.replace("b", OceanNLP.Pipeline.Stage.fromFunction("a", @(s) s)), ...
                'OceanNLP:Pipeline:DuplicateName');
            p.replace("b", OceanNLP.Pipeline.Stage.fromFunction("c", @(s) s + 1));
            testCase.verifyEqual(p.names(), ["c" "a"]);
            testCase.verifyEqual(p.run(1), 2);
        end

        function testStructStagesStillWorkButWarn(testCase)
            p = OceanNLP.Pipeline.ComposablePipeline();
            testCase.verifyWarning(@() p.add(struct('Name', "inc", 'Step', @(x) x + 1)), ...
                'OceanNLP:Pipeline:StructStageDeprecated');
            testCase.verifyEqual(p.run(1), 2);
            w = warning('off', 'OceanNLP:Pipeline:StructStageDeprecated');
            restore = onCleanup(@() warning(w));
            testCase.verifyError(@() p.add(struct('Name', "bad", 'Step', @() 1)), 'OceanNLP:Pipeline:BadStage');
        end

        function testValidateFindsMissingAndOverwrittenFields(testCase)
            S = @OceanNLP.Pipeline.Stage.fromFunction;
            p = OceanNLP.Pipeline.ComposablePipeline();
            p.add(S("retrieve", @(s) setfield(s, 'Chunks', "c"), 'Requires', "Query", 'Provides', "Chunks"));
            p.add(S("prompt", @(s) setfield(s, 'Prompt', s.Query + s.Chunks), 'Requires', ["Query" "Chunks"], 'Provides', "Prompt"));
            testCase.verifyTrue(p.validate('InitialFields', "Query").IsValid);
            testCase.verifyEqual(p.run(struct('Query', "q")).Prompt, "qc");

            p.move("prompt", 1);
            report = p.validate('InitialFields', "Query");
            testCase.verifyFalse(report.IsValid);
            testCase.verifyEqual(report.Issues.Severity, "error");
            testCase.verifyEqual(report.Issues.Stage, "prompt");
            testCase.verifySubstring(char(report.Issues.Message), '"Chunks"');
            % run refuses before executing any stage
            testCase.verifyError(@() p.run(struct('Query', "q")), 'OceanNLP:Pipeline:InvalidPipeline');

            p.move("prompt", 2);
            p.add(S("rewrite", @(s) s, 'Requires', "Query", 'Provides', "Query"));
            report = p.validate('InitialFields', "Query");
            testCase.verifyTrue(report.IsValid);
            testCase.verifyEqual(report.Issues.Severity, "warning");
        end

        function testUncontractedStagesAreLenient(testCase)
            S = @OceanNLP.Pipeline.Stage.fromFunction;
            p = OceanNLP.Pipeline.ComposablePipeline();
            p.add(S("opaque", @(s) setfield(s, 'Chunks', "c")));
            p.add(S("prompt", @(s) setfield(s, 'Prompt', s.Chunks), 'Requires', "Chunks", 'Provides', "Prompt"));
            report = p.validate();
            testCase.verifyTrue(report.IsValid);
            testCase.verifyEqual(report.Issues.Severity, ["info"; "warning"]);
            testCase.verifyEqual(p.run(struct()).Prompt, "c");
            summary = p.analyze();
            testCase.verifyEqual(summary.Status, ["unchecked"; "warning"]);
        end

        function testRuntimeErrorsNameTheStageAndContractsCanBeChecked(testCase)
            S = @OceanNLP.Pipeline.Stage.fromFunction;
            p = OceanNLP.Pipeline.ComposablePipeline();
            p.add(S("boom", @(s) throwingStage(s, 'Test:Boom', 'inner failure')));
            try
                p.run(1);
                testCase.verifyFail('Expected an error.');
            catch ME
                testCase.verifyEqual(ME.identifier, 'Test:Boom');
                testCase.verifySubstring(ME.message, 'Stage 1 "boom" failed: inner failure');
                testCase.verifyEqual(ME.cause{1}.message, 'inner failure');
            end

            q = OceanNLP.Pipeline.ComposablePipeline();
            q.add(S("liar", @(s) s, 'Provides', "Answer"));
            testCase.verifyEqual(q.run(struct('x', 1)), struct('x', 1));
            testCase.verifyError(@() q.run(struct('x', 1), 'CheckContracts', true), ...
                'OceanNLP:Pipeline:ContractViolated');
        end

        function testEditsAreSilentUnlessAutoValidateIsOn(testCase)
            S = @OceanNLP.Pipeline.Stage.fromFunction;
            p = OceanNLP.Pipeline.ComposablePipeline('InitialFields', "Raw");
            p.add(S("clean", @(s) s, 'Requires', "Raw", 'Provides', "Text"));
            p.add(S("count", @(s) s, 'Requires', "Text", 'Provides', "N"));
            % Off (default): a deliberately broken intermediate state is silent
            testCase.verifyWarningFree(@() p.move("count", 1));
            testCase.verifyFalse(p.validate().IsValid);   % uses p.InitialFields
            testCase.verifyWarningFree(@() p.move("count", 2));
            testCase.verifyTrue(p.validate().IsValid);

            % On: every breaking edit warns, the edit still happens
            p.AutoValidate = true;
            testCase.verifyWarning(@() p.move("count", 1), 'OceanNLP:Pipeline:ChainBroken');
            testCase.verifyEqual(p.names(), ["count" "clean"]);
            testCase.verifyWarningFree(@() p.move("count", 2));   % chain repaired
            testCase.verifyWarningFree(@() p.enable("count", true));
            testCase.verifyWarning(@() p.enable("clean", false), 'OceanNLP:Pipeline:ChainBroken');
            testCase.verifyWarningFree(@() p.enable("clean", true));
            testCase.verifyWarning(@() p.replace("clean", S("clean", @(s) s, 'Requires', "Raw", 'Provides', "Words")), ...
                'OceanNLP:Pipeline:ChainBroken');
            testCase.verifyWarning(@() p.remove("clean"), 'OceanNLP:Pipeline:ChainBroken');
            testCase.verifyWarningFree(@() p.add(S("clean", @(s) s, 'Requires', "Raw", 'Provides', "Text"), 'Position', 1));

            % Constructor: initial stages are added before AutoValidate turns on
            stages = {S("count", @(s) s, 'Requires', "Text", 'Provides', "N"), ...
                      S("clean", @(s) s, 'Requires', "Raw", 'Provides', "Text")};
            testCase.verifyWarning(@() OceanNLP.Pipeline.ComposablePipeline(stages, ...
                'AutoValidate', true, 'InitialFields', "Raw"), 'OceanNLP:Pipeline:ChainBroken');
            testCase.verifyWarningFree(@() OceanNLP.Pipeline.ComposablePipeline(fliplr(stages), ...
                'AutoValidate', true, 'InitialFields', "Raw"));
        end

        function testRAGWorkflowIsContracted(testCase)
            ranker = OceanTest.RAGTestRanker();
            generator = OceanNLP.Generation.FunctionGenerator(@(p) "ok");
            pipe = OceanNLP.RAG.RAGPipeline('Chunker', OceanNLP.Chunking.FixedTokenChunker('ChunkSize', 3), ...
                'Ranker', ranker, 'Generator', generator, 'TopK', 1);
            testCase.verifyEqual(pipe.Workflow.InitialFields, "Query");
            testCase.verifyFalse(pipe.Workflow.AutoValidate);
            testCase.verifyTrue(pipe.Workflow.validate().IsValid);
            testCase.verifyEmpty(pipe.Workflow.validate('InitialFields', "Query").Issues);
            pipe.Workflow.move("generate", 1);
            testCase.verifyFalse(pipe.Workflow.validate('InitialFields', "Query").IsValid);
        end
    end
end

function out = throwingStage(state, id, message)
    % An anonymous @(s) error(...) has no output; R2026a then raises
    % MATLAB:maxlhs instead of the intended error.
    out = state;
    error(id, message);
end
