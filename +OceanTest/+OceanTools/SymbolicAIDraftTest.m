classdef SymbolicAIDraftTest < OceanTest.BaseTestCase
    % SYMBOLICAIDRAFTTEST Unit tests for AI draft validation without API calls.

    methods (Test)
        function testAcceptsSourceGroundedDraft(testCase)
            payload = createValidPayload();

            draft = OceanTools.SymbolicWorkbench.validateAIDraft(payload, 20);

            testCase.verifyEqual(draft.title, "Gaussian score candidate");
            testCase.verifyEqual(draft.candidates.parameters, ["x", "mu"]);
            testCase.verifyEqual(draft.candidates.sourceLines, [3, 7]);
        end

        function testRejectsCitationOutsideSource(testCase)
            payload = createValidPayload();
            payload.candidates.sourceLines = [3, 21];

            testCase.verifyError(@() OceanTools.SymbolicWorkbench.validateAIDraft(payload, 20), ...
                'OceanTools:SymbolicWorkbench:InvalidSourceLines');
        end

        function testRejectsUnsupportedDomain(testCase)
            payload = createValidPayload();
            payload.candidates.outputDomain = "matrix";

            testCase.verifyError(@() OceanTools.SymbolicWorkbench.validateAIDraft(payload, 20), ...
                'OceanTools:SymbolicWorkbench:InvalidDomain');
        end

        function testRejectsNonScalarFunctionCall(testCase)
            payload = createValidPayload();
            payload.candidates.expression = "rand(x)";

            testCase.verifyError(@() OceanTools.SymbolicWorkbench.validateAIDraft(payload, 20), ...
                'OceanTools:SymbolicWorkbench:InvalidExpression');
        end

        function testRejectsUnlabelledExplanation(testCase)
            payload = createValidPayload();
            payload.candidates.explanation = "The source defines a score-like scalar expression.";

            testCase.verifyError(@() OceanTools.SymbolicWorkbench.validateAIDraft(payload, 20), ...
                'OceanTools:SymbolicWorkbench:InvalidExplanation');
        end

        function testAllowsEmptyCandidateList(testCase)
            payload = createValidPayload();
            payload.candidates = [];

            draft = OceanTools.SymbolicWorkbench.validateAIDraft(payload, 20);

            testCase.verifyEmpty(draft.candidates);
        end

        function testEmptySourceStopsBeforeCredentialOrNetwork(testCase)
            testCase.verifyError(@() OceanTools.SymbolicWorkbench.aiDraft( ...
                "", "source.m", "Extract scalar mathematics."), ...
                'OceanTools:SymbolicWorkbench:EmptySource');
        end
    end
end

function payload = createValidPayload()
candidate = struct;
candidate.name = "Gaussian score";
candidate.parameters = ["x", "mu"];
candidate.parameterDomains = ["real", "real"];
candidate.expression = "x - mu";
candidate.outputDomain = "real";
candidate.explanation = "Source faithful: the cited lines define a score-like scalar expression.";
candidate.sourceLines = [3, 7];

payload = struct;
payload.title = "Gaussian score candidate";
payload.summary = "A source-grounded scalar symbolic candidate.";
payload.assumptions = "The inspected expression is scalar.";
payload.limitations = "No numerical evaluation is performed.";
payload.candidates = candidate;
end
