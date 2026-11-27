classdef TransformerEncoderBlockTest < OceanTest.BaseTestCase
    % TRANSFORMERENCODERBLOCKTEST  Unit tests for
    % OceanNLP.Embedding.Modules.TransformerEncoderBlock.
    %
    % HAND-WORKED POST-LN EXAMPLE (d=2, numHeads=1, d_ff=2, T=2): builds a
    % real MultiHeadAttention (already independently verified by its own
    % test file) plus two LayerNormLayer objects with NON-DEFAULT
    % gamma/beta (so the scale/shift step is actually exercised, not
    % hidden behind gamma=1/beta=0), and independently re-derives the
    % EXACT Post-LN wiring documented in the source
    % (norm(x + sublayer(x)) for BOTH the attention and feed-forward
    % sublayers) via explicit MATLAB expressions in this test -- never by
    % calling the block's own forward() to produce its own expectation.

    properties
        MHA
        LN1
        LN2
        W1; B1; W2; B2
        X
    end

    methods (TestClassSetup)
        function buildFixture(testCase)
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            import OceanDL.Norm.LayerNormLayer

            WQ = [1 0.5; 0.2 1];   bQ = [0.1; -0.1];
            WK = [0.9 -0.3; 0.4 0.8]; bK = [0; 0.2];
            WV = eye(2);            bV = [0; 0];
            WO = [0.7 0.3; 0.1 0.9]; bO = [0.05; -0.05];
            testCase.MHA = MultiHeadAttention(WQ, bQ, WK, bK, WV, bV, WO, bO, 1);

            ln1 = LayerNormLayer(2);
            ln1.Gamma = [1.2; 0.8];
            ln1.Beta  = [0.1; -0.1];
            testCase.LN1 = ln1;

            ln2 = LayerNormLayer(2);
            ln2.Gamma = [1.0; 1.0];
            ln2.Beta  = [0.0; 0.0];
            testCase.LN2 = ln2;

            testCase.W1 = [0.5 0.5; -0.2 0.3];   % (d_ff x d)
            testCase.B1 = [0; 0.1];
            testCase.W2 = [0.6 -0.4; 0.2 0.9];   % (d x d_ff)
            testCase.B2 = [0; 0];

            testCase.X = [1 0; 0 1];   % (d x T) = (2 x 2)
        end
    end

    methods (Test)
        function testHandWorkedPostLNOrdering(testCase)
            import OceanNLP.Embedding.Modules.TransformerEncoderBlock
            block = TransformerEncoderBlock(testCase.MHA, testCase.LN1, testCase.LN2, ...
                testCase.W1, testCase.B1, testCase.W2, testCase.B2);

            output = block.forward(testCase.X);

            % Independent re-derivation of the documented Post-LN order:
            %   X1 = LayerNorm1( X_in + Attention(X_in) )       residual THEN norm
            %   X2 = LayerNorm2( X1   + FFN(X1) )                 residual THEN norm
            attnOut = testCase.MHA.forward(testCase.X);   % trusted: MHA has its own dedicated tests
            sum1 = testCase.X + attnOut;
            X1 = manualLayerNorm(sum1, testCase.LN1.Gamma, testCase.LN1.Beta, testCase.LN1.Epsilon);

            ffnPre    = testCase.W1 * X1 + testCase.B1;
            ffnHidden = ffnPre .* normcdf(ffnPre);         % GELU, written independently
            ffnOut    = testCase.W2 * ffnHidden + testCase.B2;
            sum2 = X1 + ffnOut;
            X2 = manualLayerNorm(sum2, testCase.LN2.Gamma, testCase.LN2.Beta, testCase.LN2.Epsilon);

            testCase.verifyEqual(output, X2, 'AbsTol', 1e-9);
        end

        function testPreLNAlternativeWiringDoesNotMatch(testCase)
            % NEGATIVE CONTROL: a plausible Pre-LN-style alternative
            % (norm BEFORE the sublayer, i.e. X1 = X + Attn(LN1(X)),
            % X2 = X1 + FFN(LN2(X1)) -- no final norm on the residual sum)
            % must NOT reproduce the real block's output. If it did, the
            % Post-LN/Pre-LN distinction would be a no-op for this input,
            % which would mean the ordering test above wasn't actually
            % locking in anything.
            import OceanNLP.Embedding.Modules.TransformerEncoderBlock
            block = TransformerEncoderBlock(testCase.MHA, testCase.LN1, testCase.LN2, ...
                testCase.W1, testCase.B1, testCase.W2, testCase.B2);

            realOutput = block.forward(testCase.X);

            % Pre-LN-style alternative wiring:
            normedX = manualLayerNorm(testCase.X, testCase.LN1.Gamma, testCase.LN1.Beta, testCase.LN1.Epsilon);
            attnOutPre = testCase.MHA.forward(normedX);
            preX1 = testCase.X + attnOutPre;   % residual added AFTER attention on the NORMED input, no post-norm

            normedX1 = manualLayerNorm(preX1, testCase.LN2.Gamma, testCase.LN2.Beta, testCase.LN2.Epsilon);
            ffnPrePre    = testCase.W1 * normedX1 + testCase.B1;
            ffnHiddenPre = ffnPrePre .* normcdf(ffnPrePre);
            ffnOutPre    = testCase.W2 * ffnHiddenPre + testCase.B2;
            preX2 = preX1 + ffnOutPre;   % no final norm, unlike Post-LN

            testCase.verifyGreaterThan(max(abs(preX2(:) - realOutput(:))), 1e-2);
        end

        function testOutputShapeMatchesInput(testCase)
            import OceanNLP.Embedding.Modules.TransformerEncoderBlock
            block = TransformerEncoderBlock(testCase.MHA, testCase.LN1, testCase.LN2, ...
                testCase.W1, testCase.B1, testCase.W2, testCase.B2);
            output = block.forward(testCase.X);
            testCase.verifySize(output, [2 2]);
        end

        function testW1ShapeMismatchErrorsAtConstruction(testCase)
            import OceanNLP.Embedding.Modules.TransformerEncoderBlock
            badW1 = [0.5 0.5 0.1; -0.2 0.3 0.1];   % 3 columns, but d=2
            testCase.verifyError(@() TransformerEncoderBlock(testCase.MHA, testCase.LN1, testCase.LN2, ...
                badW1, testCase.B1, testCase.W2, testCase.B2), ...
                'OceanNLP:Embedding:Modules:TransformerEncoderBlock:W1ShapeMismatch');
        end
    end
end

function Zout = manualLayerNorm(Zin, gamma, beta, epsilon)
    % MANUALLAYERNORM  Independent re-derivation of LayerNormLayer's own
    % documented formula (Ba et al., 2016), written directly here rather
    % than delegating to the LayerNormLayer instance's normalize() method,
    % so this file's Post-LN verification does not silently depend on
    % LayerNormLayer being bug-for-bug identical to its own docstring.
    % BIASED (population, divide-by-N) variance -- std(Zin,1,1), not
    % MATLAB's std() default -- matching PyTorch/TensorFlow LayerNorm and
    % Ba et al.'s actual definition. This independent re-derivation
    % originally used the unbiased estimator too (mirroring a bug that was
    % also present in LayerNormLayer.m at the time); both were corrected
    % together after the discrepancy was found via real-bert-tiny external
    % validation -- see LayerNormLayer.m's class docstring and
    % Planning/System/Retrieval/Retrieval_System_Master_Plan.md §7.
    mu  = mean(Zin, 1);
    sig = std(Zin, 1, 1) + epsilon;
    Zout = gamma .* ((Zin - mu) ./ sig) + beta;
end
