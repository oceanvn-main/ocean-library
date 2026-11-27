classdef MultiHeadAttentionTest < OceanTest.BaseTestCase
    % MULTIHEADATTENTIONTEST  Unit tests for
    % OceanNLP.Embedding.Modules.MultiHeadAttention.
    %
    % Both hand-worked examples below independently recompute the expected
    % output via an EXPLICIT exp()/sum() softmax expression written
    % directly in the test (never by calling the class's private
    % softmaxRows_ helper) -- this is the same "reimplement the formula
    % independently" discipline CosineSimilarityTest uses (cosd(45) as the
    % expected value, not a call into the class under test).

    methods (Test)
        function testHandWorkedTwoHeadContiguousSplit(testCase)
            % d=4, numHeads=2 (dk=2), T=2, all-identity weights (WQ=WK=WV=
            % WO=I, zero bias) so Q=K=V=X and the contiguous row-block head
            % split (rows 1:2 = head 1, rows 3:4 = head 2) is directly
            % verifiable: head 2's input slice is all-zero, so head 2's
            % contribution must come out all-zero too.
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            I4 = eye(4); z4 = zeros(4, 1);
            mha = MultiHeadAttention(I4, z4, I4, z4, I4, z4, I4, z4, 2);

            X = [1 0; ...
                 0 1; ...
                 0 0; ...
                 0 0];   % (4 x 2): head-1 slice = eye(2), head-2 slice = zeros(2)

            Y = mha.forward(X);

            dk = 2;
            X1 = X(1:2, :);
            X2 = X(3:4, :);

            scores1 = (X1' * X1) / sqrt(dk);
            A1 = exp(scores1 - max(scores1, [], 2)) ./ sum(exp(scores1 - max(scores1, [], 2)), 2);
            head1Out = X1 * A1';

            scores2 = (X2' * X2) / sqrt(dk);   % all zeros
            A2 = exp(scores2 - max(scores2, [], 2)) ./ sum(exp(scores2 - max(scores2, [], 2)), 2);
            head2Out = X2 * A2';

            expectedY = [head1Out; head2Out];
            testCase.verifyEqual(Y, expectedY, 'AbsTol', 1e-10);

            % Head 2's contiguous slice (rows 3:4) is all-zero input -> all-zero output.
            testCase.verifyEqual(Y(3:4, :), zeros(2, 2), 'AbsTol', 1e-12);

            % Softmax axis check: each row of the per-head score matrix sums to 1.
            testCase.verifyEqual(sum(A1, 2), ones(2, 1), 'AbsTol', 1e-12);
        end

        function testHandWorkedSoftmaxAxisMatters(testCase)
            % d=2, numHeads=1 (dk=2), T=2, ASYMMETRIC WQ~=WK (WV=WO=I) so
            % the (T x T) score matrix is genuinely non-symmetric -- this
            % is essential: with Q=K (as in the identity-weight test
            % above), Q'K is always a symmetric Gram matrix, and row- vs
            % column-softmax coincidentally agree, hiding a wrong-axis bug.
            % Here they must differ, proving the softmax axis is actually
            % exercised, not just assumed correct by a lucky symmetric case.
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            WQ = [1 2; 0 1]; bQ = [0; 0];
            WK = [1 0; 2 1]; bK = [0; 0];
            I2 = eye(2); z2 = zeros(2, 1);
            mha = MultiHeadAttention(WQ, bQ, WK, bK, I2, z2, I2, z2, 1);

            X = [1 0; 0 1];   % (2 x 2) basis vectors, T=2
            Y = mha.forward(X);

            dk = 2;
            Q = WQ * X; K = WK * X; V = X;
            scores = (Q' * K) / sqrt(dk);

            % Confirm the score matrix is genuinely asymmetric (otherwise
            % this example would not distinguish the two softmax axes).
            testCase.verifyGreaterThan(abs(scores(2,1) - scores(1,2)), 1);

            correctA = exp(scores - max(scores, [], 2)) ./ sum(exp(scores - max(scores, [], 2)), 2);
            expectedY = V * correctA';
            testCase.verifyEqual(Y, expectedY, 'AbsTol', 1e-10);

            % Row-softmax property: every row of A sums to 1.
            testCase.verifyEqual(sum(correctA, 2), ones(2, 1), 'AbsTol', 1e-12);

            % NEGATIVE CONTROL: a column-normalized ("wrong axis") softmax
            % over the SAME score matrix does NOT have row sums of 1, and
            % produces a genuinely different output -- proving the correct
            % row-wise axis is what's actually being exercised.
            wrongA = exp(scores - max(scores, [], 1)) ./ sum(exp(scores - max(scores, [], 1)), 1);
            testCase.verifyGreaterThan(max(abs(sum(wrongA, 2) - 1)), 1e-3);
            wrongY = V * wrongA';
            testCase.verifyGreaterThan(max(abs(wrongY(:) - Y(:))), 1e-3);
        end

        function testHiddenDimMismatchAtConstruction(testCase)
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            I2 = eye(2); z2 = zeros(2, 1);
            badWK = eye(3);   % wrong dimension vs. WQ's d=2
            testCase.verifyError(@() MultiHeadAttention(I2, z2, badWK, zeros(3,1), I2, z2, I2, z2, 1), ...
                'OceanNLP:Embedding:Modules:MultiHeadAttention:ShapeMismatch');
        end

        function testNumHeadsMustEvenlyDivideHiddenDim(testCase)
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            I4 = eye(4); z4 = zeros(4, 1);
            testCase.verifyError(@() MultiHeadAttention(I4, z4, I4, z4, I4, z4, I4, z4, 3), ...
                'OceanNLP:Embedding:Modules:MultiHeadAttention:HeadDimMismatch');
        end

        function testForwardHiddenDimMismatchErrors(testCase)
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            I2 = eye(2); z2 = zeros(2, 1);
            mha = MultiHeadAttention(I2, z2, I2, z2, I2, z2, I2, z2, 1);
            X3 = ones(3, 2);   % wrong number of rows (d=3 vs constructed d=2)
            testCase.verifyError(@() mha.forward(X3), ...
                'OceanNLP:Embedding:Modules:MultiHeadAttention:HiddenDimMismatch');
        end

        function testOutputShapeMatchesInput(testCase)
            import OceanNLP.Embedding.Modules.MultiHeadAttention
            I4 = eye(4); z4 = zeros(4, 1);
            mha = MultiHeadAttention(I4, z4, I4, z4, I4, z4, I4, z4, 2);
            X = reshape(1:20, 4, 5);   % fixed, deterministic (4 x 5)
            Y = mha.forward(X);
            testCase.verifySize(Y, [4 5]);
        end
    end
end
