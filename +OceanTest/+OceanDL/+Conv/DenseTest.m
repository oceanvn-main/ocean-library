classdef DenseTest < OceanTest.BaseTestCase
    % DENSETEST  Unit tests for OceanDL.Conv.Dense.

    methods (Test)
        function testHandWorkedForward(testCase)
            import OceanDL.Conv.Dense
            W = [1 2; 3 4];
            b = [10; 20];
            dense = Dense(2, 2, 'Weight', W, 'Bias', b);
            X = [1; 1];             % single sample
            Y = dense.forward(X);
            testCase.verifyEqual(Y, [1*1+2*1+10; 3*1+4*1+20], 'AbsTol', 1e-12);
        end

        function testDeferredInFeaturesInferredFromFirstForward(testCase)
            import OceanDL.Conv.Dense
            dense = Dense(0, 5, 'Seed', 1);
            X = randn(12, 3);
            Y = dense.forward(X);
            testCase.verifyEqual(dense.InFeatures, 12);
            testCase.verifySize(Y, [5 3]);
        end

        function testInputSizeMismatchErrors(testCase)
            import OceanDL.Conv.Dense
            dense = Dense(4, 2, 'Seed', 1);
            testCase.verifyError(@() dense.forward(randn(5, 1)), ...
                'OceanDL:Conv:Dense:InputSizeMismatch');
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Conv.Dense
            dense = Dense(4, 2, 'Seed', 1);
            testCase.verifyError(@() dense.backward(ones(2, 1)), ...
                'OceanDL:Conv:Dense:NoForwardCache');
        end

        function testFiniteDifferenceGradcheck(testCase)
            import OceanDL.Conv.Dense
            rng(5);
            X = randn(4, 3);
            W = randn(3, 4);
            b = randn(3, 1);
            dense = Dense(4, 3, 'Weight', W, 'Bias', b);
            Y = dense.forward(X);
            dX = dense.backward(2 * Y);           % loss = sum(Y.^2)
            [dW, db] = dense.getWeightGrad();

            lossOf = @(Xv, Wv, bv) sum((Wv*Xv + bv).^2, 'all');
            eps_ = 1e-5;

            dXnum = zeros(size(X));
            for k = 1:numel(X)
                Xp = X; Xp(k) = Xp(k) + eps_;
                Xm = X; Xm(k) = Xm(k) - eps_;
                dXnum(k) = (lossOf(Xp, W, b) - lossOf(Xm, W, b)) / (2 * eps_);
            end
            testCase.verifyEqual(dX, dXnum, 'AbsTol', 1e-4);

            dWnum = zeros(size(W));
            for k = 1:numel(W)
                Wp = W; Wp(k) = Wp(k) + eps_;
                Wm = W; Wm(k) = Wm(k) - eps_;
                dWnum(k) = (lossOf(X, Wp, b) - lossOf(X, Wm, b)) / (2 * eps_);
            end
            testCase.verifyEqual(dW, dWnum, 'AbsTol', 1e-4);

            dbnum = zeros(size(b));
            for k = 1:numel(b)
                bp = b; bp(k) = bp(k) + eps_;
                bm = b; bm(k) = bm(k) - eps_;
                dbnum(k) = (lossOf(X, W, bp) - lossOf(X, W, bm)) / (2 * eps_);
            end
            testCase.verifyEqual(db, dbnum, 'AbsTol', 1e-4);
        end
    end
end
