classdef Conv2DTest < OceanTest.BaseTestCase
    % CONV2DTEST  Unit tests for OceanDL.Conv.Conv2D.
    %
    % HAND-WORKED EXAMPLE (4x4 input, 3x3 diagonal kernel W=eye(3), no bias,
    % stride 1, no padding -- cross-correlation, so window start (s_i,s_j)
    % picks x(s_i,s_j)+x(s_i+1,s_j+1)+x(s_i+2,s_j+2)):
    %   X = [ 1  2  3  4
    %         5  6  7  8
    %         9 10 11 12
    %        13 14 15 16 ]
    %   Window(1,1): x(1,1)+x(2,2)+x(3,3) =  1+ 6+11 = 18
    %   Window(1,2): x(1,2)+x(2,3)+x(3,4) =  2+ 7+12 = 21
    %   Window(2,1): x(2,1)+x(3,2)+x(4,3) =  5+10+15 = 30
    %   Window(2,2): x(2,2)+x(3,3)+x(4,4) =  6+11+16 = 33
    %   Y = [18 21; 30 33]

    properties
        X4x4
    end

    methods (TestClassSetup)
        function buildInput(testCase)
            testCase.X4x4 = reshape([1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16], 4, 4, 1, 1);
        end
    end

    methods (Test)
        function testHandWorkedForwardDiagonalKernel(testCase)
            import OceanDL.Conv.Conv2D
            W = reshape(eye(3), 3, 3, 1, 1);
            conv = Conv2D([3 3], 1, 'InChannels', 1, 'HasBias', false, 'Weight', W);
            Y = conv.forward(testCase.X4x4);
            testCase.verifyEqual(Y, reshape([18 21; 30 33], 2, 2, 1, 1), 'AbsTol', 1e-12);
        end

        function testBiasIsAddedPerOutChannel(testCase)
            import OceanDL.Conv.Conv2D
            W = reshape(eye(3), 3, 3, 1, 1);
            conv = Conv2D([3 3], 1, 'InChannels', 1, 'Weight', W, 'Bias', 100);
            Y = conv.forward(testCase.X4x4);
            testCase.verifyEqual(Y, reshape([118 121; 130 133], 2, 2, 1, 1), 'AbsTol', 1e-12);
        end

        function testOutputSizeWithStrideAndPadding(testCase)
            import OceanDL.Conv.Conv2D
            conv = Conv2D([3 3], 4, 'InChannels', 2, 'Stride', [2 2], 'Padding', [1 1], 'Seed', 1);
            X = zeros(7, 7, 2, 3);
            Y = conv.forward(X);
            % Ho = floor((7+2-3)/2)+1 = 4
            testCase.verifySize(Y, [4 4 4 3]);
        end

        function testInChannelsInferredFromFirstForward(testCase)
            import OceanDL.Conv.Conv2D
            conv = Conv2D(3, 5, 'Seed', 7);
            X = randn(6, 6, 3, 1);
            Y = conv.forward(X);
            testCase.verifyEqual(conv.InChannels, 3);
            testCase.verifySize(Y, [4 4 5 1]);
            testCase.verifyFalse(any(isnan(Y(:))));
        end

        function testChannelMismatchErrors(testCase)
            import OceanDL.Conv.Conv2D
            conv = Conv2D(3, 2, 'InChannels', 2, 'Seed', 1);
            testCase.verifyError(@() conv.forward(randn(5, 5, 3, 1)), ...
                'OceanDL:Conv:Conv2D:ChannelMismatch');
        end

        function testBadWeightShapeErrors(testCase)
            import OceanDL.Conv.Conv2D
            testCase.verifyError(@() Conv2D([3 3], 2, 'Weight', randn(3, 3, 1, 5)), ...
                'OceanDL:Conv:Conv2D:BadWeightShape');
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Conv.Conv2D
            conv = Conv2D(3, 2, 'InChannels', 1, 'Seed', 1);
            testCase.verifyError(@() conv.backward(ones(2, 2, 2, 1)), ...
                'OceanDL:Conv:Conv2D:NoForwardCache');
        end

        function testGetWeightGradBeforeBackwardErrors(testCase)
            import OceanDL.Conv.Conv2D
            conv = Conv2D(3, 2, 'InChannels', 1, 'Seed', 1);
            conv.forward(randn(5, 5, 1, 1));
            testCase.verifyError(@() conv.getWeightGrad(), ...
                'OceanDL:Conv:Conv2D:NoBackwardCache');
        end

        function testFiniteDifferenceGradcheck(testCase)
            % Small multi-channel, multi-sample, strided+padded case.
            % loss = sum(Y.^2, 'all') => dLoss/dY = 2Y, so
            % analytic dX = conv.backward(2Y), and getWeightGrad() gives
            % dWeight/dBias -- compare every one against central finite
            % differences of the same scalar loss.
            import OceanDL.Conv.Conv2D

            rng(123);
            X = randn(5, 5, 2, 1);
            W = randn(3, 3, 2, 2) * 0.5;
            B = [0.1; -0.2];
            kernelSize = [3 3]; stride = [1 1]; padding = [1 1];

            lossOf = @(Xv, Wv, Bv) OceanTest.OceanDL.Conv.Conv2DTest.lossFn(Xv, Wv, Bv, kernelSize, stride, padding);

            conv = Conv2D(kernelSize, 2, 'InChannels', 2, 'Stride', stride, ...
                'Padding', padding, 'Weight', W, 'Bias', B);
            Y = conv.forward(X);
            dX = conv.backward(2 * Y);
            [dW, dB] = conv.getWeightGrad();

            eps_ = 1e-4;

            % --- dX gradcheck (every element -- tensor is small) ---
            dXnum = zeros(size(X));
            for k = 1:numel(X)
                Xp = X; Xp(k) = Xp(k) + eps_;
                Xm = X; Xm(k) = Xm(k) - eps_;
                dXnum(k) = (lossOf(Xp, W, B) - lossOf(Xm, W, B)) / (2 * eps_);
            end
            testCase.verifyEqual(dX, dXnum, 'AbsTol', 1e-3);

            % --- dWeight gradcheck ---
            dWnum = zeros(size(W));
            for k = 1:numel(W)
                Wp = W; Wp(k) = Wp(k) + eps_;
                Wm = W; Wm(k) = Wm(k) - eps_;
                dWnum(k) = (lossOf(X, Wp, B) - lossOf(X, Wm, B)) / (2 * eps_);
            end
            testCase.verifyEqual(dW, dWnum, 'AbsTol', 1e-3);

            % --- dBias gradcheck ---
            dBnum = zeros(size(B));
            for k = 1:numel(B)
                Bp = B; Bp(k) = Bp(k) + eps_;
                Bm = B; Bm(k) = Bm(k) - eps_;
                dBnum(k) = (lossOf(X, W, Bp) - lossOf(X, W, Bm)) / (2 * eps_);
            end
            testCase.verifyEqual(dB, dBnum, 'AbsTol', 1e-3);
        end

        function testOverlappingWindowsAccumulateInBackward(testCase)
            % Stride < KernelSize => overlapping windows; an input pixel
            % touched by multiple windows must SUM their gradients (same
            % lesson as MaxPool2D's overlapping-window accumulation test).
            import OceanDL.Conv.Conv2D
            W = ones(2, 2, 1, 1);
            conv = Conv2D([2 2], 1, 'InChannels', 1, 'HasBias', false, ...
                'Weight', W, 'Stride', [1 1]);
            X = ones(3, 3, 1, 1);
            conv.forward(X);              % Y is 2x2, every window sums to 4
            dX = conv.backward(ones(2, 2, 1, 1));
            % Center pixel (2,2) is covered by all 4 overlapping windows.
            testCase.verifyEqual(dX(2, 2), 4, 'AbsTol', 1e-12);
            % Corner pixel (1,1) is covered by exactly 1 window.
            testCase.verifyEqual(dX(1, 1), 1, 'AbsTol', 1e-12);
        end
    end

    methods (Static, Access = private)
        function L = lossFn(X, W, B, kernelSize, stride, padding)
            import OceanDL.Conv.Conv2D
            conv = Conv2D(kernelSize, size(W, 4), 'InChannels', size(W, 3), ...
                'Stride', stride, 'Padding', padding, 'Weight', W, 'Bias', B);
            Y = conv.forward(X);
            L = sum(Y.^2, 'all');
        end
    end
end
