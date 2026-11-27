classdef TapeTest < OceanTest.BaseTestCase
    % TAPETEST  Unit tests for OceanDL.Autograd.Tape.

    properties
        X3x3
    end

    methods (TestClassSetup)
        function buildInput(testCase)
            testCase.X3x3 = [0.10 0.40 -0.30; 0.75 -0.60 0.20; -0.15 0.55 0.90];
        end
    end

    methods (Test)
        function testSimpleSquareGradient(testCase)
            % y = x.^2  =>  dy/dx = 2x  (elementwise, seeded with ones)
            import OceanDL.Autograd.Tape
            X = testCase.X3x3;
            tape = Tape();
            xId = tape.watch(X, 'x');
            yId = tape.record('square', {xId}, X.^2, @(dY) {dY .* 2 .* X}, 'y');

            dX = tape.gradient(yId, xId);
            testCase.verifyEqual(dX, 2 * X, 'AbsTol', 1e-12);
        end

        function testNonCellInGivesNonCellOut(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch(5, 'x');
            yId = tape.record('square', {xId}, 25, @(dY) {dY * 10}, 'y');
            dX = tape.gradient(yId, xId);
            testCase.verifyEqual(dX, 10);
            testCase.verifyFalse(iscell(dX));
        end

        function testCellInGivesCellOut(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch(5, 'x');
            yId = tape.record('square', {xId}, 25, @(dY) {dY * 10}, 'y');
            grads = tape.gradient(yId, {xId});
            testCase.verifyTrue(iscell(grads));
            testCase.verifyEqual(grads{1}, 10);
        end

        function testDiamondAccumulatesBothPaths(testCase)
            % z = x + x  (x consumed by two "copy" branches that are both
            % summed) => dz/dx = 2, NOT 1 -- verifies gradient accumulation
            % at a node with multiple consumers.
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch(3, 'x');
            aId = tape.record('copyA', {xId}, 3, @(dY) {dY}, 'a');
            bId = tape.record('copyB', {xId}, 3, @(dY) {dY}, 'b');
            zId = tape.record('add', {aId, bId}, 6, @(dY) {dY, dY}, 'z');

            dX = tape.gradient(zId, xId);
            testCase.verifyEqual(dX, 2);
        end

        function testGradientWithRespectToIntermediateNode(testCase)
            % Grad-CAM-shaped query: gradient of a downstream node w.r.t. a
            % MIDDLE node (not the leaf input) must work directly.
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch(2, 'x');
            hId = tape.record('square', {xId}, 4, @(dY) {dY * 4}, 'h');       % h = x^2
            yId = tape.record('double', {hId}, 8, @(dY) {dY * 2}, 'y');       % y = 2h

            dH = tape.gradient(yId, hId);   % dy/dh = 2
            testCase.verifyEqual(dH, 2);
        end

        function testUnreachedSourceReturnsZero(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch([1 2 3], 'x');
            wId = tape.watch([4 5 6], 'w');   % never consumed by anything
            yId = tape.record('sum', {xId}, 6, @(dY) {dY * ones(1, 3)}, 'y');

            dW = tape.gradient(yId, wId);
            testCase.verifyEqual(dW, zeros(1, 3));
        end

        function testFiniteDifferenceGradcheck(testCase)
            % f(X) = sum(sin(X).^2, 'all') built as a 3-op chain on the tape.
            % Analytic gradient (elementwise) is sin(2X); verify it matches
            % central finite differences at every element.
            import OceanDL.Autograd.Tape
            X = testCase.X3x3;

            gradFn = @(Xv) OceanTest.OceanDL.Autograd.TapeTest.forwardAndGradient(Xv);
            [~, dX] = gradFn(X);

            eps_ = 1e-5;
            dXnum = zeros(size(X));
            for k = 1:numel(X)
                Xp = X; Xp(k) = Xp(k) + eps_;
                Xm = X; Xm(k) = Xm(k) - eps_;
                fp = gradFn(Xp);
                fm = gradFn(Xm);
                dXnum(k) = (fp - fm) / (2 * eps_);
            end

            testCase.verifyEqual(dX, dXnum, 'AbsTol', 1e-6);
            testCase.verifyEqual(dX, sin(2 * X), 'AbsTol', 1e-10);
        end

        function testRecordWithUnknownParentErrors(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            testCase.verifyError(...
                @() tape.record('op', {'ghost'}, 1, @(dY) {dY}), ...
                'OceanDL:Autograd:Tape:UnknownParent');
        end

        function testDuplicateIdErrors(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            tape.watch(1, 'x');
            testCase.verifyError(@() tape.watch(2, 'x'), ...
                'OceanDL:Autograd:Tape:DuplicateId');
        end

        function testUnknownNodeErrors(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            tape.watch(1, 'x');
            testCase.verifyError(@() tape.gradient('ghost', 'x'), ...
                'OceanDL:Autograd:Tape:UnknownNode');
        end

        function testResetClearsTape(testCase)
            import OceanDL.Autograd.Tape
            tape = Tape();
            tape.watch(1, 'x');
            testCase.verifyEqual(tape.numNodes(), 1);
            tape.reset();
            testCase.verifyEqual(tape.numNodes(), 0);
            testCase.verifyFalse(tape.hasNode('x'));
            tape.watch(2, 'x');   % must not error post-reset
            testCase.verifyEqual(tape.numNodes(), 1);
        end
    end

    methods (Static, Access = private)
        function [f, dX] = forwardAndGradient(X)
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch(X, 'x');
            sId = tape.record('sin', {xId}, sin(X), @(dY) {dY .* cos(X)}, 's');
            qId = tape.record('square', {sId}, sin(X).^2, @(dY) {dY .* 2 .* sin(X)}, 'q');
            f = sum(sin(X).^2, 'all');
            yId = tape.record('sum', {qId}, f, @(dY) {dY * ones(size(X))}, 'y');

            if nargout > 1
                dX = tape.gradient(yId, xId);
            end
        end
    end
end
