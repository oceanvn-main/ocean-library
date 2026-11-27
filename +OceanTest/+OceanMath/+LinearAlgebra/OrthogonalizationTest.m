classdef OrthogonalizationTest < OceanTest.BaseTestCase
    % ORTHOGONALIZATIONTEST Unit tests for OceanMath linearly algebra orthogonalizers.
    
    properties
        A
        H
        Y
        N
        M
        K
    end
    
    methods (TestClassSetup)
        function generateData(testCase)
            rng(42); % Reproducibility
            testCase.N = 30;
            testCase.M = 5;
            testCase.K = 2;
            
            testCase.A = randn(testCase.N, testCase.M);
            testCase.H = randn(testCase.N, testCase.K);
            testCase.Y = randn(testCase.N, 1);
        end
    end
    
    methods (Test)
        function testHouseholderReflectionDecompose(testCase)
            import OceanMath.LinearAlgebra.Orthogonalization.HouseholderReflection;
            strategy = HouseholderReflection();
            
            [Q, R] = strategy.decompose(testCase.A);
            
            % 1. Check dimensions
            testCase.verifyEqual(size(Q), [testCase.N, testCase.M], 'Q dimensions wrong');
            testCase.verifyEqual(size(R), [testCase.M, testCase.M], 'R dimensions wrong');
            
            % 2. Check Orthogonality (Q'Q = I)
            testCase.assertMatrixEquals(Q' * Q, eye(testCase.M));
            
            % 3. Check upper triangular
            testCase.assertMatrixEquals(tril(R, -1), zeros(testCase.M, testCase.M));
            
            % 4. Check Reconstruction
            testCase.assertMatrixEquals(Q * R, testCase.A);
        end
        
        function testHouseholderReflectionAddColumns(testCase)
            import OceanMath.LinearAlgebra.Orthogonalization.HouseholderReflection;
            strategy = HouseholderReflection();
            
            [Q1, R1] = strategy.decompose(testCase.A);
            [Q2, R2] = strategy.addColumns(Q1, R1, testCase.H);
            
            % 1. Check dimensions
            total_cols = testCase.M + testCase.K;
            testCase.verifyEqual(size(Q2), [testCase.N, total_cols]);
            testCase.verifyEqual(size(R2), [total_cols, total_cols]);
            
            % 2. Check Orthogonality
            testCase.assertMatrixEquals(Q2' * Q2, eye(total_cols));
            
            % 3. Check Reconstruction
            testCase.assertMatrixEquals(Q2 * R2, [testCase.A, testCase.H]);
        end
        
        function testModifiedGramSchmidtDecompose(testCase)
            import OceanMath.LinearAlgebra.Orthogonalization.ModifiedGramSchmidt;
            strategy = ModifiedGramSchmidt();
            
            [Q, R] = strategy.decompose(testCase.A);
            
            % 1. Check dimensions
            testCase.verifyEqual(size(Q), [testCase.N, testCase.M]);
            testCase.verifyEqual(size(R), [testCase.M, testCase.M]);
            
            % 2. Check Orthogonality
            testCase.assertMatrixEquals(Q' * Q, eye(testCase.M));
            
            % 3. Check upper triangular
            testCase.assertMatrixEquals(tril(R, -1), zeros(testCase.M, testCase.M));
            
            % 4. Check Reconstruction
            testCase.assertMatrixEquals(Q * R, testCase.A);
        end
        
        function testModifiedGramSchmidtAddColumns(testCase)
            import OceanMath.LinearAlgebra.Orthogonalization.ModifiedGramSchmidt;
            strategy = ModifiedGramSchmidt();
            
            [Q1, R1] = strategy.decompose(testCase.A);
            [Q2, R2] = strategy.addColumns(Q1, R1, testCase.H);
            
            total_cols = testCase.M + testCase.K;
            
            % 1. Check Orthogonality
            testCase.assertMatrixEquals(Q2' * Q2, eye(total_cols));
            
            % 2. Check Reconstruction
            testCase.assertMatrixEquals(Q2 * R2, [testCase.A, testCase.H]);
        end
        
        function testFastProject(testCase)
            import OceanMath.LinearAlgebra.Orthogonalization.ModifiedGramSchmidt;
            strategy = ModifiedGramSchmidt();
            
            [Q, ~] = strategy.decompose(testCase.A);
            h_new = testCase.H(:, 1);
            
            % Initial residual
            res_old = testCase.Y - Q * (Q' * testCase.Y);
            
            % Use fastProject
            [rss_fast, q_new_fast] = strategy.fastProject(Q, h_new, res_old);
            
            % Ground truth: decompose full matrix [A h_new] and find RSS
            A_full = [testCase.A, h_new];
            beta_full = A_full \ testCase.Y;
            res_full = testCase.Y - A_full * beta_full;
            rss_true = sum(res_full.^2);
            
            % Fast output should match true RSS closely
            testCase.verifyEqual(rss_fast, rss_true, 'AbsTol', 1e-8, 'Fast Project RSS mismatch');
            
            % Fast q_new should be orthogonal to Q
            testCase.assertMatrixEquals(Q' * q_new_fast, zeros(testCase.M, 1));
        end
        
        function testIncrementalQR(testCase)
            import OceanMath.LinearAlgebra.IncrementalQR;
            import OceanMath.LinearAlgebra.Orthogonalization.HouseholderReflection;
            
            iqr = IncrementalQR(testCase.A, HouseholderReflection());
            
            % Add columns and get beta
            [beta_inc, rss_inc, res_inc] = iqr.addColumns(testCase.H, testCase.Y);
            
            % Ground truth OLS
            X_full = [testCase.A, testCase.H];
            beta_true = X_full \ testCase.Y;
            res_true = testCase.Y - X_full * beta_true;
            rss_true = sum(res_true.^2);
            
            testCase.assertMatrixEquals(beta_inc, beta_true);
            testCase.verifyEqual(rss_inc, rss_true, 'AbsTol', 1e-8, 'Incremental QR RSS mismatch');
            
            % Test recalculate (should be identical since X doesn't change)
            [beta_recalc, rss_recalc, res_recalc] = iqr.recalculate(testCase.Y);
            testCase.assertMatrixEquals(beta_recalc, beta_true);
            testCase.verifyEqual(rss_recalc, rss_true, 'AbsTol', 1e-8, 'Recalculate RSS mismatch');
        end
    end
end
