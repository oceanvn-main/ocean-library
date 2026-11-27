classdef TestSplineStrategies < OceanTest.BaseTestCase
    methods (Test)
        function test1BackwardCompatNoArgConstructor(testCase)
            % 1. Backward compat: no-arg constructor
            clc;
            x = sort(rand(N,1));
            y = 2*x + 1 + sin(4*pi*x) + 0.3*randn(N,1);
            x_new = linspace(0.05, 0.95, 30)';
            import OceanStats.Selection.Strategies.SplineSmoother
            s0 = SplineSmoother();
            fprintf('1) Default ctor name: %s\n', s0.Name);
            m0 = s0.train(x, y);
            fprintf('   df = %.2f\n', m0.df);
        end

        function test2PsplineViaFactory(testCase)
            % 2. PSpline via factory
            clc;
            x = sort(rand(N,1));
            y = 2*x + 1 + sin(4*pi*x) + 0.3*randn(N,1);
            x_new = linspace(0.05, 0.95, 30)';
            import OceanStats.Selection.Strategies.SplineSmoother
            s1 = SplineSmoother.create('PSpline', 'TargetDF', 6);
            m1 = s1.train(x, y);
            yp1 = s1.predict(m1, x_new);
            fprintf('2) PSpline df=%.2f, pred range=[%.2f %.2f]\n', m1.df, min(yp1), max(yp1));
        end

        function test3NaturalcubicViaFactory(testCase)
            % 3. NaturalCubic via factory
            clc;
            x = sort(rand(N,1));
            y = 2*x + 1 + sin(4*pi*x) + 0.3*randn(N,1);
            x_new = linspace(0.05, 0.95, 30)';
            import OceanStats.Selection.Strategies.SplineSmoother
            s2 = SplineSmoother.create('NaturalCubic', 'DF', 8);
            m2 = s2.train(x, y);
            yp2 = s2.predict(m2, x_new);
            fprintf('3) NaturalCubic df=%.2f, pred range=[%.2f %.2f]\n', m2.df, min(yp2), max(yp2));
        end

        function test4NaturalcubicLinearReproductionTest(testCase)
            % 4. NaturalCubic linear reproduction test
            clc;
            x = sort(rand(N,1));
            y = 2*x + 1 + sin(4*pi*x) + 0.3*randn(N,1);
            x_new = linspace(0.05, 0.95, 30)';
            import OceanStats.Selection.Strategies.SplineSmoother
            X_d = [ones(N,1), x]; beta = X_d\y; y_hat = X_d*beta;
            diff_lin = max(abs(m2.S * y_hat - y_hat));
            fprintf('4) NaturalCubic linear repro error: %.2e  ', diff_lin);
            if diff_lin < 1e-7, fprintf('[PASS]\n'); else fprintf('[WARN]\n'); end
        end

        function test5RegressionSplineViaFactory(testCase)
            % 5. Regression Spline via factory
            clc;
            x = sort(rand(N,1));
            y = 2*x + 1 + sin(4*pi*x) + 0.3*randn(N,1);
            x_new = linspace(0.05, 0.95, 30)';
            import OceanStats.Selection.Strategies.SplineSmoother
            s3 = SplineSmoother.create('Regression', 'NumKnots', 5);
            m3 = s3.train(x, y);
            yp3 = s3.predict(m3, x_new);
            fprintf('5) RegressionSpline df=%d, pred range=[%.2f %.2f]\n', m3.df, min(yp3), max(yp3));
        end

        function test6ImodelstrategyContractCheck(testCase)
            % 6. IModelStrategy contract check
            clc;
            x = sort(rand(N,1));
            y = 2*x + 1 + sin(4*pi*x) + 0.3*randn(N,1);
            x_new = linspace(0.05, 0.95, 30)';
            import OceanStats.Selection.Strategies.SplineSmoother
            fprintf('6) isa IModelStrategy: %d\n', isa(s1, 'OceanStats.Base.IModelStrategy'));
            fprintf('\nAll checks done.\n');
        end

    end
end