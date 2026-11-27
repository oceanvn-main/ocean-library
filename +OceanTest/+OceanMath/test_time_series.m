classdef test_time_series < OceanTest.BaseTestCase
    % TEST_TIME_SERIES Tests ARMA model definitions, simulation, ACF, PACF, and MA(1) duality.
    
    methods (Test)
        function testConstructorAndEquation(testCase)
            % Test default values
            model1 = OceanMath.TimeSeries.Models.ARMA();
            testCase.verifyEmpty(model1.ARCoeffs);
            testCase.verifyEmpty(model1.MACoeffs);
            testCase.verifyEqual(model1.Constant, 0.0);
            testCase.verifyEqual(model1.getEquation(), "Y(t) = e(t)");
            
            % Test custom values
            model2 = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', [0.8, -0.2], 'MACoeffs', 0.5, 'Constant', 1.5);
            testCase.verifyEqual(model2.ARCoeffs, [0.8, -0.2]);
            testCase.verifyEqual(model2.MACoeffs, 0.5);
            testCase.verifyEqual(model2.Constant, 1.5);
            testCase.verifyEqual(model2.getEquation(), "Y(t) = 1.50 + 0.80*Y(t-1) - 0.20*Y(t-2) + e(t) + 0.50*e(t-1)");
        end
        
        function testStationarityAndInvertibility(testCase)
            % White noise: stationary and invertible
            modelWN = OceanMath.TimeSeries.Models.ARMA();
            testCase.verifyTrue(modelWN.isStationary());
            testCase.verifyTrue(modelWN.isInvertible());
            
            % AR(1) with phi = 0.8: stationary
            modelAR1 = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', 0.8);
            testCase.verifyTrue(modelAR1.isStationary());
            testCase.verifyTrue(modelAR1.isInvertible());
            
            % AR(1) with phi = 1.0 (Random Walk): non-stationary
            modelRW = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', 1.0);
            testCase.verifyFalse(modelRW.isStationary());
            
            % AR(2) with phi = [1.2, -0.4]: roots of z^2 - 1.2z + 0.4 = 0
            % roots are 0.6 +/- 0.2i, magnitude is sqrt(0.36 + 0.04) = sqrt(0.4) < 1 -> stationary
            modelAR2 = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', [1.2, -0.4]);
            testCase.verifyTrue(modelAR2.isStationary());
            
            % AR(2) unstable: phi = [1.5, -0.6] -> roots are 0.75 +/- 0.19i (stable)
            % phi = [1.5, -0.56] -> roots are 0.75 +/- 0.03i (stable)
            % phi = [1.8, -0.9] -> roots are 0.9 +/- 0.3i, modulus is sqrt(0.81 + 0.09) = sqrt(0.9) < 1
            % phi = [2.0, -1.0] -> roots are 1.0, 1.0 -> non-stationary
            modelAR2Unstable = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', [2.0, -1.0]);
            testCase.verifyFalse(modelAR2Unstable.isStationary());
            
            % MA(1) with theta = 0.5: invertible
            modelMA1Invertible = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 0.5);
            testCase.verifyTrue(modelMA1Invertible.isInvertible());
            
            % MA(1) with theta = 2.0: non-invertible
            modelMA1NonInvertible = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 2.0);
            testCase.verifyFalse(modelMA1NonInvertible.isInvertible());
        end
        
        function testSimulation(testCase)
            model = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', 0.5, 'MACoeffs', 0.3, 'Constant', 2.0);
            N = 150;
            sigma = 1.2;
            
            y = model.simulate(N, sigma);
            
            % Check size
            testCase.verifyEqual(size(y), [N, 1]);
            
            % Check reproducibility with the same generator properties (internal filters)
            rng(100);
            y1 = model.simulate(N, sigma);
            rng(100);
            y2 = model.simulate(N, sigma);
            testCase.assertMatrixEquals(y1, y2, 1e-12);
        end
        
        function testTheoreticalACFAndPACF(testCase)
            % AR(1) theoretical values: rho_k = phi^k
            phi = 0.8;
            modelAR = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', phi);
            lags = 5;
            
            theoACF = modelAR.theoreticalACF(lags);
            expectedACF = phi .^ (0:lags);
            testCase.assertMatrixEquals(theoACF, expectedACF, 1e-12);
            
            % AR(1) PACF: phi_11 = phi, phi_kk = 0 for k > 1
            theoPACF = modelAR.theoreticalPACF(lags);
            expectedPACF = [phi, zeros(1, lags-1)];
            testCase.assertMatrixEquals(theoPACF, expectedPACF, 1e-9);
            
            % MA(1) theoretical values: rho_1 = theta / (1 + theta^2), others = 0
            theta = 0.5;
            modelMA = OceanMath.TimeSeries.Models.ARMA('MACoeffs', theta);
            
            theoACF_MA = modelMA.theoreticalACF(lags);
            expectedACF_MA = [1, theta / (1 + theta^2), zeros(1, lags-1)];
            testCase.assertMatrixEquals(theoACF_MA, expectedACF_MA, 1e-12);
        end
        
        function testMADuality(testCase)
            % Section 3.7: theta = 0.5 and theta = 2.0 must have same ACF
            modelA = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 0.5);
            modelB = OceanMath.TimeSeries.Models.ARMA('MACoeffs', 2.0);
            
            lags = 8;
            acfA = modelA.theoreticalACF(lags);
            acfB = modelB.theoreticalACF(lags);
            
            testCase.assertMatrixEquals(acfA, acfB, 1e-12);
            testCase.verifyEqual(acfA(2), 0.4, 'AbsTol', 1e-12);
        end
        
        function testEmpiricalEstimators(testCase)
            % Test empirical autocorrelation and partial autocorrelation functions
            rng(100);
            N = 500;
            y = randn(N, 1); % Independent white noise
            
            maxLag = 10;
            acf = OceanMath.TimeSeries.Diagnostics.autocorr(y, maxLag);
            pacf = OceanMath.TimeSeries.Diagnostics.parcorr(y, maxLag);
            
            testCase.verifyEqual(length(acf), maxLag + 1);
            testCase.verifyEqual(length(pacf), maxLag);
            
            % Lag 0 is always 1.0
            testCase.verifyEqual(acf(1), 1.0, 'AbsTol', 1e-12);
            
            % For independent random sequence, lag > 0 ACF/PACF should be close to 0
            % (95% bounds are +/- 1.96 / sqrt(500) = 0.087)
            testCase.verifyTrue(all(abs(acf(2:end)) < 0.15));
            testCase.verifyTrue(all(abs(pacf) < 0.15));
        end
        
        function testARConstructorAndSim(testCase)
            % Test default values
            model1 = OceanMath.TimeSeries.Models.AR();
            testCase.verifyEmpty(model1.ARCoeffs);
            testCase.verifyEqual(model1.Constant, 0.0);
            testCase.verifyEqual(model1.NoiseStd, 1.0);
            testCase.verifyEqual(model1.getEquation(), "Y(t) = e(t)");
            
            % Test custom values
            model2 = OceanMath.TimeSeries.Models.AR('ARCoeffs', [0.8, -0.2], 'Constant', 1.5, 'NoiseStd', 0.5);
            testCase.verifyEqual(model2.ARCoeffs, [0.8, -0.2]);
            testCase.verifyEqual(model2.Constant, 1.5);
            testCase.verifyEqual(model2.NoiseStd, 0.5);
            testCase.verifyEqual(model2.getEquation(), "Y(t) = 1.50 + 0.80*Y(t-1) - 0.20*Y(t-2) + e(t)");
            
            % Check stationarity checks
            testCase.verifyTrue(model2.isStationary());
            modelUnstable = OceanMath.TimeSeries.Models.AR('ARCoeffs', [1.2, 0.5]);
            testCase.verifyFalse(modelUnstable.isStationary());
            
            % Check simulation output dimensions
            y = model2.simulate(100, 0.5);
            testCase.verifyEqual(size(y), [100, 1]);
        end
        
        function testARFitting(testCase)
            % Simulate a known AR(1) process and fit it
            rng(100);
            trueModel = OceanMath.TimeSeries.Models.AR('ARCoeffs', 0.6, 'Constant', 0.0, 'NoiseStd', 1.0);
            y = trueModel.simulate(1000, 1.0);
            
            % Fit AR(1) model directly
            fittedModel = OceanMath.TimeSeries.Models.AR.fit(y, 'Order', 1);
            
            % Assert coefficients are close to true values (large sample property)
            testCase.verifyEqual(fittedModel.ARCoeffs(1), 0.6, 'AbsTol', 0.08);
            testCase.verifyEqual(fittedModel.Constant, 0.0, 'AbsTol', 0.15);
            testCase.verifyEqual(fittedModel.NoiseStd, 1.0, 'AbsTol', 0.1);
            
            % Fit using automatic order selection (should find p=1)
            modelAuto = OceanMath.TimeSeries.Models.AR.fit(y, 'MaxOrder', 5, 'Criterion', 'BIC');
            testCase.verifyEqual(length(modelAuto.ARCoeffs), 1);
        end
        
        function testARForecasting(testCase)
            % Create known AR(1) process
            model = OceanMath.TimeSeries.Models.AR('ARCoeffs', 0.8, 'Constant', 1.0, 'NoiseStd', 0.5);
            yHistory = [1.0; 2.0; 1.5; 3.0];
            
            % Forecast 5 steps ahead
            [yHat, bands] = model.forecast(yHistory, 5);
            
            % Assert sizes
            testCase.verifyEqual(size(yHat), [5, 1]);
            testCase.verifyEqual(size(bands), [5, 2]);
            
            % Step-by-step point forecast verification
            % h = 1: yHat(1) = 1.0 + 0.8 * 3.0 = 3.4
            testCase.verifyEqual(yHat(1), 3.4, 'AbsTol', 1e-12);
            % h = 2: yHat(2) = 1.0 + 0.8 * 3.4 = 3.72
            testCase.verifyEqual(yHat(2), 3.72, 'AbsTol', 1e-12);
            
            % Verify confidence bands (bands(:,1) < yHat < bands(:,2))
            testCase.verifyTrue(all(bands(:, 1) < yHat));
            testCase.verifyTrue(all(bands(:, 2) > yHat));
        end
    end
end
