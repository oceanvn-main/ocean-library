classdef TestMcnemar < OceanTest.BaseTestCase
    methods (Test)
        function testT1SignificantDifferenceModel1MuchWorseThanModel2(testCase)
            % T1: Significant difference — model1 much worse than model2
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            % n=200 samples; model1 wrong where model2 right = 35, vice versa = 5
            % Expected: reject H0 (strong disagreement asymmetry)
            r = t.test(struct('n01', 35, 'n10', 5));
            testCase.verifyTrue(r.Reject && r.PValue < 0.05, "T1_reject_when_different");
        end

        function testT2NoSignificantDifferenceBalancedDisagreements(testCase)
            % T2: No significant difference — balanced disagreements
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            % n01=12, n10=11 → approximately symmetric
            r = t.test(struct('n01', 12, 'n10', 11));
            testCase.verifyTrue(~r.Reject && r.PValue > 0.05, "T2_failReject_when_same");
        end

        function testT3RawPredictionInterfaceProducesConsistentResultWithCountInterface(testCase)
            % T3: Raw prediction interface produces consistent result with count interface
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            N      = 200;
            Ytrue  = randi([0 1], N, 1);
            % Build Yhat1 / Yhat2 such that n01=35, n10=5 exactly
            Yhat1  = Ytrue;
            Yhat2  = Ytrue;
            idx    = randperm(N);
            Yhat1(idx(1:35))  = ~Ytrue(idx(1:35));   % model1 wrong on 35
            Yhat2(idx(36:40)) = ~Ytrue(idx(36:40));  % model2 wrong on 5
            data_raw   = struct('Yhat1', Yhat1, 'Yhat2', Yhat2, 'Ytrue', Ytrue);
            data_count = struct('n01', 35, 'n10', 5);
            r_raw   = t.test(data_raw);
            r_count = t.test(data_count);
            % Both should reject H0
            testCase.verifyTrue(r_raw.Reject == r_count.Reject, "T3_raw_vs_count_consistent");
        end

        function testT4ContinuityCorrectionSmallerStatisticLargerPValueMoreConservative(testCase)
            % T4: Continuity correction → smaller statistic, larger p-value (more conservative)
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            % IMPORTANT: counts must satisfy n01+n10 >= 25 to use chi-squared path (not exact binomial)
            r_corr   = t.test(struct('n01', 22, 'n10', 5, 'Correction', true));   % sum=27 >=25
            r_nocorr = t.test(struct('n01', 22, 'n10', 5, 'Correction', false));
            % With correction, statistic is strictly smaller (more conservative)
            results.T4_correction_conservative = r_corr.Statistic < r_nocorr.Statistic ...
            && r_corr.PValue  >= r_nocorr.PValue;
        end

        function testT5SmallSampleExactBinomialFiresNotesContainsWarning(testCase)
            % T5: Small sample → exact binomial fires, Notes contains warning
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            r = t.test(struct('n01', 8, 'n10', 1));  % n01+n10 = 9 < 25
            testCase.verifyTrue(contains(r.Notes, 'exact binomial'), "T5_small_sample_note");
        end

        function testT6PerfectlySymmetricPvalueShouldBeHighFailToReject(testCase)
            % T6: Perfectly symmetric → PValue should be high (fail to reject)
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            r = t.test(struct('n01', 15, 'n10', 15));
            testCase.verifyTrue(~r.Reject && r.PValue > 0.5, "T6_symmetric_high_pval");
        end

        function testT7ErrorWhenN01N100BothClassifiersAgreeOnEverything(testCase)
            % T7: Error when n01+n10==0 (both classifiers agree on everything)
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            thrown = false;
            t.test(struct('n01', 0, 'n10', 0));
            testCase.verifyTrue(thrown, "T7_all_agree_error");
        end

        function testT8ErrorOnDimensionMismatchInRawPredictions(testCase)
            % T8: Error on dimension mismatch in raw predictions
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            thrown = false;
            t.test(struct('Yhat1', [1;0;1], 'Yhat2', [1;0], 'Ytrue', [1;0;1]));
            testCase.verifyTrue(thrown, "T8_dim_mismatch_error");
        end

        function testT9TestresultRejectIsDerivedCorrectlyFromPvalueVsAlpha(testCase)
            % T9: TestResult.Reject is derived correctly from PValue vs Alpha
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            r1 = OceanStats.Inference.HypothesisTest.Base.TestResult('PValue', 0.03, 'Alpha', 0.05);
            r2 = OceanStats.Inference.HypothesisTest.Base.TestResult('PValue', 0.07, 'Alpha', 0.05);
            testCase.verifyTrue(r1.Reject && ~r2.Reject, "T9_reject_derived");
        end

        function testT10NameAndAssumptionsImplemented(testCase)
            % T10: name() and assumptions() implemented
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\mcnemar_test_result.txt';
            t       = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
            n = t.name();
            a = t.assumptions();
            testCase.verifyTrue(contains(n, 'McNemar') && numel(a) >= 3, "T10_name_assumptions");
            % ================================================================== %
        end

    end
end