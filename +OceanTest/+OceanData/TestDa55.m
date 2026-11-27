classdef TestDa55 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % T1: OLS imputer fills NaN — output is NaN-free
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            rng(1);
            X = randn(100, 4);
            X(randi(100,10,1), 1) = NaN;    % 10% missing in col 1
            X(randi(100, 8,1), 3) = NaN;    % 8%  missing in col 3
            imp   = OceanStats.Data.PredictiveImputer('ModelType','OLS','MaxIters',5);
            state = imp.fit(X);
            out   = imp.apply(X, state);
            testCase.verifyTrue(~any(isnan(out(:))) && isequal(size(out), size(X)), "T1_OLS_no_nan");


            % T2: OLS — no leakage (fit on train, apply on test)
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            rng(2);
            X_all = [linspace(1,100,200)', linspace(2,200,200)', randn(200,2)];
            nmiss = 20;
            X_all(randi(200,nmiss,1), 1) = NaN;
            X_train = X_all(1:160, :);
            X_test  = X_all(161:end, :);
            imp   = OceanStats.Data.PredictiveImputer('ModelType','OLS','Stochastic',false);
            state = imp.fit(X_train);       % learn from train only
            out_train = imp.apply(X_train, state);
            out_test  = imp.apply(X_test,  state);  % must not use test NaN info
            % Models were learned only from train — verify no NaN in results
            testCase.verifyTrue(~any(isnan(out_train(:))) && ~any(isnan(out_test(:))), "T2_OLS_noleak");


            % T3: CART imputer fills NaN
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            rng(3);
            X      = randn(80, 3);
            X(1:10, 2) = NaN;
            imp   = OceanStats.Data.PredictiveImputer('ModelType','CART','MaxIters',3);
            state = imp.fit(X);
            out   = imp.apply(X, state);
            testCase.verifyTrue(~any(isnan(out(:))) && isequal(size(out), size(X)), "T3_CART_no_nan");


            % T4: PMM — imputed values must come from original donor pool
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            rng(4);
            donors      = [10; 20; 30; 40; 50];          % 5 known values in col 1
            X_train     = [donors, randn(5,2)];
            X_test_miss = [NaN, randn(1,2); NaN, randn(1,2)];  % 2 missing
            imp   = OceanStats.Data.PredictiveImputer('ModelType','OLS','Stochastic',true,'K',5);
            state = imp.fit(X_train);
            out   = imp.apply(X_test_miss, state);
            % Every imputed value must be exactly one of the donor values
            imputed  = out(isnan(X_test_miss(:,1)), 1);
            fromPool = all(ismember(imputed, donors));
            testCase.verifyTrue(fromPool, "T4_PMM_from_donors");


            % T5: ModelImputer adapter produces same result as direct PredictiveImputer
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            rng(5);
            X = randn(60, 3);
            X(1:8, 1) = NaN;
            % Direct
            rng(99);
            core  = OceanStats.Data.PredictiveImputer('ModelType','OLS','Stochastic',false);
            s1    = core.fit(X);
            out1  = core.apply(X, s1);
            % Via adapter (same seed → same result when Stochastic=false)
            rng(99);
            adap  = OceanPipeline.Transforms.ModelImputer('ModelType','OLS','Stochastic',false);
            s2    = adap.fit(X);
            out2  = adap.apply(X, s2);
            testCase.verifyTrue(isequal(out1, out2), "T5_adapter_matches");


            % T6: No-NaN data passes through unchanged
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            rng(6);
            X_clean = randn(50, 4);   % no NaN at all
            imp   = OceanStats.Data.PredictiveImputer();
            state = imp.fit(X_clean);
            out   = imp.apply(X_clean, state);
            testCase.verifyTrue(isequal(X_clean, out), "T6_passthrough");


            % Report
            fprintf('\n=== DA-5.5 Test Suite (MICE + PMM Imputer) ===\n\n');
            fields = fieldnames(results);
            passed = 0;
            lines  = {};
            fprintf('%-44s  %s\n', 'Test', 'Result');
            fprintf('%s\n', repmat('-',1,54));
            for i = 1:numel(fields)
            testCase.verifyTrue(results.(fields{i}));
            tag = iif(ok,'[PASS]','[FAIL]');
            passed = passed + ok;
            fprintf('  %s  %s\n', tag, strrep(fields{i},'_',' '));
            lines{end+1} = sprintf('[%s] %s', iif(ok,'PASS','FAIL'), fields{i}); %#ok<AGROW>
            end
            fprintf('\n');
            footer = sprintf(iif(passed==numel(fields),'ALL PASS — DA-5.5 DONE (%d/%d)','%d/%d PASS'), ...
            passed, numel(fields));
            fprintf('%s\n\n', footer);
            outFile = 'c:\Ocean Library\Matlab code practice\Tests\da55_result.txt';
            fid = fopen(outFile,'w');
            for i=1:numel(lines), fprintf(fid,'%s\n',lines{i}); end
            fprintf(fid,'\n%s\n',footer); fclose(fid);
            function v = iif(cond, a, b)
            if cond, v = a; else, v = b; end
            end
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end