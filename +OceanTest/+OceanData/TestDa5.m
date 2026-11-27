classdef TestDa5 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % T1: MissingValueImputer — mean fills NaN correctly
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            rng(1);
            X = [1 2; NaN 4; 5 NaN; 7 8];
            imp   = OceanPipeline.Transforms.MissingValueImputer('Strategy','mean');
            state = imp.fit(X);
            % Expected column means (ignoring NaN): col1=(1+5+7)/3=4.333, col2=(2+4+8)/3=4.667
            out   = imp.apply(X, state);
            results.T1_imputer_mean = abs(out(2,1) - mean([1,5,7])) < 1e-10 && ...
            abs(out(3,2) - mean([2,4,8])) < 1e-10 && ...
            ~any(isnan(out(:)));


            % T2: MissingValueImputer — no leakage (test uses train stats, not its own)
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            rng(2);
            X_train = [10 20; 30 40; 50 60];          % no NaN in train
            X_test  = [NaN 80; 90 NaN];               % NaN in test
            imp   = OceanPipeline.Transforms.MissingValueImputer('Strategy','mean');
            state = imp.fit(X_train);
            out_test = imp.apply(X_test, state);
            % Fill values must come from train, not test
            trainMean1 = mean([10;30;50]);   % = 30
            trainMean2 = mean([20;40;60]);   % = 40
            noLeak = abs(out_test(1,1) - trainMean1) < 1e-10 && ...
            abs(out_test(2,2) - trainMean2) < 1e-10;
            testCase.verifyTrue(noLeak && ~any(isnan(out_test(:))), "T2_imputer_noleak");


            % T3: OutlierClipper — sigma mode clips correctly
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            rng(3);
            X_train = randn(200, 3);
            X_test  = [X_train; 100 * ones(5, 3)];   % add extreme outliers
            clip  = OceanPipeline.Transforms.OutlierClipper('Mode','sigma','N',3);
            state = clip.fit(X_train);
            out   = clip.apply(X_test, state);
            % All values should be within [lower, upper]
            withinBounds = all(out(:) >= min(state.lower) - 1e-10) && ...
            all(out(:) <= max(state.upper) + 1e-10);
            results.T3_clipper_sigma = withinBounds && ...
            size(out,1) == size(X_test,1);  % no rows dropped


            % T4: OutlierClipper — IQR mode, bounds from training only
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            rng(4);
            X_train = randn(300, 2);
            X_test  = [randn(50,2); 999*ones(3,2)];
            clip  = OceanPipeline.Transforms.OutlierClipper('Mode','iqr');
            state = clip.fit(X_train);
            out   = clip.apply(X_test, state);
            clippedOk = all(out(:,1) <= state.upper(1) + 1e-10) && ...
            all(out(:,1) >= state.lower(1) - 1e-10);
            testCase.verifyTrue(clippedOk && size(out,1) == size(X_test,1), "T4_clipper_iqr");


            % T5: DataValidator — reports schema violation
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            % Build a minimal schema manually
            schema.ColumnNames = {'age','income'};
            schema.Nullable    = [false, true];
            schema.Min         = [0, NaN];
            schema.Max         = [120, NaN];
            reg = OceanData.DataArch.SchemaRegistry();
            reg.register('people', schema);
            val   = OceanPipeline.Transforms.DataValidator(reg, 'people');
            state = val.fit(zeros(1,2));   % stateless but must call fit
            % Valid data
            X_testCase.verifyTrue([25 50000; 40 80000]);
            [~, r1] = val.apply(X_ok, state);
            % Invalid: age=-5 (below Min=0)
            X_bad   = [-5 50000; 40 80000];
            [~, r2] = val.apply(X_bad, state);
            results.T5_validator = r1.isValid && ~r2.isValid && ...
            numel(r2.violations) >= 1;


            % T6: DataMasker — column drop removes correct columns
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            X5 = [1 2 3 4 5; 6 7 8 9 10];
            mask  = OceanPipeline.Transforms.DataMasker('Drop', [2, 4]);
            state = mask.fit(X5);
            out   = mask.apply(X5, state);
            % Should keep columns 1, 3, 5
            correctCols = isequal(out, X5(:, [1,3,5]));
            testCase.verifyTrue(correctCols && size(out,2) == 3, "T6_masker_drop");


            % T7: Chain — impute → clip → mask on single dataset
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            rng(7);
            N = 100; p = 4;
            X_base = randn(N, p);
            % Inject NaN and outlier
            X_dirty = X_base;
            X_dirty(1, 1) = NaN;
            X_dirty(2, 2) = 999;   % outlier
            [Xtr, Xte] = deal(X_dirty(1:80,:), X_dirty(81:end,:));
            % Step 1: Impute
            imp = OceanPipeline.Transforms.MissingValueImputer('Strategy','mean');
            s1  = imp.fit(Xtr);
            Xtr = imp.apply(Xtr, s1);   Xte = imp.apply(Xte, s1);
            % Step 2: Clip
            clip = OceanPipeline.Transforms.OutlierClipper('Mode','sigma','N',3);
            s2   = clip.fit(Xtr);
            Xtr  = clip.apply(Xtr, s2);   Xte = clip.apply(Xte, s2);
            % Step 3: Mask (drop last col)
            masker = OceanPipeline.Transforms.DataMasker('Drop', [p]);
            s3     = masker.fit(Xtr);
            Xtr    = masker.apply(Xtr, s3);   Xte = masker.apply(Xte, s3);
            chainOk = ~any(isnan(Xtr(:))) && ~any(isnan(Xte(:))) && ...
            size(Xtr, 2) == p-1 && size(Xte, 2) == p-1;
            testCase.verifyTrue(chainOk, "T7_chain");


            % Report
            fprintf('\n=== DA-5 Test Suite ===\n\n');
            fields = fieldnames(results);
            passed = 0;
            lines  = {};
            fprintf('\n%-42s  %s\n', 'Test', 'Result');
            fprintf('%s\n', repmat('-',1,52));
            for i = 1:numel(fields)
            testCase.verifyTrue(results.(fields{i}));
            tag = iif(ok,'[PASS]','[FAIL]');
            passed = passed + ok;
            fprintf('  %s  %s\n', tag, strrep(fields{i},'_',' '));
            lines{end+1} = sprintf('[%s] %s', iif(ok,'PASS','FAIL'), fields{i}); %#ok<AGROW>
            end
            fprintf('\n');
            footer = sprintf(iif(passed==numel(fields), 'ALL PASS — DA-5 DONE (%d/%d)', '%d/%d PASS'), ...
            passed, numel(fields));
            fprintf('%s\n\n', footer);
            outFile = 'c:\Ocean Library\Matlab code practice\Tests\da5_result.txt';
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