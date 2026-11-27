classdef TestDa6 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % T1: Pipeline — reader + imputer + clipper → NaN-free output
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            rng(1);
            X = randn(100, 5);
            X(randi(100,8,1), 2) = NaN;
            X(randi(100,5,1), 4) = NaN;
            X([3,7], :) = X([3,7],:) * 50;    % add outliers
            imp   = OceanPipeline.Transforms.MissingValueImputer('Strategy','mean');
            clip  = OceanPipeline.Transforms.OutlierClipper('Mode','sigma','N',3);
            pipeline = OceanPipeline.Orchestration.DataPipeline(imp, clip);
            pipeline.fit(X);
            result = pipeline.run(X);
            results.T1_pipeline_clean = ...
            ~any(isnan(result.data(:)))  && ...
            isfield(result, 'data')      && ...
            isfield(result, 'meta')      && ...
            result.meta.nSamples  == 100 && ...
            result.meta.nFeatures == 5;


            % T2: Pipeline — fit on train, run on test (no leakage)
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            rng(2);
            X_all = randn(200, 4);
            X_all(randi(200,15,1), 1) = NaN;
            X_train = X_all(1:150, :);
            X_test  = X_all(151:end, :);
            imp2     = OceanPipeline.Transforms.MissingValueImputer('Strategy','median');
            pipeline2 = OceanPipeline.Orchestration.DataPipeline(imp2);
            pipeline2.fit(X_train);
            r_train = pipeline2.run(X_train);
            r_test  = pipeline2.run(X_test);
            results.T2_noleak = ...
            ~any(isnan(r_train.data(:))) && ...
            ~any(isnan(r_test.data(:)))  && ...
            r_train.meta.nSamples == 150 && ...
            r_test.meta.nSamples  == 50;


            % T3: Unfitted pipeline throws  meaningful error
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            imp3      = OceanPipeline.Transforms.MissingValueImputer();
            pipeline3 = OceanPipeline.Orchestration.DataPipeline(imp3);
            errThrown = false;
            pipeline3.run(randn(10,3));
            testCase.verifyTrue(errThrown, "T3_unfitted_error");


            % T4: MLDataAdapter holdout — shapes correct, no overlap
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            rng(4);
            N = 100; p = 6;
            X_raw = randn(N, p);
            Y     = randi(2, N, 1);
            imp4     = OceanPipeline.Transforms.MissingValueImputer();
            pl4      = OceanPipeline.Orchestration.DataPipeline(imp4);
            pl4.fit(X_raw);
            sp4      = OceanData.DataArch.DataSplitter('Strategy','holdout','Ratio',0.8);
            adapter4 = OceanPipeline.Orchestration.MLDataAdapter(pl4, sp4);
            pkg4     = adapter4.prepare(X_raw, Y);
            nTr  = size(pkg4.X_train, 1);
            nTe  = size(pkg4.X_test,  1);
            noOverlap = isempty(intersect(pkg4.trainIdx, pkg4.testIdx));
            covAll    = (nTr + nTe) == N;
            colOk     = size(pkg4.X_train, 2) == p && size(pkg4.X_test, 2) == p;
            results.T4_holdout_shapes = noOverlap && covAll && colOk && ...
            numel(pkg4.Y_train) == nTr && numel(pkg4.Y_test) == nTe;


            % T5: MLDataAdapter k-fold — 5 folds, each row covered exactly once
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            rng(5);
            N = 100; p = 4;
            X_raw = randn(N, p);
            Y     = randn(N, 1);
            clip5    = OceanPipeline.Transforms.OutlierClipper('Mode','iqr');
            pl5      = OceanPipeline.Orchestration.DataPipeline(clip5);
            pl5.fit(X_raw);
            sp5      = OceanData.DataArch.DataSplitter('Strategy','kfold','K',5);
            adapter5 = OceanPipeline.Orchestration.MLDataAdapter(pl5, sp5);
            pkg5     = adapter5.prepare(X_raw, Y);
            % Every row should appear in test set exactly once across all folds
            testRows = [];
            shapeOk  = true;
            for k = 1:5
            f = pkg5.folds{k};
            testRows = [testRows; f.testIdx(:)]; %#ok<AGROW>
            shapeOk = shapeOk && size(f.X_train,2) == p && size(f.X_test,2) == p;
            end
            results.T5_kfold_coverage = ...
            isequal(sort(testRows), (1:N)') && shapeOk && pkg5.K == 5;


            % T6: MLDataAdapter — FeatureStore cache hit on second call
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            rng(6);
            X_raw = randn(60, 3);
            tmpDir = tempname();
            mkdir(tmpDir);
            imp6  = OceanPipeline.Transforms.MissingValueImputer();
            pl6   = OceanPipeline.Orchestration.DataPipeline(imp6);
            pl6.fit(X_raw);
            fs   = OceanData.DataArch.FeatureStore('rootDir', tmpDir);
            sp6  = OceanData.DataArch.DataSplitter('Strategy','holdout');
            adp6 = OceanPipeline.Orchestration.MLDataAdapter(pl6, sp6, ...
            'FeatureStore', fs, 'CacheKey', 'test/cache/v1');
            pkg_first  = adp6.prepare(X_raw);    % computes and stores
            pkg_second = adp6.prepare(X_raw);    % should load from cache
            % Both must produce identical X_train (same data, same split seed)
            results.T6_cache_hit = ...
            isequal(pkg_first.X_train, pkg_second.X_train) && ...
            fs.exists('test/cache/v1');
            % Cleanup
            rmdir(tmpDir, 's');


            % Report
            fprintf('\n=== DA-6 Test Suite ===\n\n');
            fields = fieldnames(results);
            passed = 0;
            lines  = {};
            fprintf('%-46s  %s\n', 'Test', 'Result');
            fprintf('%s\n', repmat('-',1,56));
            for i = 1:numel(fields)
            testCase.verifyTrue(results.(fields{i}));
            tag = iif(ok,'[PASS]','[FAIL]');
            passed = passed + ok;
            fprintf('  %s  %s\n', tag, strrep(fields{i},'_',' '));
            lines{end+1} = sprintf('[%s] %s', iif(ok,'PASS','FAIL'), fields{i}); %#ok<AGROW>
            end
            fprintf('\n');
            footer = sprintf(iif(passed==numel(fields),'ALL PASS — DA-6 DONE (%d/%d)','%d/%d PASS'), ...
            passed, numel(fields));
            fprintf('%s\n\n', footer);
            outFile = 'c:\Ocean Library\Matlab code practice\Tests\da6_result.txt';
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