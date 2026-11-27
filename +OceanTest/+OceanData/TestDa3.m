classdef TestDa3 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            % T1: FeatureStore put / get round-trip
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            fs  = OceanData.DataArch.FeatureStore('rootDir', tmpDir);
            X   = randn(100, 50);
            meta = struct('sigma', 3.5, 'transform', 'fourier');
            fs.put('fourier/spam/v1', X, meta);
            entry = fs.get('fourier/spam/v1');
            results.T1_put_get = isequal(entry.data, X) && ...
            entry.meta.sigma == 3.5 && ...
            strcmp(entry.meta.transform, 'fourier');
            % T2: Versioned keys (multi-level subdirectory path)
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            X2 = randn(80, 30);
            fs.put('pca/boston/v3', X2, struct('nComponents', 30));
            e2 = fs.get('pca/boston/v3');
            results.T2_versioned_keys = isequal(e2.data, X2) && ...
            e2.meta.nComponents == 30;
            % T3: getOrCompute — returns cached data on second call
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            rng(7);
            X3a  = randn(10, 5);              % result of "first compute"
            fn1  = @() X3a;                  % always returns X3a
            fn2  = @() randn(10, 5);         % would return different data if called
            e3a = fs.getOrCompute('test/cache/v1', fn1, struct('src','test'));
            % Second call: use fn2. If caching works, result == X3a, not fn2's output.
            e3b = fs.getOrCompute('test/cache/v1', fn2, struct('src','test'));
            results.T3_cache_hit = isequal(e3a.data, e3b.data) && ...
            isequal(e3a.data, X3a);
            % T4: list / exists / delete
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            fs2 = OceanData.DataArch.FeatureStore('rootDir', tmpDir);
            keys = fs2.list();
            % Should contain fourier/spam/v1, pca/boston/v3, test/cache/v1
            hasExpected = any(contains(keys, 'spam')) && any(contains(keys, 'boston'));
            exBefore = fs2.exists('pca/boston/v3');
            fs2.delete('pca/boston/v3');
            exAfter  = fs2.exists('pca/boston/v3');
            testCase.verifyTrue(hasExpected && exBefore && ~exAfter, "T4_list_exists_delete");
            % T5: DataSplitter holdout — train/test proportions
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            N  = 1000;
            X5 = randn(N, 5);
            sp = OceanData.DataArch.DataSplitter('Strategy','holdout','Ratio',0.8);
            s  = sp.split(X5);
            nTrain = sum(s.trainIdx);
            nTest  = sum(s.testIdx);
            results.T5_holdout_ratio = abs(nTrain/N - 0.8) < 0.02 && ...
            abs(nTest/N  - 0.2) < 0.02;
            % T6: DataSplitter holdout — zero overlap + full coverage
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            sp6 = OceanData.DataArch.DataSplitter('Strategy','holdout');
            s6  = sp6.split(randn(500, 3));
            noOverlap = ~any(s6.trainIdx & s6.testIdx);
            fullCover  = (sum(s6.trainIdx) + sum(s6.testIdx)) == 500;
            testCase.verifyTrue(noOverlap && fullCover, "T6_holdout_overlap");
            % T7: DataSplitter k-fold — zero overlap + full coverage per fold
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            N7   = 300;
            K7   = 5;
            sp7  = OceanData.DataArch.DataSplitter('Strategy','kfold','K',K7);
            folds = sp7.split(randn(N7, 4));
            allOk = true;
            for k = 1:K7
            noOv = ~any(folds{k}.trainIdx & folds{k}.testIdx);
            full = (sum(folds{k}.trainIdx) + sum(folds{k}.testIdx)) == N7;
            allOk = allOk && noOv && full;
            end
            % Verify all test folds together cover all N7 samples exactly once
            coverage = false(N7, 1);
            for k = 1:K7
            coverage = coverage | folds{k}.testIdx;
            end
            testCase.verifyTrue(allOk && all(coverage), "T7_kfold_overlap");
            % T8: DataSplitter stratified — class proportions preserved
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            N8  = 500;
            Y8  = [ones(400,1); zeros(100,1)];   % 80/20 class imbalance
            X8  = randn(N8, 5);
            sp8 = OceanData.DataArch.DataSplitter('Strategy','stratified','K',5,'Seed',0);
            f8  = sp8.split(X8, Y8);
            propOk = true;
            trueRatio = mean(Y8 == 1);
            for k = 1:5
            testY   = Y8(f8{k}.testIdx);
            foldRatio = mean(testY == 1);
            if abs(foldRatio - trueRatio) > 0.05   % allow 5% tolerance
            propOk = false;
            end
            end
            testCase.verifyTrue(propOk, "T8_stratified_props");
            % T9: DataSplitter seed reproducibility
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            X9  = randn(200, 3);
            sp9a = OceanData.DataArch.DataSplitter('Strategy','holdout','Seed',99);
            sp9b = OceanData.DataArch.DataSplitter('Strategy','holdout','Seed',99);
            s9a  = sp9a.split(X9);
            s9b  = sp9b.split(X9);
            testCase.verifyTrue(isequal(s9a.trainIdx, s9b.trainIdx), "T9_seed_repro");
            % Cleanup
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            % Report
            tmpDir  = fullfile(tempdir, 'test_da3_featurestore');
            if isfolder(tmpDir), rmdir(tmpDir,'s'); end
            fprintf('\n=== DA-3 Test Suite ===\n\n');
            fields = fieldnames(results);
            passed = 0;
            lines  = {};
            fprintf('\n%-40s  %s\n', 'Test', 'Result');
            fprintf('%s\n', repmat('-', 1, 50));
            for i = 1:numel(fields)
            testCase.verifyTrue(results.(fields{i}));
            tag = '[PASS]';
            if ~ok, tag = '[FAIL]'; end
            passed = passed + ok;
            lbl = strrep(fields{i}, '_', ' ');
            fprintf('  %s  %s\n', tag, lbl);
            lines{end+1} = sprintf('[%s] %s', iif(ok,'PASS','FAIL'), fields{i}); %#ok<AGROW>
            end
            fprintf('\n');
            if passed == numel(fields)
            footer = sprintf('ALL PASS — DA-3 DONE (%d/%d)', passed, numel(fields));
            else
            footer = sprintf('%d/%d PASS', passed, numel(fields));
            end
            fprintf('%s\n\n', footer);
            % Write result file
            outFile = 'c:\Ocean Library\Matlab code practice\Tests\da3_result.txt';
            fid = fopen(outFile, 'w');
            for i = 1:numel(lines), fprintf(fid, '%s\n', lines{i}); end
            fprintf(fid, '\n%s\n', footer);
            fclose(fid);
            function v = iif(cond, a, b)
            if cond, v = a; else, v = b; end
            end

        end
    end
end
