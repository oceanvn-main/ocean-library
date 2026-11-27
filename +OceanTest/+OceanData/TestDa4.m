classdef TestDa4 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % T1: CsvFileReader — correct shape and header
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            reader = OceanPipeline.Ingestion.CsvFileReader(csvPath);
            res    = reader.read();
            % 4 rows, 3 numeric cols (age, income, label); 'name' is text → dropped
            shapeOk  = size(res.data,1) == 4 && size(res.data,2) == 3;
            headerOk = numel(res.meta.header) == 3;
            metaOk   = res.meta.nSamples == 4 && res.meta.nFeatures == 3;
            testCase.verifyTrue(shapeOk && headerOk && metaOk, "T1_csv_shape");


            % T2: CsvFileReader — NaN for missing value (Charlie's income)
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            reader = OceanPipeline.Ingestion.CsvFileReader(csvPath);
            res    = reader.read();
            % income column is index 2, row 3 should be NaN
            incomCol = find(strcmp(res.meta.header, 'income'));
            testCase.verifyTrue(~isempty(incomCol) && isnan(res.data(3, incomCol)), "T2_csv_nan");


            % T3: MatFileReader — load by variable name
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            reader = OceanPipeline.Ingestion.MatFileReader(matPath, 'Variable','X_mat');
            res    = reader.read();
            results.T3_mat_named = isequal(res.data, double(magic(4))) && ...
            res.meta.nSamples == 4 && res.meta.nFeatures == 4;


            % T4: MatFileReader — auto-detect numeric variable
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            reader = OceanPipeline.Ingestion.MatFileReader(matPath);
            res    = reader.read();
            testCase.verifyTrue(size(res.data,1) == 4 && size(res.data,2) == 4, "T4_mat_auto");


            % T5: DataLakeReader — wraps an existing DataLake entry
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            lake   = OceanData.DataArch.DataLake('rootDir', fullfile(tmpDir,'lake'));
            X_lake = randn(50, 8);
            lake.put('raw/test/v1', X_lake, struct('source','synthetic'));
            reader = OceanPipeline.Ingestion.DataLakeReader(lake, 'raw/test/v1');
            res    = reader.read();
            results.T5_lake = isequal(res.data, X_lake) && ...
            res.meta.nSamples == 50 && res.meta.nFeatures == 8 && ...
            reader.isReady();


            % T6: SyntheticReader — generates correct shape via DataFactory
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            reader = OceanPipeline.Ingestion.SyntheticReader('spam');
            res    = reader.read();
            % spam dataset has features + label col appended → at least 1 col
            results.T6_synthetic = res.meta.nSamples > 0 && res.meta.nFeatures > 0 && ...
            isnumeric(res.data) && reader.isReady();


            % T7: All concrete readers return identical struct field names
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            readers = { ...
            OceanPipeline.Ingestion.CsvFileReader(csvPath), ...
            OceanPipeline.Ingestion.MatFileReader(matPath, 'Variable','X_mat') ...
            };
            requiredDataFields = {};   % result struct: data + meta
            requiredMetaFields = {'nSamples','nFeatures','source','header','timestamp'};
            allOk = true;
            for i = 1:numel(readers)
            res = readers{i}.read();
            if ~isfield(res, 'data') || ~isfield(res, 'meta')
            allOk = false; break;
            end
            for f = requiredMetaFields
            if ~isfield(res.meta, f{1})
            allOk = false; break;
            end
            end
            end
            testCase.verifyTrue(allOk, "T7_uniform_shape");


            % T8: StreamingReader — throws NotImplemented on read()
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            stub = OceanPipeline.Ingestion.StreamingReader('kafka://localhost:9092');
            stub.read();


            % Cleanup
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end


            % Report
            tmpDir   = fullfile(tempdir, 'test_da4');
            if isfolder(tmpDir), rmdir(tmpDir, 's'); end; mkdir(tmpDir);
            fprintf('\n=== DA-4 Test Suite ===\n\n');
            csvPath = fullfile(tmpDir, 'test.csv');
            fprintf(fid, 'age,income,label,name\n');
            fprintf(fid, '35,60000,1,Alice\n');
            fprintf(fid, '42,80000,0,Bob\n');
            fprintf(fid, '60,,1,Charlie\n');   % missing income → NaN
            fprintf(fid, '28,45000,0,Dana\n');
            fclose(fid);
            matPath  = fullfile(tmpDir, 'test.mat');
            X_mat    = magic(4);    % 4×4 numeric
            save(matPath, 'X_mat');
            fields = fieldnames(results);
            passed = 0;
            lines  = {};
            fprintf('\n%-40s  %s\n', 'Test', 'Result');
            fprintf('%s\n', repmat('-',1,50));
            for i = 1:numel(fields)
            testCase.verifyTrue(results.(fields{i}));
            tag = iif(ok, '[PASS]', '[FAIL]');
            passed = passed + ok;
            fprintf('  %s  %s\n', tag, strrep(fields{i},'_',' '));
            lines{end+1} = sprintf('[%s] %s', iif(ok,'PASS','FAIL'), fields{i}); %#ok<AGROW>
            end
            fprintf('\n');
            if passed == numel(fields)
            footer = sprintf('ALL PASS — DA-4 DONE (%d/%d)', passed, numel(fields));
            else
            footer = sprintf('%d/%d PASS', passed, numel(fields));
            end
            fprintf('%s\n\n', footer);
            outFile = 'c:\Ocean Library\Matlab code practice\Tests\da4_result.txt';
            fid = fopen(outFile, 'w');
            for i = 1:numel(lines), fprintf(fid, '%s\n', lines{i}); end
            fprintf(fid, '\n%s\n', footer);
            fclose(fid);
            function v = iif(cond, a, b)
            if cond, v = a; else, v = b; end
            end
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end