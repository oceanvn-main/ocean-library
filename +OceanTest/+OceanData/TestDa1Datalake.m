classdef TestDa1Datalake < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            % Test 1: Constructor creates directory
            clc;
            DA = OceanData.DataArch;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake = DA.DataLake('rootDir', testDir);
            assert(isfolder(testDir), 'Root dir was not created.');
            ok('Constructor creates rootDir'); passed = passed + 1;
            % ================================================================== %
            % Test 2: put/get round-trip — matrix
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            X = randn(50, 4);
            lake.put('raw/matrix_test', X, struct('source', 'synthetic'));
            entry = lake.get('raw/matrix_test');
            assert(isequal(entry.data, X), 'Data mismatch after put/get.');
            ok('put/get round-trip (matrix)'); passed = passed + 1;
            % ================================================================== %
            % Test 3: meta preserved
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake.put('test/meta', magic(3), struct('author', 'ocean', 'version', 2));
            entry = lake.get('test/meta');
            assert(strcmp(entry.meta.author, 'ocean'), 'meta.author mismatch.');
            assert(entry.meta.version == 2, 'meta.version mismatch.');
            ok('meta struct preserved'); passed = passed + 1;
            % ================================================================== %
            % Test 4: timestamp recorded
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake.put('test/ts', eye(2), struct());
            entry = lake.get('test/ts');
            assert(isdatetime(entry.timestamp), 'timestamp is not datetime.');
            assert(entry.timestamp <= datetime('now'), 'timestamp is in the future.');
            ok('timestamp recorded'); passed = passed + 1;
            % ================================================================== %
            % Test 5: list() returns correct keys
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake2 = DA.DataLake('rootDir', fullfile(testDir, 'list_test'));
            lake2.put('a/b', 1, struct());
            lake2.put('a/c', 2, struct());
            lake2.put('d',   3, struct());
            keys = lake2.list();
            assert(numel(keys) == 3, sprintf('Expected 3 keys, got %d.', numel(keys)));
            % Check all keys present (order not guaranteed)
            keySet = sort(keys);
            assert(any(strcmp(keys, 'a/b')), '"a/b" not in list.');
            assert(any(strcmp(keys, 'a/c')), '"a/c" not in list.');
            assert(any(strcmp(keys, 'd')),   '"d" not in list.');
            ok('list() returns correct keys'); passed = passed + 1;
            % ================================================================== %
            % Test 6: exists() correct boolean
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake.put('exist_test/x', rand(3), struct());
            assert(lake.exists('exist_test/x'),     'exists() false for present key.');
            assert(~lake.exists('nonexistent/key'), 'exists() true for absent key.');
            ok('exists() correct boolean'); passed = passed + 1;
            % ================================================================== %
            % Test 7: delete() removes key, idempotent on missing
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake.put('del_test/z', ones(2), struct());
            assert(lake.exists('del_test/z'), 'Key should exist before delete.');
            lake.delete('del_test/z');
            assert(~lake.exists('del_test/z'), 'Key should not exist after delete.');
            % Idempotent — should not throw
            lake.delete('del_test/z');
            ok('delete() correct + idempotent'); passed = passed + 1;
            % ================================================================== %
            % Test 8: get() throws OceanStats:DataArch:DataLake:KeyNotFound
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            thrown = false;
            lake.get('this/key/does/not/exist');
            assert(thrown, 'Expected KeyNotFound error was not thrown.');
            ok('get() throws KeyNotFound for missing key'); passed = passed + 1;
            % ================================================================== %
            % Test 9: Overwrite replaces data
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            lake.put('overwrite/x', zeros(3), struct('v', 1));
            lake.put('overwrite/x', ones(3),  struct('v', 2));
            entry = lake.get('overwrite/x');
            assert(all(entry.data(:) == 1), 'Overwrite did not replace data.');
            assert(entry.meta.v == 2, 'Overwrite did not replace meta.');
            ok('put() overwrites existing key'); passed = passed + 1;
            % ================================================================== %
            % Test 10: Different data types (struct, cell)
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            s = struct('x', rand(4,2), 'y', rand(4,1), 'label', 'synthetic');
            lake.put('types/struct', s, struct());
            entry = lake.get('types/struct');
            assert(isstruct(entry.data), 'Struct type not preserved.');
            assert(isequal(entry.data.x, s.x), 'Struct field .x mismatch.');
            ok('put/get struct data type'); passed = passed + 1;
            % ================================================================== %
            % Cleanup
            clc;
            testDir = fullfile(tempdir, sprintf('ocean_datalake_test_%s', ...
            datestr(now, 'yyyymmdd_HHMMSS')));
            fprintf('\n=== DA-1 DataLake Tests ===\n\n');
            ok  = @(name) fprintf('  [PASS] %s\n', name);
            err = @(name, msg) fprintf('  [FAIL] %s — %s\n', name, msg);
            rmdir(testDir, 's');
            % ================================================================== %

        end
    end
end
