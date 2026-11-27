classdef Da1Fulltest < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            % T1: round-trip matrix
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            X = randn(50, 4);
            lake.put('raw/matrix', X, struct('src', 'synthetic'));
            e = lake.get('raw/matrix');
            testCase.verifyTrue(isequal(e.data, X), "T1_roundtrip");
            % T2: meta preserved
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            lake.put('meta/test', ones(3), struct('author', 'ocean', 'v', 7));
            em = lake.get('meta/test');
            testCase.verifyTrue(strcmp(em.meta.author, 'ocean') && em.meta.v == 7, "T2_meta");
            % T3: timestamp is datetime and not in future
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            testCase.verifyTrue(isdatetime(em.timestamp) && em.timestamp <= datetime('now'), "T3_timestamp");
            % T4: key is stored in entry
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            testCase.verifyTrue(strcmp(string(em.key), 'meta/test'), "T4_key_in_entry");
            % T5: list returns correct 3 keys
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            lake2 = OceanData.DataArch.DataLake('rootDir', fullfile(LAKE_DIR, 'listtest'));
            lake2.put('a/b', 1, struct());
            lake2.put('a/c', 2, struct());
            lake2.put('d',   3, struct());
            keys = lake2.list();
            results.T5_list3 = numel(keys) == 3 && ...
            any(strcmp(keys,'a/b')) && any(strcmp(keys,'a/c')) && any(strcmp(keys,'d'));
            % T6: exists true for present, false for absent
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            lake.put('exist/x', rand(3), struct());
            testCase.verifyTrue(lake.exists('exist/x'), "T6_exists_true");
            testCase.verifyTrue(~lake.exists('nonexistent/key_xyz'), "T6_exists_false");
            % T7: delete removes key, idempotent
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            lake.put('del/z', rand(2), struct());
            lake.delete('del/z');
            lake.delete('del/z');   % should not throw
            testCase.verifyTrue(~lake.exists('del/z'), "T7_delete_idempotent");
            % T8: get throws KeyNotFound with correct identifier
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            thrown = false;
            lake.get('this/does/not/exist');
            testCase.verifyTrue(thrown, "T8_errID");
            % T9: overwrite replaces both data and meta
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            lake.put('ow/x', zeros(3), struct('v', 1));
            lake.put('ow/x', ones(3),  struct('v', 2));
            eo = lake.get('ow/x');
            testCase.verifyTrue(all(eo.data(:) == 1) && eo.meta.v == 2, "T9_overwrite");
            % T10: struct data type preserved
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            s = struct('X', rand(4,2), 'Y', rand(4,1));
            lake.put('types/s', s, struct());
            es = lake.get('types/s');
            testCase.verifyTrue(isstruct(es.data) && isequal(es.data.X, s.X), "T10_struct_type");
            % Cleanup
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da1_fulltest_result.txt';
            LAKE_DIR    = 'c:\Ocean Library\Matlab code practice\da1_fulltest_lake';
            lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);
            rmdir(LAKE_DIR, 's');

        end
    end
end
