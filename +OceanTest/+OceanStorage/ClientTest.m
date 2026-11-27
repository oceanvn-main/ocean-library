classdef ClientTest < OceanTest.BaseTestCase
    methods (Test)
        function roundtripAndReopen(testCase)
            root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
            executable = string(getenv('OCEAN_STORAGE_EXE'));
            if strlength(executable) == 0
                executable = string(fullfile(root, '+OceanStorage', 'rust', 'target', 'debug', 'ocean-storage.exe'));
            end
            testCase.assertTrue(isfile(executable), 'Build the Rust engine before running client tests.');
            directory = string(tempname);
            mkdir(directory);
            cleanup = onCleanup(@() rmdir(directory, 's'));
            client = OceanStorage.Client(executable, directory);
            guard = onCleanup(@() delete(client));
            key = uint8(unicode2native(fileread(fullfile(root,'AGENTS.md')), 'UTF-8'));
            key = key(1:min(64,numel(key)));
            client.put(key, uint8.empty(1,0));
            [value, found] = client.get(key);
            testCase.verifyTrue(found);
            testCase.verifyEmpty(value);
            testCase.verifyError(@() client.put(zeros(1,129,'uint8'), uint8(1)), 'OceanStorage:Engine');
            client.close();
            client = OceanStorage.Client(executable, directory);
            guard2 = onCleanup(@() delete(client));
            [~, found] = client.get(key);
            testCase.verifyTrue(found);
            testCase.verifySize(client.scan(), [1,2]);
            testCase.verifyNotEmpty(client.inspect());
            client.remove(key);
            [~, found] = client.get(key);
            testCase.verifyFalse(found);
            client.close();
        end
    end
end
