classdef ExternalValidationTest < matlab.unittest.TestCase
    %EXTERNALVALIDATIONTEST Independent prescribed-randomness Python parity.
    methods (Test)
        function prescribedDynaMatchesPythonReference(testCase)
            folder = fullfile(fileparts(mfilename('fullpath')), 'Validation');
            addpath(folder);
            cleanup = onCleanup(@() rmpath(folder));
            report = validatePrescribedDyna(fullfile(folder, 'prescribed_dyna_expected.json'));
            testCase.verifyTrue(report.Passed);
            testCase.verifyEqual(report.NumUpdates, 8);
        end
    end
end
