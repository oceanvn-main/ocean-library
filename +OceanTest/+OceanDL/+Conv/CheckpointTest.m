classdef CheckpointTest < OceanTest.BaseTestCase
    % CHECKPOINTTEST  Unit tests for OceanDL.Conv.Checkpoint.

    properties
        TmpDir
    end

    methods (TestMethodSetup)
        function makeTmpDir(testCase)
            testCase.TmpDir = fullfile(tempdir, ['CheckpointTest_' char(java.util.UUID.randomUUID)]);
            mkdir(testCase.TmpDir);
        end
    end

    methods (TestMethodTeardown)
        function removeTmpDir(testCase)
            if isfolder(testCase.TmpDir)
                rmdir(testCase.TmpDir, 's');
            end
        end
    end

    methods (Test)
        function testMakeDummyCheckpointIsAValidForwardableNetwork(testCase)
            import OceanDL.Conv.Checkpoint
            import OceanDL.Conv.CNNEngine
            [spec, weights] = Checkpoint.makeDummyCheckpoint();
            eng = CNNEngine(spec, weights);
            Y = eng.predict(zeros(8, 8, 1, 2));
            testCase.verifySize(Y, [10 2]);
            testCase.verifyEqual(sum(Y, 1), ones(1, 2), 'AbsTol', 1e-10);
        end

        function testSaveLoadRoundTripPreservesArchitecture(testCase)
            import OceanDL.Conv.Checkpoint
            import OceanDL.Conv.CNNTopologySpec
            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 4, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(5), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 3);

            path = fullfile(testCase.TmpDir, 'mininet');
            Checkpoint.save(path, spec, struct());
            [spec2, ~] = Checkpoint.load(path);

            testCase.verifyEqual(spec2.InputChannels, spec.InputChannels);
            testCase.verifyEqual(spec2.LayerNames, spec.LayerNames);
            testCase.verifyEqual(numel(spec2.LayerSpecs), numel(spec.LayerSpecs));
            for i = 1:numel(spec.LayerSpecs)
                testCase.verifyEqual(spec2.LayerSpecs{i}.Type, spec.LayerSpecs{i}.Type);
            end
            testCase.verifyEqual(spec2.LayerSpecs{1}.KernelSize, spec.LayerSpecs{1}.KernelSize);
            testCase.verifyEqual(spec2.LayerSpecs{1}.OutChannels, spec.LayerSpecs{1}.OutChannels);
        end

        function testSaveLoadRoundTripPreservesWeightsNumerically(testCase)
            import OceanDL.Conv.Checkpoint
            import OceanDL.Conv.CNNEngine
            [spec, weights, imageSize] = Checkpoint.makeDummyCheckpoint();
            path = fullfile(testCase.TmpDir, 'dummy');
            Checkpoint.save(path, spec, weights, imageSize);
            [spec2, weights2, imageSize2] = Checkpoint.load(path);

            testCase.verifyEqual(imageSize2, imageSize);
            eng1 = CNNEngine(spec, weights);
            eng2 = CNNEngine(spec2, weights2);
            X = randn(imageSize(1), imageSize(2), 1, 1);
            Y1 = eng1.predict(X);
            Y2 = eng2.predict(X);
            testCase.verifyEqual(Y2, Y1, 'AbsTol', 1e-10);
        end

        function testImageSizeOmittedWhenNotProvided(testCase)
            import OceanDL.Conv.Checkpoint
            import OceanDL.Conv.CNNTopologySpec
            spec = CNNTopologySpec.make({CNNTopologySpec.conv2d(3, 4, 'Padding', [1 1])}, 'InputChannels', 1);
            path = fullfile(testCase.TmpDir, 'noSize');
            Checkpoint.save(path, spec, struct());
            [~, ~, imageSize] = Checkpoint.load(path);
            testCase.verifyEmpty(imageSize);
        end

        function testLoadMissingFileErrors(testCase)
            import OceanDL.Conv.Checkpoint
            testCase.verifyError(@() Checkpoint.load(fullfile(testCase.TmpDir, 'nope')), ...
                'OceanDL:Conv:Checkpoint:NotFound');
        end

        function testMakeDummyCheckpointSavesToDiskWhenPathGiven(testCase)
            import OceanDL.Conv.Checkpoint
            path = fullfile(testCase.TmpDir, 'dummy2');
            Checkpoint.makeDummyCheckpoint(path);
            testCase.verifyTrue(isfile([path '.json']));
            testCase.verifyTrue(isfile([path '.mat']));
        end
    end
end
