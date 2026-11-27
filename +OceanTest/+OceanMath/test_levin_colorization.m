classdef test_levin_colorization < OceanTest.BaseTestCase
    % TEST_LEVIN_COLORIZATION  Unit tests for Levin colorization under unified architecture.
    %
    %   Author: Ocean Library  |  2026-08-05

    methods (Test)
        function testAffinityOperatorRowStochasticity(testCase)
            import OceanMath.Operators.DifferenceOperatorAffinity

            I_gray = rand(20, 20);
            op_affinity = DifferenceOperatorAffinity(I_gray, 'Sigma', 0.05, 'Neighborhood', 8);
            L = op_affinity.buildMatrix([20, 20]);
            W_affinity = speye(400) - L;

            row_sums = sum(W_affinity, 2);
            testCase.verifyEqual(row_sums, ones(400, 1), 'AbsTol', 1e-10, ...
                'Affinity matrix rows must sum to 1.0.');
        end

        function testScribbleColorPreservation(testCase)
            import OceanMath.Image.Variational.LevinColorizer

            % 16x16 grayscale synthetic image
            I_gray = 0.5 * ones(16, 16);
            scribbles_rgb = repmat(I_gray, [1 1 3]);
            mask_binary = false(16, 16);

            % Set Red scribble at (4,4) and Blue scribble at (12,12)
            scribbles_rgb(4, 4, :) = [1, 0, 0];   mask_binary(4, 4) = true;
            scribbles_rgb(12, 12, :) = [0, 0, 1]; mask_binary(12, 12) = true;

            colorizer = LevinColorizer('Sigma', 0.05);
            I_colorized = colorizer.colorize(I_gray, scribbles_rgb, mask_binary);

            % Verify scribble positions maintain requested colors
            testCase.verifyEqual(squeeze(I_colorized(4, 4, :)), [1; 0; 0], 'AbsTol', 0.05, ...
                'Scribble pixel (4,4) must remain Red.');
            testCase.verifyEqual(squeeze(I_colorized(12, 12, :)), [0; 0; 1], 'AbsTol', 0.05, ...
                'Scribble pixel (12,12) must remain Blue.');
        end
    end
end
