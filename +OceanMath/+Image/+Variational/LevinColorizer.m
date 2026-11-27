classdef LevinColorizer < handle
    % LEVINCOLORIZER  Optimization-based image colorization (Levin et al. 2004).
    %
    %   Implements Szeliski §4.2.4 "Application: Interactive Colorization".
    %   Refactored to integrate into the OceanMath Unified Regularization Architecture
    %   using OceanMath.Operators.DifferenceOperatorAffinity and IRobustPenalty.
    %
    %   Usage:
    %       colorizer = OceanMath.Image.Variational.LevinColorizer('Lambda', 1.0);
    %       I_rgb = colorizer.colorize(I_gray, scribbles_rgb, mask_binary);
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Levin, Lischinski & Weiss (2004), Szeliski (2021) §4.2.4

    properties
        % SIGMA  Luminance affinity decay factor (default 0.05).
        Sigma  (1,1) double {mustBePositive} = 0.05

        % LAMBDA  Smoothness regularization weight (default 1.0).
        Lambda (1,1) double {mustBeNonnegative} = 1.0

        % NEIGHBORHOOD  Neighbor graph connectivity (4 or 8, default 8).
        Neighborhood (1,1) double {mustBeMember(Neighborhood, [4, 8])} = 8

        % PENALTY  Robust penalty strategy (default QuadraticPenalty)
        Penalty OceanMath.Regularization.IRobustPenalty = OceanMath.Regularization.QuadraticPenalty()
    end

    methods
        function obj = LevinColorizer(options)
            % LEVINCOLORIZER Constructor.
            arguments
                options.Sigma        (1,1) double = 0.05
                options.Lambda       (1,1) double = 1.0
                options.Neighborhood (1,1) double = 8
                options.Penalty                   = OceanMath.Regularization.QuadraticPenalty()
            end
            obj.Sigma        = options.Sigma;
            obj.Lambda       = options.Lambda;
            obj.Neighborhood = options.Neighborhood;
            obj.Penalty      = options.Penalty;
        end

        function I_rgb = colorize(obj, I_gray, scribbles_rgb, mask_binary)
            % COLORIZE Interpolate color scribbles across grayscale image.
            [H, W] = size(I_gray);
            N = H * W;

            % Normalize data inputs to double [0, 1]
            I_gray = double(I_gray);
            if max(I_gray(:)) > 1.0
                I_gray = I_gray / 255.0;
            end
            
            scribbles_rgb = double(scribbles_rgb);
            if max(scribbles_rgb(:)) > 1.0
                scribbles_rgb = scribbles_rgb / 255.0;
            end

            % Convert scribble RGB colors to YUV space
            [~, U_scribble, V_scribble] = obj.rgb2yuv(scribbles_rgb);

            % Build sparse affinity difference operator L = I - W using DifferenceOperatorAffinity
            op_affinity = OceanMath.Operators.DifferenceOperatorAffinity(I_gray, ...
                'Sigma', obj.Sigma, ...
                'Neighborhood', obj.Neighborhood);
            L = op_affinity.buildMatrix([H, W]);

            % Diagonal mask matrix M and complement (I - M)
            mask_vec = double(mask_binary(:));
            M = spdiags(mask_vec, 0, N, N);
            I_minus_M = spdiags(1.0 - mask_vec, 0, N, N);

            % Assemble exact Levin et al. (2004) linear system:
            % At scribble pixels: U(r) = U_scribble(r)
            % At non-scribble pixels: U(r) = sum w_rs * U(s)
            LHS = I_minus_M * L + M;

            % Solve for U and V chrominance channels
            RHS_U = M * U_scribble(:);
            RHS_V = M * V_scribble(:);

            U_solved = LHS \ RHS_U;
            V_solved = LHS \ RHS_V;

            U_mat = reshape(U_solved, H, W);
            V_mat = reshape(V_solved, H, W);

            % Recombine original luminance Y = I_gray with solved U and V
            I_rgb = obj.yuv2rgb(I_gray, U_mat, V_mat);
        end

        function W_affinity = buildAffinityMatrix(obj, I_gray)
            % BUILDAFFINITYMATRIX Helper for backwards compatibility.
            [H, W] = size(I_gray);
            op_affinity = OceanMath.Operators.DifferenceOperatorAffinity(I_gray, ...
                'Sigma', obj.Sigma, ...
                'Neighborhood', obj.Neighborhood);
            L = op_affinity.buildMatrix([H, W]);
            W_affinity = speye(H*W) - L;
        end
    end

    methods (Static)
        function [Y, U, V] = rgb2yuv(I_rgb)
            % RGB2YUV Convert RGB tensor to YUV components.
            R = I_rgb(:, :, 1);
            G = I_rgb(:, :, 2);
            B = I_rgb(:, :, 3);

            Y = 0.299 * R + 0.587 * G + 0.114 * B;
            U = -0.147 * R - 0.289 * G + 0.436 * B;
            V = 0.615 * R - 0.515 * G - 0.100 * B;
        end

        function I_rgb = yuv2rgb(Y, U, V)
            % YUV2RGB Convert YUV components back to RGB tensor.
            R = Y + 1.140 * V;
            G = Y - 0.395 * U - 0.581 * V;
            B = Y + 2.032 * U;

            % Clip to valid RGB range [0, 1]
            R = max(0.0, min(1.0, R));
            G = max(0.0, min(1.0, G));
            B = max(0.0, min(1.0, B));

            I_rgb = cat(3, R, G, B);
        end
    end
end
