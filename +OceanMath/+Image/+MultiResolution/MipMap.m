classdef MipMap < handle
    % MIPMAP  Texture mapping mip-map implementation with trilinear filtering.
    %
    %   A MipMap represents a pre-filtered sequence of images at progressively
    %   lower resolutions, used to prevent aliasing and speed up rendering.
    %
    %   Constructor:
    %       mipmap = MipMap('Boundary', boundary, 'Kernel', kernel)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-15

    properties (Access = public)
        Levels     cell    % Cell array holding the image at each level (1 is original)
        NumLevels  double  % Total number of levels
        Boundary   char = 'replicate'
        Kernel             % Resolved IFilterKernel instance
    end

    methods
        function obj = MipMap(options)
            arguments
                options.Boundary char = 'replicate'
                options.Kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 5)
            end
            obj.Boundary = options.Boundary;
            obj.Kernel = OceanMath.Signal.Kernels.KernelFactory.resolve(options.Kernel);
        end

        function build(obj, I_in, num_levels)
            % BUILD  Constructs the mip-map levels using a Gaussian pyramid.
            arguments
                obj
                I_in {mustBeNumeric}
                num_levels double {mustBeInteger, mustBePositive}
            end
            obj.NumLevels = num_levels;
            obj.Levels = cell(1, num_levels);
            obj.Levels{1} = double(I_in);

            % Recursively reduce for subsequent levels using PyramidOperator
            for level = 2:num_levels
                obj.Levels{level} = OceanMath.Image.MultiResolution.PyramidOperator.reduce(...
                    obj.Levels{level-1}, obj.Boundary, obj.Kernel);
            end
        end

        function d = computeLevel(obj, J)
            % COMPUTELEVEL  Computes the fractional mip-map level from the 2x2 Jacobian.
            %   d = log2(max_singular_value(J))
            %   Clamped to [0, NumLevels - 1].
            arguments
                obj
                J (2,2) double
            end
            % Singular values of the Jacobian represent the scaling along the principal axes.
            % The maximum singular value corresponds to the maximum scale change.
            s = svd(J);
            if isempty(s)
                d = 0;
            else
                d = log2(max(s(1), eps));
            end
            d = max(0, min(obj.NumLevels - 1, d));
        end

        function val = sampleLevel(obj, lvl_idx, u, v)
            % SAMPLELEVEL  Samples a specific level of the mip-map using bilinear interpolation.
            %   u, v are defined in the coordinate space of the original image (Level 1).
            %   We scale u and v by 1 / 2^(lvl_idx - 1) to match the level's resolution.
            arguments
                obj
                lvl_idx (1,1) double {mustBeInteger, mustBePositive}
                u (1,1) double
                v (1,1) double
            end
            img = obj.Levels{lvl_idx};
            [H, W, C] = size(img);

            % Scale coordinates from Level 1 space to this level's space
            scale = 1 / (2^(lvl_idx - 1));
            u_scaled = (u - 0.5) * scale + 0.5;
            v_scaled = (v - 0.5) * scale + 0.5;

            % Bilinear interpolation
            u_scaled = max(1, min(W, u_scaled));
            v_scaled = max(1, min(H, v_scaled));

            u1 = floor(u_scaled);
            u2 = min(W, u1 + 1);
            v1 = floor(v_scaled);
            v2 = min(H, v1 + 1);

            wu = u_scaled - u1;
            wv = v_scaled - v1;

            val = (1 - wu) * (1 - wv) * img(v1, u1, :) + ...
                  wu * (1 - wv) * img(v1, u2, :) + ...
                  (1 - wu) * wv * img(v2, u1, :) + ...
                  wu * wv * img(v2, u2, :);
        end

        function val = sample(obj, u, v, J, mode)
            % SAMPLE  Samples the mip-map at texture coordinates (u, v) with Jacobian J.
            %   mode: 'Nearest', 'Floor', 'Ceil', or 'Trilinear'
            arguments
                obj
                u (1,1) double
                v (1,1) double
                J (2,2) double
                mode (1,1) string {mustBeMember(mode, ["Nearest", "Floor", "Ceil", "Trilinear"])} = "Trilinear"
            end

            d = obj.computeLevel(J);

            switch mode
                case 'Floor'
                    lvl = floor(d) + 1;
                    val = obj.sampleLevel(lvl, u, v);
                case 'Ceil'
                    lvl = min(obj.NumLevels, ceil(d) + 1);
                    val = obj.sampleLevel(lvl, u, v);
                case 'Nearest'
                    lvl = round(d) + 1;
                    val = obj.sampleLevel(lvl, u, v);
                case 'Trilinear'
                    d_low = floor(d) + 1;
                    d_high = min(obj.NumLevels, d_low + 1);
                    frac = d - floor(d);

                    val_low = obj.sampleLevel(d_low, u, v);
                    val_high = obj.sampleLevel(d_high, u, v);

                    val = (1 - frac) * val_low + frac * val_high;
            end
        end
    end
end
