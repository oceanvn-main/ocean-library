classdef ThinPlateSpline
    % THINPLATESPLINE  Radial Basis Function (RBF) landmark interpolator and warper.
    %
    %   ThinPlateSpline supports landmark-based image warping and coordinate
    %   interpolation using swappable RBF kernels, including Thin Plate Spline (TPS),
    %   Gaussian RBF, Multiquadric RBF, and Inverse Distance Weighting.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    properties
        W       % Weights matrix (N x 2)
        A       % Affine coefficients (3 x 2)
        PtsSrc  % Source control points (N x 2) - independent variables
        Kernel  % Concrete IRBFKernel strategy object
    end

    methods
        function obj = ThinPlateSpline(pts_src, pts_tgt, kernel)
            % THINPLATESPLINE  Fits a TPS/RBF model mapping pts_src -> pts_tgt.
            %
            %   obj = ThinPlateSpline(pts_src, pts_tgt, kernel)
            %
            %   Inputs:
            %     pts_src - N x 2 matrix of source landmark coordinates (X, Y)
            %     pts_tgt - N x 2 matrix of target landmark coordinates (X, Y)
            %     kernel  - Optional. Can be a string ('thinplate', 'gaussian',
            %               'multiquadric', 'inversedistance') or a concrete class
            %               inheriting from IRBFKernel. Defaults to 'thinplate'.

            arguments
                pts_src (:, 2) double
                pts_tgt (:, 2) double
                kernel = 'thinplate'
            end

            N = size(pts_src, 1);
            if N < 3
                error('ThinPlateSpline:TooFewPoints', 'At least 3 non-collinear control points are required.');
            end

            obj.PtsSrc = pts_src;

            % Resolve the kernel strategy
            import OceanMath.Image.Resampling.*
            if ischar(kernel) || isstring(kernel)
                switch lower(string(kernel))
                    case {'thinplate', 'tps'}
                        obj.Kernel = ThinPlateKernel();
                    case 'gaussian'
                        obj.Kernel = GaussianRBFKernel();
                    case {'multiquadric', 'mq'}
                        obj.Kernel = MultiquadricKernel();
                    case {'inversedistance', 'idw'}
                        obj.Kernel = InverseDistanceKernel();
                    otherwise
                        error('ThinPlateSpline:UnknownKernel', 'Unknown kernel type: %s', kernel);
                end
            elseif isa(kernel, 'OceanMath.Image.Resampling.IRBFKernel')
                obj.Kernel = kernel;
            else
                error('ThinPlateSpline:InvalidKernel', 'Kernel must be a valid string or inherit from IRBFKernel.');
            end

            % 1. Compute pairwise distance matrix K in pts_src
            dx = pts_src(:, 1) - pts_src(:, 1)';
            dy = pts_src(:, 2) - pts_src(:, 2)';
            r = sqrt(dx.^2 + dy.^2);
            K = obj.Kernel.evaluate(r);

            % 2. Construct design matrix P for affine terms
            P = [ones(N, 1), pts_src(:, 1), pts_src(:, 2)];

            % 3. Assemble full system matrix L and RHS
            L = [K, P;
                 P', zeros(3, 3)];

            RHS = [pts_tgt; zeros(3, 2)];

            % 4. Solve the linear system
            X = L \ RHS;

            obj.W = X(1:N, :);
            obj.A = X(N+1:end, :);
        end

        function pts_eval = evaluate(obj, pts_query)
            % EVALUATE  Evaluates the TPS/RBF model at query coordinates.
            %
            %   pts_eval = evaluate(obj, pts_query)
            %
            %   Inputs:
            %     pts_query - M x 2 matrix of query coordinates (X, Y)
            %
            %   Outputs:
            %     pts_eval  - M x 2 matrix of interpolated target coordinates

            arguments
                obj
                pts_query (:, 2) double
            end

            M = size(pts_query, 1);

            % 1. Compute distance matrix between query points and source control points (M x N)
            dx = pts_query(:, 1) - obj.PtsSrc(:, 1)';
            dy = pts_query(:, 2) - obj.PtsSrc(:, 2)';
            r = sqrt(dx.^2 + dy.^2);
            K_q = obj.Kernel.evaluate(r);

            % 2. Construct design matrix for affine terms
            P_q = [ones(M, 1), pts_query(:, 1), pts_query(:, 2)];

            % 3. Evaluate mapping function
            pts_eval = P_q * obj.A + K_q * obj.W;
        end

        function I_warped = warpImage(obj, I_src, out_size)
            % WARPIMAGE  Warps the source image using backward mapping.
            %
            %   I_warped = warpImage(obj, I_src, out_size)
            %
            %   Inputs:
            %     I_src    - Source image (H x W x C) to be warped
            %     out_size - Optional [H_tgt, W_tgt] vector specifying output size
            %                (defaults to source size)

            arguments
                obj
                I_src {mustBeNumeric}
                out_size (1, 2) double = [size(I_src, 1), size(I_src, 2)]
            end

            I_dbl = double(I_src);
            [H_src, W_src, C] = size(I_dbl, 1:3);
            H_tgt = out_size(1);
            W_tgt = out_size(2);

            % Create coordinate grid of target image pixels
            [X_tgt, Y_tgt] = meshgrid(1:W_tgt, 1:H_tgt);
            pts_tgt_grid = [X_tgt(:), Y_tgt(:)];

            % Evaluate TPS to map target pixel grid coordinates back to source space
            % Note: To do backward mapping, this TPS object must have been fitted
            % mapping target control points -> source control points!
            pts_src_mapped = obj.evaluate(pts_tgt_grid);

            u_src = pts_src_mapped(:, 1);
            v_src = pts_src_mapped(:, 2);

            % Bilinear Interpolation
            u0 = floor(u_src);
            v0 = floor(v_src);
            u1 = u0 + 1;
            v1 = v0 + 1;

            % Clamp to source image boundaries
            u0_c = max(1, min(W_src, u0));
            u1_c = max(1, min(W_src, u1));
            v0_c = max(1, min(H_src, v0));
            v1_c = max(1, min(H_src, v1));

            du = u_src - u0;
            dv = v_src - v0;

            w00 = (1 - du) .* (1 - dv);
            w01 = du .* (1 - dv);
            w10 = (1 - du) .* dv;
            w11 = du .* dv;

            % Linear indices
            idx00 = v0_c + (u0_c - 1) * H_src;
            idx01 = v0_c + (u1_c - 1) * H_src;
            idx10 = v1_c + (u0_c - 1) * H_src;
            idx11 = v1_c + (u1_c - 1) * H_src;

            % Reshape output
            I_warped = zeros(H_tgt, W_tgt, C);

            for c = 1:C
                Ic = I_dbl(:, :, c);
                val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                I_warped(:, :, c) = reshape(val, H_tgt, W_tgt);
            end

            % Cast back to original type if uint8
            if isa(I_src, 'uint8')
                I_warped = uint8(max(0, min(255, round(I_warped))));
            end
        end
    end
end
