classdef PyramidOperator
    % PYRAMIDOPERATOR  Utility containing static decimation and interpolation operations.
    %
    %   This class implements the mathematical primitives for Gaussian reduction
    %   and expansion completely from scratch (without using imfilter). It supports
    %   both raw numeric arrays/vectors and polymorphic IFilterKernel objects.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    methods (Static)
        function I_red = reduce(I, boundary, kernel, options)
            % REDUCE  Applies Gaussian blur and decimates by a factor of 2.
            %
            %   I_red = reduce(I, boundary, kernel)
            %   I_red = reduce(I, boundary, kernel, BypassFilter=true)
            %
            %   Input:
            %       I             - Input image (2D or 3D, double)
            %       boundary      - Padding boundary mode (default 'replicate')
            %       kernel        - An IFilterKernel, a numeric matrix, or a string name.
            %                       Defaults to a 5x5 binomial GaussianKernel.
            %       BypassFilter  - (optional, logical) If true, skip the low-pass filter
            %                       and decimate directly by subsampling every 2nd pixel.
            %                       This demonstrates aliasing (Moire) artifacts.
            
            arguments
                I {mustBeNumeric}
                boundary char = 'replicate'
                kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 5)
                options.BypassFilter logical = false
            end
            
            % Cast to double for high-precision operations
            I_dbl = double(I);
            [H, W, ~] = size(I_dbl);
            H_out = ceil(H / 2);
            W_out = ceil(W / 2);
            
            % --- BYPASS: Direct subsampling without pre-filtering (demonstrates aliasing) ---
            if options.BypassFilter
                I_red = I_dbl(1:2:end, 1:2:end, :);
                % Crop to expected output dimensions in case of odd input
                I_red = I_red(1:H_out, 1:W_out, :);
                return;
            end
            
            % --- NORMAL: Blur then decimate (anti-aliased Gaussian reduction) ---
            % Resolve kernel object using the factory
            kernel_obj = OceanMath.Signal.Kernels.KernelFactory.resolve(kernel);
            
            % 1. Filter and decimate
            if kernel_obj.isSeparable()
                [h, v] = kernel_obj.get1DDecomposition();
                
                % Horizontal 1D pass with column downsampling:
                Kh = length(h);  % length of horizontal 1‑D filter
                pad_cols = ceil((Kh - 1) / 2);
                I_pad_h = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [0, pad_cols], boundary);
                I_h_down = zeros(H, W_out, size(I_dbl,3));
                for m = 1:Kh
                    I_h_down = I_h_down + h(m) * I_pad_h(:, m : 2 : (m + 2*W_out - 2), :);
                end
                
                % Vertical 1D pass with row downsampling:
                Kv = length(v);
                pad_rows = ceil((Kv - 1) / 2);
                I_pad_v = OceanMath.Image.Pixels.BoundaryPadder.pad(I_h_down, [pad_rows, 0], boundary);
                I_red = zeros(H_out, W_out, size(I_dbl,3));
                for m = 1:Kv
                    I_red = I_red + v(m) * I_pad_v(m : 2 : (m + 2*H_out - 2), :, :);
                end
            else
                % Non-separable 2D filter with strided downsampling
                K = kernel_obj.getMatrix();
                [Kh, Kw] = size(K);
                
                pad_rows = ceil((Kh - 1) / 2);
                pad_cols = ceil((Kw - 1) / 2);
                
                I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], boundary);
                
                I_red = zeros(H_out, W_out, size(I_dbl,3));
                for r = 1:Kh
                    for c = 1:Kw
                        I_red = I_red + K(r, c) * I_pad(r : 2 : (r + 2*H_out - 2), c : 2 : (c + 2*W_out - 2), :);
                    end
                end
            end
        end
        
        function I_exp = expand(I, target_size, boundary, kernel, options)
            % EXPAND  Upsamples by a factor of 2 and interpolates.
            %
            %   Uses polyphase decomposition for efficiency: instead of
            %   upsampling (zero-stuffing) to full size then filtering,
            %   we decompose the filter into even/odd polyphase sub-filters
            %   and convolve the *small* input directly, then interleave.
            %   This avoids ~50% of wasted multiply-by-zero operations.
            %
            %   I_exp = expand(I, target_size, boundary, kernel)
            %   I_exp = expand(I, target_size, boundary, kernel, BypassFilter=true)
            %
            %   Input:
            %       I             - Input image (2D or 3D, double)
            %       target_size   - Optional [H, W, C] vector specifying exact expanded size.
            %                       Allows handling odd dimension bounds correctly.
            %       boundary      - Padding boundary mode (default 'replicate')
            %       kernel        - An IFilterKernel, a numeric matrix, or a string name.
            %                       Defaults to a 5x5 binomial GaussianKernel.
            %       BypassFilter  - (optional, logical) If true, use nearest-neighbor
            %                       replication instead of interpolated polyphase expansion.
            
            arguments
                I {mustBeNumeric}
                target_size double = []
                boundary char = 'replicate'
                kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 5)
                options.BypassFilter logical = false
            end
            
            % Cast to double for high-precision operations
            I_dbl = double(I);
            [H_in, W_in, C_in] = size(I_dbl);
            
            % Compute target dimensions if not provided
            if isempty(target_size)
                target_size = [2 * H_in, 2 * W_in, C_in];
            end
            
            H_target = target_size(1);
            W_target = target_size(2);
            C_target = C_in;
            if length(target_size) >= 3
                C_target = target_size(3);
            end
            
            % --- BYPASS: Nearest-neighbor replication (no interpolation) ---
            if options.BypassFilter
                % Replicate each pixel to a 2x2 block, then crop to target
                I_rep = repelem(I_dbl, 2, 2, 1);
                I_exp = I_rep(1:H_target, 1:W_target, :);
                return;
            end
            
            % --- NORMAL: Polyphase interpolation ---
            % Resolve kernel object using the factory
            kernel_obj = OceanMath.Signal.Kernels.KernelFactory.resolve(kernel);
            
            % ---- Polyphase Expansion ----
            %
            % For a 1D filter h of length K, the polyphase components are:
            %   h_e[k] = h[2k+1]   (even output positions: y[2n])
            %   h_o[k] = h[2k+2]   (odd  output positions: y[2n+1])
            %
            % The output is:
            %   y[2n]   = sum_k  h_e[k] * x[n - k]
            %   y[2n+1] = sum_k  h_o[k] * x[n - k]
            %
            % We convolve the SMALL input x (size H_in or W_in) directly,
            % then interleave into the full-size output, cutting computation
            % by roughly a factor of 2 per separable pass.
            
            if kernel_obj.isSeparable()
                [h, v] = kernel_obj.get1DDecomposition();
                % Scale by 2 for each 1D component (sum becomes 4.0 in 2D to preserve energy)
                w_h = 2.0 * h;
                w_v = 2.0 * v;
                
                % ------- Horizontal polyphase pass -------
                % Decompose w_h into even and odd polyphase sub-filters.
                h_e = w_h(1:2:end);  % coeffs for even output columns
                h_o = w_h(2:2:end);  % coeffs for odd  output columns
                
                K_e = length(h_e);
                K_o = length(h_o);
                
                % Number of output columns at even and odd positions
                N_even_cols = ceil(W_target / 2);   % same as W_in (or W_in when target matches)
                N_odd_cols  = W_target - N_even_cols;
                
                % Pad the small input for horizontal filtering
                % For the even sub-filter of length K_e, half-width is (K_e-1)/2.
                % For odd-length K_e, this is integer; for even-length, we pad
                % ceil and floor to cover the support.
                pad_e = floor(K_e / 2);
                pad_o = floor(K_o / 2);
                max_pad_cols = max(pad_e, pad_o);
                
                I_padH = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [0, max_pad_cols], boundary);
                
                % Even output columns (y[:, 2n, :])
                I_even_cols = zeros(H_in, N_even_cols, C_target);
                for m = 1:K_e
                    col_start = (max_pad_cols - pad_e) + m;
                    I_even_cols = I_even_cols + h_e(m) * I_padH(:, col_start : col_start + N_even_cols - 1, :);
                end
                
                % Odd output columns (y[:, 2n+1, :])
                I_odd_cols = zeros(H_in, N_odd_cols, C_target);
                for m = 1:K_o
                    col_start = (max_pad_cols - pad_o) + m;
                    I_odd_cols = I_odd_cols + h_o(m) * I_padH(:, col_start : col_start + N_odd_cols - 1, :);
                end
                
                % Interleave horizontally
                I_h_exp = zeros(H_in, W_target, C_target);
                I_h_exp(:, 1:2:W_target, :) = I_even_cols;
                I_h_exp(:, 2:2:W_target, :) = I_odd_cols;
                
                % ------- Vertical polyphase pass -------
                % Decompose w_v into even and odd polyphase sub-filters.
                v_e = w_v(1:2:end);  % coeffs for even output rows
                v_o = w_v(2:2:end);  % coeffs for odd  output rows
                
                K_ve = length(v_e);
                K_vo = length(v_o);
                
                N_even_rows = ceil(H_target / 2);
                N_odd_rows  = H_target - N_even_rows;
                
                pad_ve = floor(K_ve / 2);
                pad_vo = floor(K_vo / 2);
                max_pad_rows = max(pad_ve, pad_vo);
                
                I_padV = OceanMath.Image.Pixels.BoundaryPadder.pad(I_h_exp, [max_pad_rows, 0], boundary);
                
                % Even output rows (y[2n, :, :])
                I_even_rows = zeros(N_even_rows, W_target, C_target);
                for m = 1:K_ve
                    row_start = (max_pad_rows - pad_ve) + m;
                    I_even_rows = I_even_rows + v_e(m) * I_padV(row_start : row_start + N_even_rows - 1, :, :);
                end
                
                % Odd output rows (y[2n+1, :, :])
                I_odd_rows = zeros(N_odd_rows, W_target, C_target);
                for m = 1:K_vo
                    row_start = (max_pad_rows - pad_vo) + m;
                    I_odd_rows = I_odd_rows + v_o(m) * I_padV(row_start : row_start + N_odd_rows - 1, :, :);
                end
                
                % Interleave vertically
                I_exp = zeros(H_target, W_target, C_target);
                I_exp(1:2:H_target, :, :) = I_even_rows;
                I_exp(2:2:H_target, :, :) = I_odd_rows;
                
            else
                % ---- Non-separable polyphase 2D expansion ----
                %
                % Decompose 2D kernel K into four polyphase sub-kernels:
                %   K_ee = K(1:2:end, 1:2:end)  -> output at (2r,   2c)
                %   K_eo = K(1:2:end, 2:2:end)  -> output at (2r,   2c+1)
                %   K_oe = K(2:2:end, 1:2:end)  -> output at (2r+1, 2c)
                %   K_oo = K(2:2:end, 2:2:end)  -> output at (2r+1, 2c+1)
                %
                % Each sub-kernel is applied to the small input directly,
                % and the results are placed at the corresponding grid positions.
                
                W_mat = kernel_obj.getMatrix();
                w_2d = 4.0 * W_mat;
                
                K_ee = w_2d(1:2:end, 1:2:end);
                K_eo = w_2d(1:2:end, 2:2:end);
                K_oe = w_2d(2:2:end, 1:2:end);
                K_oo = w_2d(2:2:end, 2:2:end);
                
                N_even_rows = ceil(H_target / 2);
                N_odd_rows  = H_target - N_even_rows;
                N_even_cols = ceil(W_target / 2);
                N_odd_cols  = W_target - N_even_cols;
                
                % Determine padding needed for each sub-kernel
                [Kr_ee, Kc_ee] = size(K_ee);
                [Kr_eo, Kc_eo] = size(K_eo);
                [Kr_oe, Kc_oe] = size(K_oe);
                [Kr_oo, Kc_oo] = size(K_oo);
                
                max_pad_rows = max([floor(Kr_ee/2), floor(Kr_eo/2), ...
                                    floor(Kr_oe/2), floor(Kr_oo/2)]);
                max_pad_cols = max([floor(Kc_ee/2), floor(Kc_eo/2), ...
                                    floor(Kc_oe/2), floor(Kc_oo/2)]);
                
                I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [max_pad_rows, max_pad_cols], boundary);
                
                % Helper: apply sub-kernel to padded input and extract N_r x N_c output
                applySubKernel = @(K_sub, N_r, N_c, pr, pc) ...
                    OceanMath.Image.MultiResolution.PyramidOperator.applyPolyphaseSubKernel_( ...
                        I_pad, K_sub, N_r, N_c, max_pad_rows, max_pad_cols, pr, pc);
                
                I_exp = zeros(H_target, W_target, C_target);
                I_exp(1:2:H_target, 1:2:W_target, :) = applySubKernel(K_ee, N_even_rows, N_even_cols, floor(Kr_ee/2), floor(Kc_ee/2));
                I_exp(1:2:H_target, 2:2:W_target, :) = applySubKernel(K_eo, N_even_rows, N_odd_cols,  floor(Kr_eo/2), floor(Kc_eo/2));
                I_exp(2:2:H_target, 1:2:W_target, :) = applySubKernel(K_oe, N_odd_rows,  N_even_cols, floor(Kr_oe/2), floor(Kc_oe/2));
                I_exp(2:2:H_target, 2:2:W_target, :) = applySubKernel(K_oo, N_odd_rows,  N_odd_cols,  floor(Kr_oo/2), floor(Kc_oo/2));
            end
        end
    end
    
    methods (Static, Access = private)
        function out = applyPolyphaseSubKernel_(I_pad, K_sub, N_r, N_c, max_pr, max_pc, pr, pc)
            % APPLYPOLYPHASESUBKERNEL_  Convolve padded input with a 2D polyphase sub-kernel.
            %
            %   out = applyPolyphaseSubKernel_(I_pad, K_sub, N_r, N_c, max_pr, max_pc, pr, pc)
            %
            %   I_pad    - already-padded small input
            %   K_sub    - the polyphase sub-kernel (e.g. 3x3 for a 5x5 original)
            %   N_r, N_c - output height and width
            %   max_pr, max_pc - padding applied to I_pad
            %   pr, pc   - half-width of this specific sub-kernel
            
            [~, ~, C] = size(I_pad);
            [Kr, Kc] = size(K_sub);
            out = zeros(N_r, N_c, C);
            
            for r = 1:Kr
                for c = 1:Kc
                    row_start = (max_pr - pr) + r;
                    col_start = (max_pc - pc) + c;
                    out = out + K_sub(r, c) * I_pad(row_start : row_start + N_r - 1, ...
                                                    col_start : col_start + N_c - 1, :);
                end
            end
        end
    end
end
