classdef EWASampler
    % EWASAMPLER  Elliptical Weighted Average (EWA) texture sampler.
    %
    %   Signal Processing Intuition:
    %     EWA is the "gold standard" for high-quality anisotropic texture filtering.
    %     When a texture is warped (e.g., in a perspective projection), a circular
    %     pixel on the screen maps to a stretched ellipse in the texture space.
    %
    %     Standard Isotropic filtering (like basic MIP-mapping) approximates this
    %     ellipse with a large circle that encompasses it, leading to extreme over-blur
    %     along the uncompressed axis (e.g., the horizon of a checkered floor).
    %
    %     EWA instead mathematically calculates the exact covariance matrix of the
    %     ellipse using the local Jacobian of the transformation. It then performs
    %     a weighted Gaussian sum only over the texels inside that precise ellipse.
    %     This preserves maximum sharpness along the uncompressed axis while perfectly
    %     anti-aliasing the highly compressed axis.
    %
    %   Reference:
    %     Heckbert, P. S. (1989). Fundamentals of Texture Mapping and Image Warping.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-24
    
    properties
        RadiusSquared (1,1) double = 4.0  % Cutoff threshold for the Gaussian (e.g., 2 sigma)
    end
    
    methods
        function obj = EWASampler(options)
            arguments
                options.RadiusSquared (1,1) double = 4.0
            end
            obj.RadiusSquared = options.RadiusSquared;
        end
        
        function value = sample(obj, I, u0, v0, J)
            % SAMPLE  Computes the EWA pixel value.
            %
            %   value = sample(obj, I, u0, v0, J)
            %
            %   Inputs:
            %     I  - The source texture/image (H x W x C)
            %     u0 - The mapped continuous X coordinate in texture space
            %     v0 - The mapped continuous Y coordinate in texture space
            %     J  - The 2x2 local Jacobian matrix of the inverse mapping 
            %          from screen space to texture space: d(u,v)/d(x,y).
            %          J = [ du/dx, du/dy ; dv/dx, dv/dy ]
            
            [H, W, C] = size(I, 1:3);
            
            % 1. Covariance Matrix (C) in texture space
            %    C = J * J^T + I
            %    (adding the Identity matrix represents the prefilter / inherent texel blur)
            C_mat = J * J' + eye(2);
            
            % Compute the inverse covariance matrix Q = C^-1
            det_C = C_mat(1,1)*C_mat(2,2) - C_mat(1,2)*C_mat(2,1);
            if det_C < 1e-8
                % Fallback to nearest neighbor if the matrix is singular/degenerate
                u_idx = max(1, min(W, round(u0)));
                v_idx = max(1, min(H, round(v0)));
                value = double(I(v_idx, u_idx, :));
                return;
            end
            
            Q11 =  C_mat(2,2) / det_C;
            Q22 =  C_mat(1,1) / det_C;
            Q12 = -C_mat(1,2) / det_C;
            % Note: Q21 == Q12 by symmetry
            
            % 2. Ellipse Bounding Box
            %    The max extent of the ellipse x^T Q x = R^2 is given by R * sqrt(C_ii)
            R = sqrt(obj.RadiusSquared);
            bound_u = R * sqrt(C_mat(1,1));
            bound_v = R * sqrt(C_mat(2,2));
            
            % Define the integer bounding box in texture space
            u_min = max(1, floor(u0 - bound_u));
            u_max = min(W, ceil(u0 + bound_u));
            v_min = max(1, floor(v0 - bound_v));
            v_max = min(H, ceil(v0 + bound_v));
            
            % 3. Weighted Accumulation Loop
            weight_sum = 0;
            num_channels = size(I, 3);
            accum = zeros(1, num_channels);
            
            for v = v_min:v_max
                dy = v - v0;
                for u = u_min:u_max
                    dx = u - u0;
                    
                    % Evaluate the quadratic form: d^T * Q * d
                    % which equals: Q11*dx^2 + 2*Q12*dx*dy + Q22*dy^2
                    r2 = Q11*dx*dx + 2*Q12*dx*dy + Q22*dy*dy;
                    
                    % If inside the ellipse cutoff radius
                    if r2 < obj.RadiusSquared
                        % Compute Gaussian weight (unnormalized)
                        % w = exp(-alpha * r2)
                        % Using alpha = 1 for a standard Gaussian profile
                        w = exp(-0.5 * r2);
                        
                        % Accumulate
                        weight_sum = weight_sum + w;
                        
                        % Faster to loop channels or use vector? 
                        % In MATLAB, slicing I(v, u, :) is 1x1xC.
                        texel = double(I(v, u, :));
                        accum = accum + w * reshape(texel, 1, num_channels);
                    end
                end
            end
            
            % 4. Normalization
            if weight_sum > 0
                value = accum / weight_sum;
            else
                % If no pixels fell strictly inside (extremely tight ellipse),
                % fallback to nearest neighbor to prevent black holes.
                u_idx = max(1, min(W, round(u0)));
                v_idx = max(1, min(H, round(v0)));
                value = double(I(v_idx, u_idx, :));
            end
            
            % Return in standard 1x1xC format
            value = reshape(value, [1, 1, num_channels]);
        end
    end
end
