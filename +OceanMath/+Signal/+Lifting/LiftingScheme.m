classdef LiftingScheme < OceanMath.Signal.Lifting.ILiftingTransform
    % LIFTINGSCHEME  Implements the fast Wavelet Lifting Scheme.
    %
    %   Decomposes signals and images into multi-resolution approximation and
    %   detail coefficients using in-place Split, Predict, and Update steps.
    %   Supports lossless integer-to-integer mapping for CDF 5/3 (used in JPEG 2000)
    %   and Haar wavelets.
    %
    %   Constructor:
    %       LiftingScheme('WaveletType', type, 'Levels', levels)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = public)
        WaveletType char {mustBeMember(WaveletType, {'Haar', 'CDF53'})} = 'CDF53'
        Levels      (1,1) double {mustBeInteger, mustBePositive} = 1
    end
    
    methods
        function obj = LiftingScheme(options)
            % LIFTINGSCHEME Construct a LiftingScheme transform.
            arguments
                options.WaveletType char = 'CDF53'
                options.Levels      (1,1) double = 1
            end
            obj.WaveletType = options.WaveletType;
            obj.Levels = options.Levels;
            obj.Description = 'Input: Standard image array. Output: Math struct of Wavelet Coefficients (LL, HL, LH, HH). Breaks the standard image pipeline! Use Wavelet Subbands view to visualize.';
        end
        
        function coeffs = forward(obj, data)
            % FORWARD Performs the forward wavelet transform.
            %   Auto-dispatches to 1D or 2D based on input shape.
            arguments
                obj
                data double
            end
            if isvector(data)
                [L, H] = obj.forward1D(data(:));
                coeffs = struct('L', L, 'H', H);
            else
                coeffs = obj.forward2D(data);
            end
        end
        
        function data = inverse(obj, coeffs)
            % INVERSE Performs the inverse wavelet transform (reconstruction).
            %   Auto-dispatches to 1D or 2D based on coefficient structure.
            arguments
                obj
                coeffs struct
            end
            if isfield(coeffs, 'L') && isfield(coeffs, 'H')
                data = obj.inverse1D(coeffs.L, coeffs.H);
            else
                data = obj.inverse2D(coeffs);
            end
        end
        
        function [s, d] = forward1D(obj, x)
            % FORWARD1D Applies one level of 1D forward lifting transform.
            N = length(x);
            if mod(N, 2) ~= 0
                error('LiftingScheme:OddLength', '1D input length must be even.');
            end
            
            % Split step: separate even and odd samples
            even = x(1:2:end);
            odd = x(2:2:end);
            
            if strcmp(obj.WaveletType, 'Haar')
                % Haar Predict: detail is difference
                d = odd - even;
                % Haar Update: approximation is mean
                s = even + d / 2;
            else
                % CDF 5/3 Predict: d[n] = odd[n] - floor((even[n] + even[n+1]) / 2)
                % Mirror the right boundary: even[N/2 + 1] = even[N/2]
                even_ext = [even; even(end)];
                d = odd - floor((even_ext(1:end-1) + even_ext(2:end)) / 2);
                
                % CDF 5/3 Update: s[n] = even[n] + floor((d[n-1] + d[n] + 2) / 4)
                % Mirror the left boundary: d[0] = d[1]
                d_ext = [d(1); d];
                s = even + floor((d_ext(1:end-1) + d_ext(2:end) + 2) / 4);
            end
        end
        
        function x = inverse1D(obj, s, d)
            % INVERSE1D Applies one level of 1D inverse lifting transform.
            if strcmp(obj.WaveletType, 'Haar')
                % Haar Inverse Update
                even = s - d / 2;
                % Haar Inverse Predict
                odd = d + even;
            else
                % CDF 5/3 Inverse Update
                d_ext = [d(1); d];
                even = s - floor((d_ext(1:end-1) + d_ext(2:end) + 2) / 4);
                
                % CDF 5/3 Inverse Predict
                even_ext = [even; even(end)];
                odd = d + floor((even_ext(1:end-1) + even_ext(2:end)) / 2);
            end
            
            % Interleave even and odd back into a single vector
            x = zeros(2 * length(s), 1);
            x(1:2:end) = even;
            x(2:2:end) = odd;
        end
    end
    
    methods (Access = private)
        function coeffs = forward2D(obj, I)
            % FORWARD2D Multi-level 2D Wavelet transform on images.
            [H, W, C] = size(I);
            
            % Calculate padding factor needed for Levels decomposition
            factor = 2^obj.Levels;
            pad_H = mod(factor - mod(H, factor), factor);
            pad_W = mod(factor - mod(W, factor), factor);
            
            % Pad image at bottom and right using replicate padding
            I_padded = double(I);
            if pad_H > 0
                I_padded = [I_padded; repmat(I_padded(end, :, :), pad_H, 1)];
            end
            if pad_W > 0
                I_padded = [I_padded, repmat(I_padded(:, end, :), 1, pad_W)];
            end
            
            LL = I_padded;
            levels_data = cell(obj.Levels, 1);
            
            for lvl = 1:obj.Levels
                [cur_H, cur_W, ~] = size(LL);
                
                % Row transform (1D transform along dimension 2)
                row_L = zeros(cur_H, cur_W/2, C);
                row_H = zeros(cur_H, cur_W/2, C);
                for ch = 1:C
                    for r = 1:cur_H
                        [L_1d, H_1d] = obj.forward1D(LL(r, :, ch)');
                        row_L(r, :, ch) = L_1d';
                        row_H(r, :, ch) = H_1d';
                    end
                end
                
                % Column transform (1D transform along dimension 1)
                LL_new = zeros(cur_H/2, cur_W/2, C);
                LH = zeros(cur_H/2, cur_W/2, C);
                HL = zeros(cur_H/2, cur_W/2, C);
                HH = zeros(cur_H/2, cur_W/2, C);
                
                for ch = 1:C
                    % row_L columns -> LL and LH
                    for c = 1:(cur_W/2)
                        [L_1d, H_1d] = obj.forward1D(row_L(:, c, ch));
                        LL_new(:, c, ch) = L_1d;
                        LH(:, c, ch) = H_1d;
                    end
                    % row_H columns -> HL and HH
                    for c = 1:(cur_W/2)
                        [L_1d, H_1d] = obj.forward1D(row_H(:, c, ch));
                        HL(:, c, ch) = L_1d;
                        HH(:, c, ch) = H_1d;
                    end
                end
                
                % Save detail coefficients and update approximation subband
                levels_data{lvl} = struct('LH', LH, 'HL', HL, 'HH', HH);
                LL = LL_new;
            end
            
            coeffs = struct('LL', LL, 'Levels', {levels_data}, ...
                            'OrigSize', [H, W]);
        end
        
        function I = inverse2D(obj, coeffs)
            % INVERSE2D Multi-level 2D Wavelet reconstruction.
            LL = coeffs.LL;
            levels_data = coeffs.Levels;
            orig_H = coeffs.OrigSize(1);
            orig_W = coeffs.OrigSize(2);
            C = size(LL, 3);
            
            for lvl = obj.Levels:-1:1
                det = levels_data{lvl};
                LH = det.LH;
                HL = det.HL;
                HH = det.HH;
                
                recon_H = size(LL, 1) + size(LH, 1);
                recon_W = size(LL, 2);
                
                % Column reconstruction
                row_L = zeros(recon_H, recon_W, C);
                row_H = zeros(recon_H, recon_W, C);
                
                for ch = 1:C
                    % LL and LH -> row_L
                    for c = 1:recon_W
                        row_L(:, c, ch) = obj.inverse1D(LL(:, c, ch), LH(:, c, ch));
                    end
                    % HL and HH -> row_H
                    for c = 1:recon_W
                        row_H(:, c, ch) = obj.inverse1D(HL(:, c, ch), HH(:, c, ch));
                    end
                end
                
                % Row reconstruction
                next_LL = zeros(recon_H, recon_W * 2, C);
                for ch = 1:C
                    for r = 1:recon_H
                        next_LL(r, :, ch) = obj.inverse1D(row_L(r, :, ch)', row_H(r, :, ch)')';
                    end
                end
                
                LL = next_LL;
            end
            
            % Crop reconstruction back to original sizes
            I = LL(1:orig_H, 1:orig_W, :);
        end
    end
end
