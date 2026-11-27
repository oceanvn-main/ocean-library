classdef IdealLowpass
    % IDEALLOWPASS Represents an ideal lowpass filter and its Fourier approximation
    %
    % Properties:
    %   CutoffFrequency - The cutoff frequency w_c (0 < w_c < pi)
    
    properties
        CutoffFrequency
    end
    
    methods
        function obj = IdealLowpass(w_c)
            % IDEALLOWPASS Construct an instance of this class
            %   obj = IdealLowpass(w_c)
            if nargin < 1
                error('IdealLowpass:Constructor:MissingArgument', 'Cutoff frequency must be specified.');
            end
            
            % Validate CutoffFrequency: must be a real double scalar, between 0 and pi (exclusive)
            validateattributes(w_c, {'double'}, {'real', 'scalar', '>', 0, '<', pi}, 'IdealLowpass', 'w_c');
            
            obj.CutoffFrequency = w_c;
        end
        
        function h = impulseResponse(obj, n)
            % IMPULSERESPONSE Computes impulse response h_lp[n]
            %   h = impulseResponse(obj, n)
            %   Safe division-by-zero check (e.g., n == 0 uses CutoffFrequency/pi)
            %   Works on arrays of indices.
            
            validateattributes(n, {'double'}, {'real'}, 'impulseResponse', 'n');
            
            % Allocate response
            h = zeros(size(n));
            
            % Safe element-by-element or vectorized computation
            wc = obj.CutoffFrequency;
            isZero = (n == 0);
            
            h(isZero) = wc / pi;
            
            % For non-zero elements
            nonZeroIdx = ~isZero;
            if any(nonZeroIdx)
                nz = n(nonZeroIdx);
                h(nonZeroIdx) = sin(wc * nz) ./ (pi * nz);
            end
        end
        
        function H = frequencyResponseApproximation(obj, w, M)
            % FREQUENCYRESPONSEAPPROXIMATION Truncated Fourier approximation
            %   H = frequencyResponseApproximation(obj, w, M)
            %   H_M(e^(j*w)) = h_lp[0] + 2 * sum_{n=1}^{M} h_lp[n] * cos(w * n)
            %   Works on arrays of frequencies w.
            
            validateattributes(w, {'double'}, {'real'}, 'frequencyResponseApproximation', 'w');
            validateattributes(M, {'double'}, {'scalar', 'integer', 'nonnegative'}, 'frequencyResponseApproximation', 'M');
            
            h0 = obj.impulseResponse(0);
            H = h0 * ones(size(w));
            
            for n = 1:M
                hn = obj.impulseResponse(n);
                H = H + 2 * hn * cos(w * n);
            end
        end
        
        function [peakVal, peakFreq, pctOvershoot] = calculateOvershoot(obj, M)
            % CALCULATEOVERSHOOT High-precision grid search to determine overshoot
            %   [peakVal, peakFreq, pctOvershoot] = calculateOvershoot(obj, M)
            %   Search in [0, obj.CutoffFrequency] (10,000 points) to determine
            %   overshoot peak amplitude, frequency, and percentage (peakVal - 1.0) * 100.
            
            validateattributes(M, {'double'}, {'scalar', 'integer', 'nonnegative'}, 'calculateOvershoot', 'M');
            
            % Create grid of 10,000 points spanning [0, obj.CutoffFrequency]
            wc = obj.CutoffFrequency;
            w = linspace(0, wc, 10000);
            
            % Compute approximation at these frequencies
            H = obj.frequencyResponseApproximation(w, M);
            
            % Find peak value
            [peakVal, idx] = max(H);
            peakFreq = w(idx);
            
            % Percentage overshoot relative to ideal value 1.0 (since w <= wc, ideal response is 1.0)
            pctOvershoot = (peakVal - 1.0) * 100;
        end
        
        function [errFreq, errTime] = verifyParseval(obj, M, numGridPoints)
            % VERIFYPARSEVAL Calculates frequency domain error and time domain error
            %   [errFreq, errTime] = verifyParseval(obj, M, numGridPoints)
            %   errFreq: numerical integration of |H_lp(e^(j*w)) - H_M(e^(j*w))|^2 over [-pi, pi]
            %   using trapezoidal rule (trapz).
            %   errTime: analytical tail sum: (obj.CutoffFrequency / pi) - ( impulseResponse(0)^2 + 2 * sum_{n=1}^{M} impulseResponse(n)^2 )
            
            if nargin < 3 || isempty(numGridPoints)
                numGridPoints = 10000;
            end
            
            validateattributes(M, {'double'}, {'scalar', 'integer', 'nonnegative'}, 'verifyParseval', 'M');
            validateattributes(numGridPoints, {'double'}, {'scalar', 'integer', 'positive'}, 'verifyParseval', 'numGridPoints');
            
            wc = obj.CutoffFrequency;
            
            % Create a composite grid that includes -wc and wc exactly
            % to eliminate jump discontinuity integration errors in trapz.
            N_half_stop = max(2, round(numGridPoints * (pi - wc) / (2 * pi)));
            N_pass = max(2, round(numGridPoints * (2 * wc) / (2 * pi)));
            
            w1 = linspace(-pi, -wc, N_half_stop);
            w2 = linspace(-wc, wc, N_pass);
            w3 = linspace(wc, pi, N_half_stop);
            
            w = [w1, w2(2:end), w3(2:end)];
            
            % Ideal frequency response H_lp(e^(j*w))
            % |w| <= wc is 1, otherwise 0. Since wc is exactly in the grid,
            % we use a tolerance to protect against machine precision variations.
            H_ideal = double(abs(w) <= (wc + 1e-14));
            
            % Approximated frequency response
            H_approx = obj.frequencyResponseApproximation(w, M);
            
            % Squared difference
            diffSq = (H_ideal - H_approx).^2;
            
            % Integral using trapz, then divided by 2*pi
            errFreq = trapz(w, diffSq) / (2 * pi);
            
            % 2. Time domain error calculation
            % The total energy of ideal is wc / pi
            h0 = obj.impulseResponse(0);
            sum_hn_sq = h0^2;
            for n = 1:M
                hn = obj.impulseResponse(n);
                sum_hn_sq = sum_hn_sq + 2 * (hn^2);
            end
            
            totalEnergy = wc / pi;
            errTime = totalEnergy - sum_hn_sq;
        end
    end
end
