classdef SincKernel < OceanMath.Signal.Kernels.IFilterKernel
    % SINCKERNEL  Represents a Windowed Sinc filter kernel.
    %
    %   Signal Processing Intuition:
    %     The Sinc function (sin(x)/x) is the impulse response of an ideal
    %     "brick-wall" low-pass filter in the frequency domain. It perfectly
    %     passes frequencies below the cutoff and completely blocks frequencies
    %     above it.
    %
    %     Because the true sinc function extends to infinity in both directions,
    %     it must be truncated (windowed) to create a finite impulse response (FIR)
    %     filter. Without windowing, truncating the sinc causes severe "ringing"
    %     (Gibbs phenomenon) in the frequency response.
    %
    %     This kernel is the foundation for ideal resampling, interpolation, and
    %     anti-aliasing in multirate DSP (e.g., Polyphase filtering).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-24
    
    properties (Access = private)
        Size   (1,1) double = 21
        Cutoff (1,1) double = 0.5  % Normalized cutoff frequency (0 to 1], where 1 is Nyquist (pi)
        Window (1,:) char   = 'Hamming'
    end
    
    methods
        function obj = SincKernel(options)
            % SINCKERNEL Construct a SincKernel.
            %   SincKernel('Size', N, 'Cutoff', Wn, 'Window', 'Hamming')
            arguments
                options.Size   (1,1) double {mustBeInteger, mustBePositive} = 21
                options.Cutoff (1,1) double {mustBeGreaterThan(options.Cutoff, 0), mustBeLessThanOrEqual(options.Cutoff, 1)} = 0.5
                options.Window (1,:) char {mustBeMember(options.Window, {'Hamming', 'Blackman', 'Hann', 'None'})} = 'Hamming'
            end
            if mod(options.Size, 2) == 0
                error('SincKernel:SizeMustBeOdd', 'Kernel size must be odd to center the peak exactly on a pixel.');
            end
            obj.Size = options.Size;
            obj.Cutoff = options.Cutoff;
            obj.Window = options.Window;
        end
        
        function K = getMatrix(obj)
            [h, v] = obj.get1DDecomposition();
            K = v * h;
        end
        
        function [h, v] = get1DDecomposition(obj)
            N = obj.Size;
            fc = obj.Cutoff;
            
            % 1. Evaluate the Sinc function
            x = -(N-1)/2 : (N-1)/2;
            
            % MATLAB's sinc(x) = sin(pi*x)/(pi*x). 
            % For a cutoff of fc (where fc=1 is Nyquist), the filter is fc * sinc(fc * x).
            % Note: If x=0, sinc(0)=1, so value is fc.
            h_ideal = fc * sinc(fc * x);
            
            % 2. Apply Windowing to reduce Gibbs ringing
            switch obj.Window
                case 'Hamming'
                    % Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
                    w = 0.54 - 0.46 * cos(2*pi * (0:N-1) / (N-1));
                case 'Hann'
                    % Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
                    w = 0.5 * (1 - cos(2*pi * (0:N-1) / (N-1)));
                case 'Blackman'
                    % Blackman window: 0.42 - 0.5*cos(...) + 0.08*cos(...)
                    w = 0.42 - 0.5 * cos(2*pi * (0:N-1) / (N-1)) + 0.08 * cos(4*pi * (0:N-1) / (N-1));
                case 'None'
                    w = ones(1, N);
            end
            
            % Element-wise multiplication
            k1d = h_ideal .* w;
            
            % 3. Normalize to preserve DC energy (sum = 1)
            k1d = k1d / sum(k1d);
            
            h = k1d;               % horizontal 1-D filter (row vector)
            v = k1d';              % vertical   1-D filter (column vector)
        end
        
        function tf = isSeparable(~)
            tf = true;
        end
        
        function s = getSize(obj)
            s = obj.Size;
        end
        
        function c = getCutoff(obj)
            c = obj.Cutoff;
        end
        
        function w = getWindow(obj)
            w = obj.Window;
        end
    end
end
