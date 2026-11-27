classdef GaussianKernel < OceanMath.Signal.Kernels.IFilterKernel
    % GAUSSIANKERNEL  Represents a 2D Gaussian filter kernel.
    %
    %   Signal Processing Intuition:
    %     This is a classic isotropic Low-Pass Filter (LPF). It is mathematically
    %     unique because both its spatial representation and its frequency response
    %     are Gaussian functions (Fourier Transform of a Gaussian is another Gaussian).
    %
    %     For a Gaussian filter in the frequency domain, as angular frequency w -> inf,
    %     the frequency response drops off exponentially: H(w) = exp(-w^2*sigma^2/2) -> 0.
    %     This means it completely attenuates high-frequency noise. Unlike other low-pass
    %     filters (like the ideal block/sinc filter), the Gaussian has no side-lobes in
    %     either domain, completely avoiding ringing or Gibbs artifacts.
    %
    %   Two computation modes (controlled by the Sigma property):
    %
    %     MODE 1 — Binomial approximation  (default, Sigma left empty)
    %     -----------------------------------------------------------------
    %     The binomial coefficients from Pascal's triangle are used as the
    %     discrete kernel weights. This is NOT a pure Gaussian — it is an
    %     integer-ratio approximation that converges to a Gaussian as the
    %     kernel size grows.
    %
    %     Why does this work?
    %       By the Central Limit Theorem (CLT), repeatedly convolving a
    %       uniform impulse response with itself produces a distribution that
    %       approaches a Gaussian. The binomial coefficient C(n,k) is exactly
    %       the result of convolving the box filter [1 1] with itself n times.
    %       Therefore, for any kernel of size N, the binomial row of Pascal's
    %       triangle is the best integer approximation to a discrete Gaussian.
    %
    %     Example — Size 5:
    %       Pascal row:   [1, 4, 6, 4, 1]   (sum = 16 = 2^4)
    %       Normalised:   [1, 4, 6, 4, 1] / 16
    %       Approximates: G(x, sigma~0.85) sampled at integer positions
    %
    %     Advantages over sampled Gaussian:
    %       - Weights are exact integer ratios (efficient fixed-point hardware).
    %       - No rounding needed to build the kernel.
    %       - Default choice in Burt & Adelson (1983) Laplacian pyramid paper.
    %
    %     MODE 2 — Sampled continuous Gaussian  (when Sigma is specified)
    %     -----------------------------------------------------------------
    %     exp(-x^2 / (2*sigma^2)) is evaluated at integer positions and
    %     normalised. Gives full control over the bandwidth (cutoff frequency)
    %     via sigma, at the cost of irrational floating-point weights.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-15
    
    properties (Access = private)
        Size  (1,1) double = 5
        Sigma (1,:) double = [] % Can be empty for binomial approximation
    end
    
    methods
        function obj = GaussianKernel(options)
            % GAUSSIANKERNEL Construct a GaussianKernel.
            %   GaussianKernel('Size', N, 'Sigma', sigma)
            arguments
                options.Size  (1,1) double {mustBeInteger, mustBePositive} = 5
                options.Sigma (1,:) double = []
            end
            if mod(options.Size, 2) == 0
                error('GaussianKernel:SizeMustBeOdd', 'Kernel size must be odd.');
            end
            if ~isempty(options.Sigma) && options.Sigma <= 0
                error('GaussianKernel:SigmaMustBePositive', 'Sigma must be positive.');
            end
            obj.Size = options.Size;
            obj.Sigma = options.Sigma;
        end
        
        function K = getMatrix(obj)
            [h, v] = obj.get1DDecomposition();
            K = v * h;
        end
        
        function [h, v] = get1DDecomposition(obj)
            N = obj.Size;
            if isempty(obj.Sigma)
                % --- BINOMIAL APPROXIMATION (default) ----------------------------
                % Weights = k-th row of Pascal's triangle: C(N-1, k) for k=0..N-1.
                % These are the coefficients obtained by convolving [1 1] with
                % itself (N-1) times, which converges to a Gaussian by the CLT.
                %
                %   N=3: [1 2 1]        sigma_eff ~ 0.5
                %   N=5: [1 4 6 4 1]    sigma_eff ~ 0.85  (Burt-Adelson default)
                %   N=7: [1 6 15 20 15 6 1]  sigma_eff ~ 1.1
                %
                % Note: sum = 2^(N-1), so normalisation divides by a power of 2
                % (fast bit-shift on integer hardware).
                k1d = zeros(1, N);
                for k = 0:(N-1)
                    k1d(k+1) = nchoosek(N-1, k);  % C(N-1, k) — Pascal coefficient
                end
            else
                % --- SAMPLED CONTINUOUS GAUSSIAN ---------------------------------
                % Sample G(x) = exp(-x^2 / (2*sigma^2)) at integer positions.
                % Gives exact bandwidth control via sigma, but uses floating-point
                % weights (no integer-ratio shortcut).
                x = -(N-1)/2 : (N-1)/2;
                k1d = exp(-x.^2 / (2 * obj.Sigma^2));
            end
            k1d = k1d / sum(k1d);  % normalise so kernel sums to 1 (energy preserving)
            h = k1d;               % horizontal 1-D filter (row vector)
            v = k1d';              % vertical   1-D filter (column vector)
        end
        
        function tf = isSeparable(~)
            tf = true;
        end
        
        function s = getSize(obj)
            s = obj.Size;
        end
        
        function sig = getSigma(obj)
            sig = obj.Sigma;
        end
    end
end
