classdef SmoothFunctionGenerator < OceanData.Generators.IDataGenerator
    % SMOOTHFUNCTIONGENERATOR  Generates scattered data from smoothly-varying
    %   functions built from random superpositions of sinusoids and/or Gaussians.
    %
    %   Implements Exercise 4.1, Step 1 (Szeliski §4.1):
    %     "Generate a random 1-D or 2-D function by adding together a small
    %      number of sinusoids or Gaussians of random amplitudes and frequencies
    %      or scales."
    %
    % ── Modes ────────────────────────────────────────────────────────────
    %   'sinusoid'  : f(x) = sum_k  A_k * sin(2*pi*Omega_k * x + Phi_k)
    %   'gaussian'  : f(x) = sum_k  A_k * exp(-||x-Mu_k||^2 / (2*Sigma_k^2))
    %   'mixed'     : sum of both sinusoid and Gaussian terms
    %   'piecewise' : domain split into M regions, each with an independent draw
    %                 (optional Ex 4.1 step 7 — harder interpolation problem)
    %
    % ── Dimensionality ───────────────────────────────────────────────────
    %   Dim=1 : X is (N x 1), domain is [DomainLow, DomainHigh]
    %   Dim=2 : X is (N x 2), domain is [DomainLow, DomainHigh]^2 (square)
    %
    % ── Output contract ──────────────────────────────────────────────────
    %   X         : (N x Dim)  scatter sample locations (uniform random)
    %   Y         : (N x 1)    noisy observations  f(X) + noise
    %   MetaData.f_true    : function handle  @(Xq) → (M x 1) noiseless values
    %   MetaData.components: struct with fields A, Omega/Sigma, Phi/Mu for each term
    %   MetaData.Sigma     : noise std used
    %
    % ── Usage ─────────────────────────────────────────────────────────────
    %   gen = OceanData.Generators.SmoothFunctionGenerator( ...
    %               'N', 50, 'Dim', 2, 'Mode', 'gaussian', 'NumTerms', 4, ...
    %               'NoiseStd', 0.05, 'Seed', 42);
    %   [X, Y] = gen.generate();
    %   f_true = gen.MetaData.f_true;
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        Dim        (1,1) double {mustBeMember(Dim, [1, 2])} = 1
        % 'sinusoid' | 'gaussian' | 'mixed' | 'piecewise'
        Mode       (1,1) string {mustBeMember(Mode, ...
                       ["sinusoid","gaussian","mixed","piecewise"])} = "sinusoid"
        NumTerms   (1,1) double {mustBePositive, mustBeInteger} = 3
        NoiseStd   (1,1) double {mustBeNonnegative} = 0.05
        DomainLow  (1,1) double = 0.0
        DomainHigh (1,1) double = 1.0
        % Piecewise-only: number of domain regions
        NumPieces  (1,1) double {mustBePositive, mustBeInteger} = 3
        % Amplitude range for random draws
        AmpRange   (1,2) double = [0.5, 2.0]
        % Frequency range (sinusoids): cycles over [DomainLow, DomainHigh]
        FreqRange  (1,2) double = [1.0, 5.0]
        % Scale range (Gaussians): fraction of domain width
        ScaleRange (1,2) double = [0.05, 0.25]
    end

    methods
        function obj = SmoothFunctionGenerator(opts)
            % SMOOTHFUNCTIONGENERATOR Constructor.
            %
            %   All parameters are optional Name-Value pairs.

            arguments
                opts.N          (1,1) double {mustBePositive, mustBeInteger} = 50
                opts.Seed       (1,1) double {mustBeNonnegative} = 0
                opts.Dim        (1,1) double = 1
                opts.Mode       (1,1) string = "sinusoid"
                opts.NumTerms   (1,1) double = 3
                opts.NoiseStd   (1,1) double = 0.05
                opts.DomainLow  (1,1) double = 0.0
                opts.DomainHigh (1,1) double = 1.0
                opts.NumPieces  (1,1) double = 3
                opts.AmpRange   (1,2) double = [0.5, 2.0]
                opts.FreqRange  (1,2) double = [1.0, 5.0]
                opts.ScaleRange (1,2) double = [0.05, 0.25]
            end

            obj.N          = opts.N;
            obj.Seed       = opts.Seed;
            obj.Dim        = opts.Dim;
            obj.Mode       = opts.Mode;
            obj.NumTerms   = opts.NumTerms;
            obj.NoiseStd   = opts.NoiseStd;
            obj.DomainLow  = opts.DomainLow;
            obj.DomainHigh = opts.DomainHigh;
            obj.NumPieces  = opts.NumPieces;
            obj.AmpRange   = opts.AmpRange;
            obj.FreqRange  = opts.FreqRange;
            obj.ScaleRange = opts.ScaleRange;
        end
    end

    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            L = obj.DomainHigh - obj.DomainLow;

            % 1. Generate scatter locations X ~ Uniform[DomainLow, DomainHigh]^Dim
            X = obj.DomainLow + L * rand(obj.N, obj.Dim);

            % 2. Build the ground-truth function and draw random parameters
            switch obj.Mode
                case 'sinusoid'
                    [f_true, comp] = obj.buildSinusoid_(L);
                case 'gaussian'
                    [f_true, comp] = obj.buildGaussian_(L);
                case 'mixed'
                    [f_sin, c_sin] = obj.buildSinusoid_(L);
                    [f_gau, c_gau] = obj.buildGaussian_(L);
                    f_true = @(Xq) f_sin(Xq) + f_gau(Xq);
                    comp = struct('sinusoid', c_sin, 'gaussian', c_gau);
                case 'piecewise'
                    [f_true, comp] = obj.buildPiecewise_(L);
            end

            % 3. Evaluate at scatter sites
            f_vals = f_true(X);

            % 4. Add iid Gaussian noise
            noise = obj.NoiseStd * randn(obj.N, 1);
            Y = f_vals + noise;

            % 5. Store meta
            obj.MetaData.f_true     = f_true;
            obj.MetaData.components = comp;
            obj.MetaData.Sigma      = obj.NoiseStd;
            obj.MetaData.Dim        = obj.Dim;
            obj.MetaData.Mode       = obj.Mode;
        end
    end

    methods (Access = private)

        function [f, comp] = buildSinusoid_(obj, L)
            % f(x) = sum_k  A_k * sin(2*pi*Omega_k * (x - DomainLow)/L + Phi_k)
            K   = obj.NumTerms;
            A   = obj.randInRange_(obj.AmpRange,  K, 1);
            Phi = 2*pi * rand(K, 1);

            if obj.Dim == 1
                Omega = obj.randInRange_(obj.FreqRange, K, 1);
                f = @(Xq) obj.evalSinusoid1D_(Xq, A, Omega, Phi, L);
            else
                % 2D: each term has a 2D frequency vector
                Omega = obj.randInRange_(obj.FreqRange, K, 2);  % (K x 2)
                f = @(Xq) obj.evalSinusoid2D_(Xq, A, Omega, Phi, L);
            end

            comp = struct('A', A, 'Omega', Omega, 'Phi', Phi);
        end

        function [f, comp] = buildGaussian_(obj, L)
            % f(x) = sum_k  A_k * exp(-||x - Mu_k||^2 / (2*Sigma_k^2))
            K     = obj.NumTerms;
            A     = obj.randInRange_(obj.AmpRange,  K, 1);
            Sigma = obj.randInRange_(obj.ScaleRange, K, 1) * L;
            Mu    = obj.DomainLow + L * rand(K, obj.Dim);  % centres in domain

            if obj.Dim == 1
                f = @(Xq) obj.evalGaussian1D_(Xq, A, Mu, Sigma);
            else
                f = @(Xq) obj.evalGaussian2D_(Xq, A, Mu, Sigma);
            end

            comp = struct('A', A, 'Mu', Mu, 'Sigma', Sigma);
        end

        function [f, comp] = buildPiecewise_(obj, L)
            % Split the domain into NumPieces equal slabs (along dim 1).
            % Each piece independently draws a sinusoid + Gaussian function.
            M = obj.NumPieces;
            breaks = obj.DomainLow + L * (0:M) / M;  % (1 x M+1)

            fns  = cell(M, 1);
            comp = struct();
            comp.pieces = cell(M, 1);

            for m = 1:M
                sub = OceanData.Generators.SmoothFunctionGenerator( ...
                    'N',          10, ...          % placeholder N (not used for f)
                    'Dim',        obj.Dim, ...
                    'Mode',       'mixed', ...
                    'NumTerms',   obj.NumTerms, ...
                    'NoiseStd',   0, ...
                    'DomainLow',  breaks(m), ...
                    'DomainHigh', breaks(m+1), ...
                    'AmpRange',   obj.AmpRange, ...
                    'FreqRange',  obj.FreqRange, ...
                    'ScaleRange', obj.ScaleRange);

                % Generate with N=1 to trigger random draw without returning data
                sub.generate(1);
                fns{m}        = sub.MetaData.f_true;
                comp.pieces{m} = sub.MetaData.components;
            end

            f = @(Xq) obj.evalPiecewise_(Xq, fns, breaks);
        end

        % ── Static evaluators ─────────────────────────────────────────────

        function y = evalSinusoid1D_(~, Xq, A, Omega, Phi, L)
            % Xq: (M x 1), A/Omega/Phi: (K x 1) → y: (M x 1)
            t = (Xq(:,1) - 0) / L;       % normalised coordinate
            y = (A .* sin(2*pi * Omega .* t' + Phi))';  % (M x K)
            y = sum(y, 2);
        end

        function y = evalSinusoid2D_(~, Xq, A, Omega, Phi, L)
            % Xq: (M x 2), Omega: (K x 2), A/Phi: (K x 1) → y: (M x 1)
            M = size(Xq, 1);
            K = numel(A);
            y = zeros(M, 1);
            for k = 1:K
                phase = 2*pi * (Xq * Omega(k,:)' / L) + Phi(k);  % (M x 1)
                y = y + A(k) * sin(phase);
            end
        end

        function y = evalGaussian1D_(~, Xq, A, Mu, Sigma)
            % Xq: (M x 1), Mu/Sigma: (K x 1) → y: (M x 1)
            diff = Xq(:,1) - Mu';                    % (M x K)
            y = (A' .* exp(-diff.^2 ./ (2 * Sigma'.^2)));  % (M x K)
            y = sum(y, 2);
        end

        function y = evalGaussian2D_(~, Xq, A, Mu, Sigma)
            % Xq: (M x 2), Mu: (K x 2) → y: (M x 1)
            M = size(Xq, 1);
            K = numel(A);
            y = zeros(M, 1);
            for k = 1:K
                d2 = sum((Xq - Mu(k,:)).^2, 2);  % (M x 1) squared distance
                y  = y + A(k) * exp(-d2 / (2 * Sigma(k)^2));
            end
        end

        function y = evalPiecewise_(~, Xq, fns, breaks)
            % Xq: (M x D), evaluate each piece based on x-coordinate.
            M  = size(Xq, 1);
            y  = zeros(M, 1);
            np = numel(fns);
            for m = 1:np
                if m < np
                    mask = Xq(:,1) >= breaks(m) & Xq(:,1) < breaks(m+1);
                else
                    mask = Xq(:,1) >= breaks(m) & Xq(:,1) <= breaks(m+1);
                end
                if any(mask)
                    y(mask) = fns{m}(Xq(mask,:));
                end
            end
        end

        function v = randInRange_(~, range, varargin)
            % Helper: uniform random values in [range(1), range(2)]
            lo = range(1); hi = range(2);
            v  = lo + (hi - lo) * rand(varargin{:});
        end

    end
end
