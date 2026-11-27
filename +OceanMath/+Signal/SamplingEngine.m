classdef SamplingEngine
    % SAMPLINGENGINE  1D periodic downsampling of a continuous-time signal array.
    %
    %   PEDAGOGICAL PURPOSE:
    %     This class models the A/D Sampling step: converting a dense "analog"
    %     time array (produced by ContinuousSignalGenerator) into a sparse
    %     discrete sequence by keeping every M-th sample.
    %
    %     Two operation modes are provided side-by-side to make the Nyquist
    %     theorem physically tangible:
    %
    %       MODE 1 — Direct (no prefilter):
    %         Keeps every M-th sample with no preparation.
    %         If f_max >= fs/2, the spectrum contains ALIASES — ghost frequencies
    %         that appear at wrong locations. This is the aliasing effect.
    %
    %       MODE 2 — Anti-aliased (with ideal LPF prefilter):
    %         Applies an ideal lowpass filter (cutoff = fs/2) BEFORE downsampling.
    %         Removes all spectral energy above Nyquist first, then samples.
    %         Result: no aliasing, but high-frequency signal content is lost.
    %
    %     The choice between these modes is the central decision of §4.1 and §4.6.1.
    %
    %   USAGE:
    %     eng = OceanMath.Signal.SamplingEngine('AnalogRate', 100000, 'TargetRate', 1000);
    %     [t_s, x_s] = eng.sample(t, x);                % direct (may alias)
    %     [t_s, x_s] = eng.sampleWithFilter(t, x, lpf); % anti-aliased
    %     M          = eng.getDecimationFactor();        % = 100
    %     f_alias    = eng.predictAlias(300);            % 300 Hz aliases to ?
    %
    %   REFERENCE: Oppenheim & Schafer 3rd Ed.
    %              §4.1 — Periodic Sampling, Eq. 4.1–4.10
    %              §4.6.1 — Sampling Rate Reduction by Integer Factor
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    properties (SetAccess = private)
        AnalogRate  (1,1) double   % F_ANALOG: reference "analog" sample rate (Hz)
        TargetRate  (1,1) double   % fs: target discrete sample rate (Hz)
        M           (1,1) double   % Decimation factor M = AnalogRate / TargetRate
    end

    properties (Constant, Hidden)
        ClassDescription = ['1D periodic downsampling of dense analog signal arrays. '...
            'Models A/D conversion. Two modes: direct (aliasing visible) and '...
            'prefiltered (anti-aliased). The core operation of Ch.4 §4.1 and §4.6.1. '...
            'See Oppenheim & Schafer Eq. 4.1-4.10.'];
        ConceptNodeId = 'ch4-periodic-sampling';
        TextbookRef   = 'Oppenheim & Schafer 3rd Ed., §4.1 Eq. 4.2, §4.6.1 Eq. 4.71';
        PackagePath   = '+OceanMath/+Signal/SamplingEngine.m';
    end

    methods
        % ── Constructor ──────────────────────────────────────────────────────────
        function obj = SamplingEngine(options)
            % SAMPLINGENGINE  Construct a sampling engine.
            %
            %   eng = SamplingEngine('AnalogRate', 100000, 'TargetRate', 1000)
            %
            %   Parameters:
            %     AnalogRate — Reference "analog" rate (Hz). Must match the
            %                  ContinuousSignalGenerator.SampleRate used to generate x.
            %     TargetRate — Target discrete sample rate fs (Hz).
            %                  Must be an integer factor of AnalogRate.
            arguments
                options.AnalogRate (1,1) double {mustBePositive} = 100000
                options.TargetRate (1,1) double {mustBePositive} = 1000
            end
            assert(options.AnalogRate >= options.TargetRate, ...
                'SamplingEngine:RateError', ...
                'TargetRate must be <= AnalogRate.');
            obj.AnalogRate = options.AnalogRate;
            obj.TargetRate = options.TargetRate;
            % Decimation factor: keep every M-th sample
            % M = F_ANALOG / fs  (Oppenheim §4.6.1)
            obj.M = round(options.AnalogRate / options.TargetRate);
        end

        % ── Core Sampling ────────────────────────────────────────────────────────

        function [t_s, x_s] = sample(obj, t, x)
            % SAMPLE  Direct periodic downsampling (no anti-aliasing filter).
            %
            %   [t_s, x_s] = eng.sample(t, x)
            %
            %   Operation (Oppenheim §4.1, Eq. 4.2):
            %     x[n] = x_c(n*Ts)   for n = 0, 1, 2, ...
            %     Implemented as: keep every M-th element of the dense array.
            %
            %   WARNING: If any spectral component of x has f >= fs/2, the
            %   result WILL contain aliased ghost frequencies. This is intentional
            %   for Exp 01 and Exp 02, where aliasing must be made visible.
            arguments
                obj
                t (1,:) double   % Dense time vector (from ContinuousSignalGenerator)
                x (1,:) double   % Dense signal array (same length as t)
            end
            assert(length(t) == length(x), 'SamplingEngine:LengthMismatch', ...
                't and x must have the same length.');
            % Keep every M-th sample: indices 1, M+1, 2M+1, ...
            idx = 1 : obj.M : length(x);
            t_s = t(idx);
            x_s = x(idx);
        end

        function [t_s, x_s] = sampleWithFilter(obj, t, x, lpf_kernel)
            % SAMPLEWITHFILTER  Anti-aliased downsampling: LPF then decimate.
            %
            %   [t_s, x_s] = eng.sampleWithFilter(t, x, lpf_kernel)
            %
            %   Operation (Oppenheim §4.8.1 — Prefiltering to Avoid Aliasing):
            %     1. Convolve x with anti-aliasing lowpass filter h[n]
            %        (cutoff = fs/2 in the analog domain, i.e., pi/M in discrete)
            %     2. Keep every M-th sample of the filtered signal
            %
            %   The LPF removes all spectral energy above fs/2 BEFORE sampling,
            %   preventing any frequency from "folding" into the baseband.
            %
            %   Parameters:
            %     lpf_kernel — 1D FIR filter coefficients (row vector).
            %                  Use IdealLowpass.impulseResponse() or SincKernel.
            %                  If omitted, uses a 64-tap windowed-sinc at pi/M.
            arguments
                obj
                t          (1,:) double
                x          (1,:) double
                lpf_kernel (1,:) double = []
            end
            assert(length(t) == length(x), 'SamplingEngine:LengthMismatch', ...
                't and x must have the same length.');

            if isempty(lpf_kernel)
                % Default: 64-tap windowed-sinc at cutoff pi/M (Hamming window)
                % Normalized cutoff = (fs/2) / (F_ANALOG/2) = 1/M
                n_taps    = 64;
                fc_norm   = 1 / obj.M;           % fraction of Nyquist of F_ANALOG
                n_vec     = -(n_taps-1)/2 : (n_taps-1)/2;
                h_ideal   = fc_norm * sinc(fc_norm * n_vec);   % ideal sinc
                % Hamming window to reduce Gibbs ringing
                win       = 0.54 - 0.46*cos(2*pi*(0:n_taps-1)/(n_taps-1));
                lpf_kernel = h_ideal .* win;
                lpf_kernel = lpf_kernel / sum(lpf_kernel);     % unit gain
            end

            % Step 1: Convolve with anti-aliasing filter (linear, no circular wrap)
            % 'same' keeps length, centers the delay
            x_filt = conv(x, lpf_kernel, 'same');

            % Step 2: Periodic downsampling
            idx = 1 : obj.M : length(x_filt);
            t_s = t(idx);
            x_s = x_filt(idx);
        end

        % ── Utility ──────────────────────────────────────────────────────────────

        function M = getDecimationFactor(obj)
            % GETDECIMATIONFACTOR  Return M = AnalogRate / TargetRate.
            M = obj.M;
        end

        function f_alias = predictAlias(obj, f0)
            % PREDICTALIAS  Compute predicted alias frequency for tone f0.
            %
            %   f_alias = eng.predictAlias(f0)
            %
            %   Applies the spectral folding formula (Oppenheim §4.2):
            %     f_alias = |mod(f0 + fs/2, fs) - fs/2|
            %
            %   Works on vectors: f0 may be a frequency sweep vector.
            f_alias = OceanMath.Signal.SpectralAnalyzer.aliasFormula(f0, obj.TargetRate);
        end

        function nyq = getNyquist(obj)
            % GETNYQUIST  Return Nyquist frequency fs/2.
            nyq = obj.TargetRate / 2;
        end

        function s = getMetadata(obj)
            % GETMETADATA  Return description struct for agents and OceanLearn.
            s.ClassName        = 'SamplingEngine';
            s.ConceptNodeId    = obj.ConceptNodeId;
            s.TextbookRef      = obj.TextbookRef;
            s.Description      = obj.ClassDescription;
            s.AnalogRate_Hz    = obj.AnalogRate;
            s.TargetRate_Hz    = obj.TargetRate;
            s.DecimationFactor = obj.M;
            s.Nyquist_Hz       = obj.getNyquist();
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print configuration to console.
            fprintf('=== SamplingEngine ===\n');
            fprintf('  Analog reference rate : %g Hz\n', obj.AnalogRate);
            fprintf('  Target sample rate    : %g Hz  (fs)\n', obj.TargetRate);
            fprintf('  Decimation factor M   : %d\n', obj.M);
            fprintf('  Nyquist frequency     : %g Hz  (fs/2)\n', obj.getNyquist());
            fprintf('  Concept node          : %s\n', obj.ConceptNodeId);
        end
    end
end
