classdef ContinuousSignalGenerator
    % CONTINUOUSSIGNALGENERATOR  Simulates continuous-time analog signals numerically.
    %
    %   PROBLEM MOTIVATION (The Continuous vs. Discrete Mismatch):
    %     1. Real physical signals (audio, sensor voltages, RF) are Continuous-Time (CT).
    %        They exist at every continuous instant t in R, having infinite temporal resolution.
    %     2. Digital computers are discrete-state machines. They can only store discrete arrays x[n].
    %     3. Paradox: To study the mathematical transformation of A/D sampling in MATLAB,
    %        we must represent a CT signal continuous-time x_c(t) inside a discrete computing environment
    %        BEFORE any sampling experiment even begins.
    %
    %   SCIENTIFIC PIPELINE DECOMPOSITION:
    %
    %     A. KNOWNS:
    %        - Continuous sinusoid equation: x_c(t) = A * cos(2*pi*f0*t + phi).
    %        - Target signal bandwidth of interest: f0 <= f_max (e.g., audio/sensor band < 5 kHz).
    %        - Discrete array sampling mechanism: x[n] = x_c(n * Ts).
    %
    %     B. UNKNOWNS:
    %        - True value of x_c(t) at non-grid timestamps t != n * Ts.
    %        - Exact discretization error E_approx = max | x_dense(t) - x_continuous(t) |.
    %
    %     C. ASSUMPTIONS:
    %        - Continuous-time behavior is accurately approximated by an ultra-dense temporal grid
    %          with reference rate F_ANALOG (default 100,000 Hz).
    %        - F_ANALOG is chosen such that F_ANALOG >> 2 * f_max (orders of magnitude above Nyquist).
    %        - Interpolation error on this dense grid is negligible (< 1e-8 relative error).
    %
    %     D. MATHEMATICAL MODEL:
    %        - Dense time grid: t = [0, dt, 2*dt, ..., T], where dt = 1 / F_ANALOG.
    %        - Simulated CT Signal: x_dense[m] = x_c(m * dt) for m = 0, 1, ..., N_analog-1.
    %        - Downsampling to discrete rate fs = F_ANALOG / M:
    %          x_sampled[n] = x_dense[n * M] = x_c(n * M * dt) = x_c(n * Ts).
    %
    %     E. PEDAGOGICAL DEDUCTION:
    %        - The dense grid x_dense serves as the absolute Ground Truth reference.
    %        - Downsampling from x_dense models physical A/D conversion.
    %        - Reconstruction algorithms (sinc, linear, ZOH) estimate x_dense from x_sampled,
    %          allowing direct error calculation: e_rec[m] = x_reconstructed[m] - x_dense[m].
    %
    %   USAGE:
    %     gen = OceanMath.Signal.ContinuousSignalGenerator('SampleRate', 100000, 'Duration', 2.0);
    %     [t, x] = gen.dualTone(50, 1.0, 300, 0.7);   % f1=50Hz, f2=300Hz
    %     [t, x] = gen.chirp(0, 1200);                % linear sweep 0 -> 1200 Hz
    %     [t, x] = gen.sinusoid(440, 1.0);
    %     [t, x] = gen.multiTone([100, 300, 700], [1.0, 0.5, 0.3]);
    %
    %   REFERENCE: Oppenheim & Schafer, "Discrete-Time Signal Processing", 3rd Ed.
    %              Chapter 4: Sampling of Continuous-Time Signals, §4.1
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    % ── Readable public state ────────────────────────────────────────────────────
    properties (SetAccess = private)
        SampleRate  (1,1) double   % F_ANALOG: the "analog" reference rate (Hz)
        Duration    (1,1) double   % Total signal duration T (seconds)
        Dt          (1,1) double   % Time step = 1/SampleRate (seconds)
        NumSamples  (1,1) double   % Total number of time points
        TimeVector  (1,:) double   % Dense time grid t = [0, Dt, 2*Dt, ..., T]
    end

    % ── Internal metadata (pullable by agents / OceanLearn) ─────────────────────
    properties (Constant, Hidden)
        ClassDescription = ['Simulates continuous-time analog signals as dense '...
            'discrete arrays. Ground-truth source for all Ch.4 sampling experiments. '...
            'Signals are generated from explicit trigonometric formulas. '...
            'See Oppenheim & Schafer Ch.4 §4.1.'];
        ProblemMotivation = ['Computers cannot store continuous-time signals x_c(t). '...
            'To analyze sampling without physical hardware, we construct an ultra-dense '...
            'discrete grid at F_ANALOG >> 2*f_max to act as continuous ground truth.'];
        Knowns = ['Continuous signal formulas x_c(t) = A*cos(2*pi*f0*t + phi); '...
            'Target band limit f_max; Sampling grid step Ts = 1/fs.'];
        Unknowns = ['Values at non-grid timestamps; Reconstruction error under downsampling.'];
        Assumptions = ['Dense reference rate F_ANALOG (100 kHz) approximates continuous time '...
            'with numerical error < 1e-8, serving as valid ground truth.'];
        MathematicalModel = ['t = [0 : 1/F_ANALOG : T]; x_dense = x_c(t); '...
            'x_sampled[n] = x_dense[n * M] where M = F_ANALOG / fs.'];
        ConceptNodeId  = 'ch4-continuous-signal-model';
        TextbookRef    = 'Oppenheim & Schafer 3rd Ed., Ch.4 §4.1, Eq. 4.1';
        PackagePath    = '+OceanMath/+Signal/ContinuousSignalGenerator.m';
    end

    methods
        % ── Constructor ──────────────────────────────────────────────────────────
        function obj = ContinuousSignalGenerator(options)
            % CONTINUOUSSIGNALGENERATOR  Construct generator with given analog rate.
            %
            %   gen = ContinuousSignalGenerator('SampleRate', 100000, 'Duration', 2.0)
            %
            %   Parameters:
            %     SampleRate — "analog" reference rate in Hz. Must be >> 2*f_max
            %                  of any signal you intend to generate. Default: 100000.
            %     Duration   — Signal duration T in seconds. Default: 1.0.
            arguments
                options.SampleRate (1,1) double {mustBePositive} = 100000
                options.Duration   (1,1) double {mustBePositive} = 1.0
            end
            obj.SampleRate = options.SampleRate;
            obj.Duration   = options.Duration;
            obj.Dt         = 1 / options.SampleRate;
            % t = [0, Dt, 2*Dt, ..., T]  — inclusive end for clean period
            obj.TimeVector = 0 : obj.Dt : options.Duration;
            obj.NumSamples = length(obj.TimeVector);
        end

        % ── Signal Generators ────────────────────────────────────────────────────

        function [t, x] = sinusoid(obj, f0, A, phi)
            % SINUSOID  Generate a single cosine tone.
            %
            %   [t, x] = gen.sinusoid(f0, A, phi)
            %
            %   Math: x(t) = A * cos(2*pi*f0*t + phi)
            %   Ref:  Oppenheim §4.1 Eq. 4.1 — continuous-time complex exponential
            %
            %   Parameters:
            %     f0  — frequency in Hz
            %     A   — amplitude (default 1.0)
            %     phi — phase in radians (default 0)
            arguments
                obj
                f0  (1,1) double {mustBePositive}
                A   (1,1) double = 1.0
                phi (1,1) double = 0
            end
            obj.validateFrequency_(f0);
            t = obj.TimeVector;
            % Eq 4.1: x(t) = A * cos(2*pi*f0*t + phi)
            x = A * cos(2*pi * f0 * t + phi);
        end

        function [t, x] = dualTone(obj, f1, A1, f2, A2, phi1, phi2)
            % DUALTONE  Sum of two cosine tones.
            %
            %   [t, x] = gen.dualTone(f1, A1, f2, A2)
            %
            %   Math: x(t) = A1*cos(2*pi*f1*t) + A2*cos(2*pi*f2*t)
            %
            %   Used in: Exp 01 (aliasing demo), Exp 06 (decimation)
            %   The two components model a low-frequency signal of interest (f1)
            %   and a high-frequency component that may alias (f2).
            arguments
                obj
                f1   (1,1) double {mustBePositive}
                A1   (1,1) double = 1.0
                f2   (1,1) double {mustBePositive} = 300
                A2   (1,1) double = 0.7
                phi1 (1,1) double = 0
                phi2 (1,1) double = 0
            end
            obj.validateFrequency_(f1);
            obj.validateFrequency_(f2);
            t = obj.TimeVector;
            % Explicit superposition — both terms visible for pedagogical clarity
            x = A1 * cos(2*pi * f1 * t + phi1) ...
              + A2 * cos(2*pi * f2 * t + phi2);
        end

        function [t, x] = multiTone(obj, freqs, amps, phases)
            % MULTITONE  Sum of N cosine tones.
            %
            %   [t, x] = gen.multiTone(freqs, amps)
            %   [t, x] = gen.multiTone(freqs, amps, phases)
            %
            %   Math: x(t) = sum_{k=1}^{N} A_k * cos(2*pi*f_k*t + phi_k)
            arguments
                obj
                freqs  (1,:) double {mustBePositive}
                amps   (1,:) double = []
                phases (1,:) double = []
            end
            N = length(freqs);
            if isempty(amps),   amps   = ones(1, N);  end
            if isempty(phases), phases = zeros(1, N); end
            assert(length(amps) == N && length(phases) == N, ...
                'ContinuousSignalGenerator:DimensionMismatch', ...
                'freqs, amps, and phases must have the same length.');
            for k = 1:N
                obj.validateFrequency_(freqs(k));
            end
            t = obj.TimeVector;
            x = zeros(size(t));
            for k = 1:N
                x = x + amps(k) * cos(2*pi * freqs(k) * t + phases(k));
            end
        end

        function [t, x] = chirp(obj, f_start, f_end)
            % CHIRP  Linear frequency sweep (chirp) signal.
            %
            %   [t, x] = gen.chirp(f_start, f_end)
            %
            %   Math: x(t) = cos(2*pi * (f_start + k*t) * t)
            %         where k = (f_end - f_start) / (2 * T)
            %
            %   Instantaneous frequency: f_inst(t) = f_start + 2*k*t
            %   This linearly sweeps from f_start at t=0 to f_end at t=T.
            %
            %   Ref: Used in Exp 02 to visualize aliasing as a V-shape bounce
            %        in the spectrogram — as instantaneous frequency crosses
            %        k*fs/2, the observed frequency reverses direction.
            arguments
                obj
                f_start (1,1) double {mustBeNonnegative}
                f_end   (1,1) double {mustBePositive}
            end
            obj.validateFrequency_(f_end);
            t  = obj.TimeVector;
            T  = obj.Duration;
            % Linear chirp rate: k = (f_end - f_start) / (2*T)
            k  = (f_end - f_start) / (2 * T);
            % Instantaneous phase: phi(t) = 2*pi*(f_start + k*t)*t
            x  = cos(2*pi * (f_start + k * t) .* t);
        end

        function [t, x] = sincSquared(obj, f_N)
            % SINCSQUARED  Generate continuous signal with ideal triangular spectrum.
            %
            %   [t, x] = gen.sincSquared(f_N)
            %
            %   Math: x(t) = sinc(f_N * t)^2
            %   Fourier Transform X_c(f): Triangular spectrum centered at 0 Hz,
            %   spanning [-f_N, +f_N]. Perfect for recreating Oppenheim Fig 4.3!
            arguments
                obj
                f_N (1,1) double {mustBePositive} = 300
            end
            obj.validateFrequency_(f_N);
            t = obj.TimeVector;
            x = sinc(f_N * t).^2;
        end

        function [t, x] = noise(obj, sigma)
            % NOISE  White Gaussian noise signal.
            %
            %   [t, x] = gen.noise(sigma)
            %
            %   Math: x(t) ~ N(0, sigma^2) i.i.d. at each time point.
            arguments
                obj
                sigma (1,1) double {mustBePositive} = 1.0
            end
            t = obj.TimeVector;
            x = sigma * randn(size(t));
        end

        % ── Metadata / Introspection ─────────────────────────────────────────────

        function s = getMetadata(obj)
            % GETMETADATA  Return struct describing this generator instance.
            %   Pullable by OceanLearn and any agent inspecting experiment state.
            s.ClassName         = 'ContinuousSignalGenerator';
            s.ConceptNodeId     = obj.ConceptNodeId;
            s.TextbookRef       = obj.TextbookRef;
            s.SampleRate_Hz     = obj.SampleRate;
            s.Duration_s        = obj.Duration;
            s.NumSamples        = obj.NumSamples;
            s.NyquistLimit_Hz   = obj.SampleRate / 2;
            s.Description       = obj.ClassDescription;
            s.ProblemMotivation = obj.ProblemMotivation;
            s.Knowns            = obj.Knowns;
            s.Unknowns          = obj.Unknowns;
            s.Assumptions       = obj.Assumptions;
            s.MathematicalModel = obj.MathematicalModel;
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print configuration to console.
            fprintf('=== ContinuousSignalGenerator ===\n');
            fprintf('  Analog reference rate : %g Hz\n',   obj.SampleRate);
            fprintf('  Duration              : %g s\n',    obj.Duration);
            fprintf('  Time step (dt)        : %g us\n',   obj.Dt * 1e6);
            fprintf('  Number of samples     : %d\n',      obj.NumSamples);
            fprintf('  Nyquist limit         : %g Hz\n',   obj.SampleRate/2);
            fprintf('  Concept node          : %s\n',      obj.ConceptNodeId);
            fprintf('  Textbook ref          : %s\n',      obj.TextbookRef);
        end
    end

    methods (Access = private)
        function validateFrequency_(obj, f)
            % Ensure f is well below Nyquist so signal is representable.
            nyq = obj.SampleRate / 2;
            if f >= nyq
                error('ContinuousSignalGenerator:FrequencyTooHigh', ...
                    'Frequency %g Hz exceeds analog Nyquist limit %g Hz. Increase SampleRate.', ...
                    f, nyq);
            end
        end
    end
end
