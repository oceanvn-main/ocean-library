classdef QuantizationEngine
    % QUANTIZATIONENGINE  Uniform B-bit scalar quantizer with noise analysis.
    %
    %   PEDAGOGICAL PURPOSE:
    %     Real A/D converters round continuous amplitude values to the nearest
    %     discrete level. This introduces a deterministic but practically
    %     unpredictable error: the QUANTIZATION ERROR e[n] = x_Q[n] - x[n].
    %
    %     The STATISTICAL QUANTIZATION NOISE MODEL (Oppenheim §4.8.3) assumes:
    %       1. e[n] is uniformly distributed on [-Delta/2, +Delta/2]
    %       2. e[n] is stationary and white (uncorrelated sample-to-sample)
    %       3. e[n] is uncorrelated with the input signal x[n]
    %
    %     Under these assumptions, theoretical noise power = Delta^2 / 12
    %     and SQNR for a full-scale sinusoid = 6.02*B + 1.76 dB.
    %
    %     Experiments verify when this model holds (B >= 6) and when it
    %     breaks down (B = 2,3 — error becomes correlated with input).
    %
    %   USAGE:
    %     qe     = OceanMath.Signal.QuantizationEngine('BitDepth', 8);
    %     x_q    = qe.quantize(x);          % quantized signal
    %     e      = qe.error(x);             % quantization error
    %     sqnr_t = qe.theoreticalSQNR();    % 6.02*8 + 1.76 = 49.92 dB
    %     sqnr_e = qe.empiricalSQNR(x);     % measured from actual x
    %     Delta  = qe.stepSize();           % quantization step Delta
    %
    %   REFERENCE: Oppenheim & Schafer 3rd Ed.
    %              §4.8.2 — Analog-to-Digital Conversion
    %              §4.8.3 — Analysis of Quantization Errors, Eq. 4.88–4.95
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    properties (SetAccess = private)
        BitDepth  (1,1) double   % B: number of quantization bits
        XMax      (1,1) double   % Maximum signal amplitude (full-scale = XMax)
        XMin      (1,1) double   % Minimum signal amplitude (full-scale = -XMax)
        NLevels   (1,1) double   % L = 2^B: total number of quantization levels
        Delta_    (1,1) double   % Step size Delta = 2*XMax / (2^B - 1)
    end

    properties (Constant, Hidden)
        ClassDescription = ['Uniform B-bit midrise scalar quantizer. Rounds input '...
            'to nearest discrete level. Provides error signal, theoretical noise '...
            'power (Delta^2/12), and SQNR (6.02*B + 1.76 dB for full-scale sinusoid). '...
            'Used in Exp 09 (quantization analysis) and Exp 10 (sigma-delta 1-bit). '...
            'See Oppenheim & Schafer §4.8.3 Eq. 4.88-4.95.'];
        ConceptNodeId = 'ch4-quantization-noise';
        TextbookRef   = 'Oppenheim & Schafer 3rd Ed., §4.8.3 Eq. 4.88–4.95';
        PackagePath   = '+OceanMath/+Signal/QuantizationEngine.m';
    end

    methods
        % ── Constructor ──────────────────────────────────────────────────────────
        function obj = QuantizationEngine(options)
            % QUANTIZATIONENGINE  Construct a B-bit quantizer.
            %
            %   qe = QuantizationEngine('BitDepth', 8)
            %   qe = QuantizationEngine('BitDepth', 1, 'Range', [-1, 1])
            %
            %   Parameters:
            %     BitDepth — Number of bits B (default 8). B=1 -> 1-bit comparator
            %                used inside SigmaDeltaModulator.
            %     Range    — [xmin, xmax] full-scale input range. Default [-1, 1].
            arguments
                options.BitDepth (1,1) double {mustBePositive, mustBeInteger} = 8
                options.Range    (1,2) double = [-1, 1]
            end
            assert(options.Range(2) > options.Range(1), ...
                'QuantizationEngine:InvalidRange', ...
                'Range(2) must be greater than Range(1).');
            obj.BitDepth = options.BitDepth;
            obj.XMin     = options.Range(1);
            obj.XMax     = options.Range(2);
            obj.NLevels  = 2^options.BitDepth;
            % Step size: range / (number of levels - 1)
            % Midrise quantizer: levels at XMin, XMin+Delta, ..., XMax
            % (Oppenheim §4.8.2, Eq. 4.79)
            obj.Delta_ = (obj.XMax - obj.XMin) / (obj.NLevels - 1);
        end

        % ── Core Quantization ────────────────────────────────────────────────────

        function x_q = quantize(obj, x)
            % QUANTIZE  Round input x to nearest quantization level.
            %
            %   x_q = qe.quantize(x)
            %
            %   Math (midrise uniform quantizer, Oppenheim §4.8.2 Eq. 4.80):
            %     x_q[n] = Delta * round((x[n] - XMin) / Delta) + XMin
            %
            %   Clipping: values outside [XMin, XMax] are clamped (saturation).
            %   This models the overload region of a real A/D converter.
            arguments
                obj
                x (:,:) double   % input signal (any shape)
            end
            % Step 1: Clamp to full-scale range (saturation / overload)
            x_clamped = max(obj.XMin, min(obj.XMax, x));

            % Step 2: Map to integer level index [0, NLevels-1]
            %         round() implements nearest-neighbor decision
            level_idx = round((x_clamped - obj.XMin) / obj.Delta_);

            % Step 3: Map back to output amplitude
            x_q = obj.XMin + level_idx * obj.Delta_;
        end

        function e = error(obj, x)
            % ERROR  Compute quantization error e[n] = x_Q[n] - x[n].
            %
            %   e = qe.error(x)
            %
            %   The error e[n] lies in [-Delta/2, +Delta/2] for non-overloaded inputs.
            %   The statistical model assumes e[n] ~ Uniform(-Delta/2, +Delta/2).
            %
            %   Used in Exp 09 to:
            %     1. Plot e[n] time series — should look "noise-like" for B >= 6
            %     2. Plot histogram — should look flat (uniform) for large B
            %     3. Compute PSD — should look white for large B
            arguments
                obj
                x (:,:) double
            end
            e = obj.quantize(x) - x;
        end

        % ── Theoretical Analysis ─────────────────────────────────────────────────

        function Delta = stepSize(obj)
            % STEPSIZE  Return quantization step size Delta.
            %   Delta = 2*XMax / (2^B - 1)  for symmetric range [-XMax, XMax].
            Delta = obj.Delta_;
        end

        function sigma2 = theoreticalNoisePower(obj)
            % THEORETICALNOISPOWER  Expected quantization noise power.
            %
            %   sigma2 = qe.theoreticalNoisePower()
            %
            %   Math (Oppenheim §4.8.3, Eq. 4.88):
            %     sigma_e^2 = Delta^2 / 12
            %
            %   Derivation: E[e^2] = integral_{-Delta/2}^{+Delta/2} u^2 * (1/Delta) du
            %                      = Delta^2 / 12
            sigma2 = obj.Delta_^2 / 12;
        end

        function sqnr_db = theoreticalSQNR(obj)
            % THEORETICALSQNR  Theoretical SQNR for a full-scale sinusoid.
            %
            %   sqnr_db = qe.theoreticalSQNR()
            %
            %   Math (Oppenheim §4.8.3, Eq. 4.95):
            %     For a sinusoid x(t) = A*cos(omega*t), amplitude A = XMax:
            %       Signal power = A^2 / 2
            %       Noise power  = Delta^2 / 12
            %       SQNR = 10*log10( (A^2/2) / (Delta^2/12) )
            %            ≈ 6.02*B + 1.76  [dB]
            %
            %   The 6.02 dB/bit rule: each additional bit halves Delta,
            %   reducing noise power by a factor of 4 (+6.02 dB SQNR).
            A          = (obj.XMax - obj.XMin) / 2;   % half-range amplitude
            sig_power  = A^2 / 2;                     % sinusoid power
            nse_power  = obj.theoreticalNoisePower();
            sqnr_db    = 10 * log10(sig_power / nse_power);
        end

        function sqnr_db = empiricalSQNR(obj, x)
            % EMPIRICALSQNR  Measure actual SQNR from a given signal x.
            %
            %   sqnr_db = qe.empiricalSQNR(x)
            %
            %   Compares theoretical prediction to actual noise at this bit depth.
            %   Use alongside theoreticalSQNR() in the results table of Exp 09.
            arguments
                obj
                x (1,:) double
            end
            sqnr_db = OceanMath.Signal.SpectralAnalyzer.sqnr(obj.quantize(x), x);
        end

        function sqnr_theory = sqnrFormula(~, B)
            % SQNRFORMULA  Evaluate 6.02*B + 1.76 for given bit depth(s).
            %
            %   sqnr_theory = qe.sqnrFormula(B)
            %
            %   Works on scalar or vector B for plotting SQNR vs. B curve.
            sqnr_theory = 6.02 * B + 1.76;
        end

        % ── Metadata ─────────────────────────────────────────────────────────────

        function s = getMetadata(obj)
            % GETMETADATA  Return description struct for agents and OceanLearn.
            s.ClassName            = 'QuantizationEngine';
            s.ConceptNodeId        = obj.ConceptNodeId;
            s.TextbookRef          = obj.TextbookRef;
            s.Description          = obj.ClassDescription;
            s.BitDepth             = obj.BitDepth;
            s.NLevels              = obj.NLevels;
            s.Range                = [obj.XMin, obj.XMax];
            s.StepSize_Delta       = obj.Delta_;
            s.NoisePower_theory    = obj.theoreticalNoisePower();
            s.SQNR_theory_dB       = obj.theoreticalSQNR();
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print quantizer configuration and theory predictions.
            fprintf('=== QuantizationEngine ===\n');
            fprintf('  Bit depth B         : %d\n',   obj.BitDepth);
            fprintf('  Levels L = 2^B      : %d\n',   obj.NLevels);
            fprintf('  Full-scale range    : [%g, %g]\n', obj.XMin, obj.XMax);
            fprintf('  Step size Delta     : %g\n',   obj.Delta_);
            fprintf('  Noise power (theory): %g  (Delta^2/12)\n', obj.theoreticalNoisePower());
            fprintf('  SQNR (theory)       : %.2f dB  (6.02*%d + 1.76)\n', ...
                obj.theoreticalSQNR(), obj.BitDepth);
            fprintf('  Concept node        : %s\n', obj.ConceptNodeId);
        end
    end
end
