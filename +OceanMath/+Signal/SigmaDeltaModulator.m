classdef SigmaDeltaModulator
    % SIGMADELTAMODULATOR  First and second-order sigma-delta ADC simulation.
    %
    %   PEDAGOGICAL PURPOSE:
    %     Sigma-Delta (ΣΔ) modulation achieves high-resolution A/D conversion
    %     from a LOW bit-depth quantizer (typically 1 bit) by combining:
    %
    %       1. OVERSAMPLING: Sample at OSR times the Nyquist rate.
    %          Direct benefit: +3 dB SQNR per 2x oversampling (spread noise
    %          over wider bandwidth, then filter to keep only signal band).
    %
    %       2. NOISE SHAPING: The feedback loop shapes the quantization noise
    %          spectrum, pushing noise energy OUT of the signal band into
    %          the high-frequency region (where it is later filtered away).
    %
    %     THE LOOP (1st-order, Oppenheim §4.9.2):
    %       u[n] = x[n] - y[n-1]        (error between input and feedback)
    %       v[n] = u[n] + v[n-1]        (integrator: accumulates the error)
    %       y[n] = Q{v[n]}              (1-bit quantizer: sign(v[n]))
    %
    %     SIGNAL TRANSFER FUNCTION: STF(z) = z^{-1}   (unit delay — signal passes)
    %     NOISE TRANSFER FUNCTION:  NTF(z) = 1 - z^{-1}  (highpass — shapes noise)
    %
    %     NOISE PSD after shaping:
    %       S_e(omega) = (Delta^2/12) * |1 - e^{-j*omega}|^2
    %                  = (Delta^2/12) * 4 * sin^2(omega/2)    [highpass shape!]
    %
    %     In-band noise power (after LPF at omega_B = pi/OSR):
    %       sigma_shaped^2 ≈ (Delta^2/12) * (pi^2 / (3 * OSR^3))
    %       SQNR gain vs. Nyquist: ~9 dB per octave (doubling OSR).
    %
    %   This is the payoff of Chapter 4: the 1-bit bitstream LOOKS like noise,
    %   but after reconstruction filtering it yields high-fidelity audio.
    %   The noise shaping is a deliberate application of spectral folding in reverse.
    %
    %   USAGE:
    %     sdm    = OceanMath.Signal.SigmaDeltaModulator('Order', 1, 'OSR', 64);
    %     y      = sdm.modulate(x);               % 1-bit bitstream
    %     x_rec  = sdm.reconstruct(y, fs_in, fs_out); % LPF + decimate
    %     sqnr   = sdm.measureSQNR(x, y, fs_in);  % in-band SQNR
    %     [f, S] = sdm.noisePSD(y, fs_in);        % shaped noise PSD
    %
    %   REFERENCE: Oppenheim & Schafer 3rd Ed.
    %              §4.9 — Oversampling and Noise Shaping in A/D Conversion
    %              §4.9.2 — First-Order Noise Shaping, Eq. 4.105–4.115
    %              §4.9.3 — Higher-Order Noise Shaping
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    properties (SetAccess = private)
        Order   (1,1) double   % Loop order: 1 (first-order) or 2 (second-order)
        OSR     (1,1) double   % Oversampling ratio = fs_in / (2 * f_signal)
        Quant   OceanMath.Signal.QuantizationEngine  % Internal quantizer (1-bit default)
    end

    properties (Constant, Hidden)
        ClassDescription = ['First/second-order Sigma-Delta ADC modulator. Combines '...
            'oversampling and noise shaping to achieve high SQNR from a 1-bit quantizer. '...
            'NTF = 1 - z^{-1} (1st order) shapes noise to high frequencies. '...
            'Demonstrates that 1-bit + oversampling >> B-bit at Nyquist rate. '...
            'See Oppenheim & Schafer §4.9.2-4.9.3 Eq. 4.105-4.115.'];
        ConceptNodeId = 'ch4-sigma-delta-noise-shaping';
        TextbookRef   = 'Oppenheim & Schafer 3rd Ed., §4.9.2 Eq. 4.105–4.115';
        PackagePath   = '+OceanMath/+Signal/SigmaDeltaModulator.m';
    end

    methods
        % ── Constructor ──────────────────────────────────────────────────────────
        function obj = SigmaDeltaModulator(options)
            % SIGMADELTAMODULATOR  Construct sigma-delta modulator.
            %
            %   sdm = SigmaDeltaModulator('Order', 1, 'OSR', 64)
            %   sdm = SigmaDeltaModulator('Order', 2, 'OSR', 32, 'BitDepth', 1)
            %
            %   Parameters:
            %     Order    — Loop order: 1 or 2 (default 1)
            %     OSR      — Oversampling ratio (default 64)
            %     BitDepth — Quantizer bit depth inside the loop (default 1)
            %     Range    — Full-scale input range (default [-1, 1])
            arguments
                options.Order    (1,1) double {mustBeMember(options.Order, [1,2])} = 1
                options.OSR      (1,1) double {mustBePositive} = 64
                options.BitDepth (1,1) double {mustBePositive} = 1
                options.Range    (1,2) double = [-1, 1]
            end
            obj.Order = options.Order;
            obj.OSR   = options.OSR;
            obj.Quant = OceanMath.Signal.QuantizationEngine( ...
                'BitDepth', options.BitDepth, 'Range', options.Range);
        end

        % ── Core Modulation ──────────────────────────────────────────────────────

        function y = modulate(obj, x)
            % MODULATE  Run the sigma-delta feedback loop.
            %
            %   y = sdm.modulate(x)
            %
            %   For Order = 1 (Oppenheim §4.9.2, Eq. 4.107–4.109):
            %     State: integrator output v (scalar, persists between samples)
            %     Loop:
            %       u[n] = x[n] - y[n-1]     (innovation: input minus feedback)
            %       v[n] = v[n-1] + u[n]      (first-order integrator)
            %       y[n] = Q{v[n]}            (quantize integrator output)
            %
            %   For Order = 2 (§4.9.3):
            %     Two integrators in cascade, feedback from y to both.
            %     NTF(z) = (1 - z^{-1})^2   [steeper highpass, more noise shaping]
            %     SQNR improves by ~15 dB per octave (vs ~9 dB for 1st order).
            %
            %   Parameters:
            %     x — input signal (1D row vector). Should be bandlimited to
            %         f_signal << fs_in / 2.  Amplitude must fit within Quant range.
            %
            %   Returns:
            %     y — quantized output bitstream (same length as x).
            %         For 1-bit quantizer: y in {-1, +1} (or two-level values).
            arguments
                obj
                x (1,:) double
            end
            N = length(x);
            y = zeros(1, N);

            if obj.Order == 1
                % ── First-Order ΣΔ Loop ──────────────────────────────────────────
                % State: v = integrator accumulator
                v = 0;
                y_prev = 0;
                for n = 1:N
                    % Error accumulator: difference between input and last output
                    u    = x(n) - y_prev;
                    % First-order integrator: v[n] = v[n-1] + u[n]
                    v    = v + u;
                    % Quantize integrator output
                    y(n) = obj.Quant.quantize(v);
                    y_prev = y(n);
                end

            else
                % ── Second-Order ΣΔ Loop ─────────────────────────────────────────
                % Two cascaded integrators with two feedback paths.
                % NTF(z) = (1 - z^{-1})^2  =>  ~15 dB/octave gain vs Nyquist rate
                %
                % Loop equations (Candy & Temes, 1992):
                %   e1[n] = x[n]  - y[n-1]
                %   e2[n] = e1[n] - y[n-1]
                %   v1[n] = v1[n-1] + e1[n]   (first integrator)
                %   v2[n] = v2[n-1] + e2[n]   (second integrator)
                %   y[n]  = Q{v2[n]}
                v1 = 0;
                v2 = 0;
                y_prev = 0;
                for n = 1:N
                    e1   = x(n) - y_prev;
                    e2   = e1   - y_prev;
                    v1   = v1 + e1;
                    v2   = v2 + e2;
                    y(n) = obj.Quant.quantize(v2);
                    y_prev = y(n);
                end
            end
        end

        % ── Reconstruction ────────────────────────────────────────────────────────

        function x_rec = reconstruct(obj, y, fs_in, fs_out, options)
            % RECONSTRUCT  Recover baseband signal from the 1-bit bitstream.
            %
            %   x_rec = sdm.reconstruct(y, fs_in, fs_out)
            %
            %   Operation:
            %     1. Apply lowpass FIR filter with cutoff = fs_out / 2 (Hz)
            %        (removes the shaped high-frequency noise)
            %     2. Decimate by OSR = fs_in / fs_out
            %
            %   This is the "decimation filter" stage of the sigma-delta ADC.
            %   After this step, the 1-bit 64 kHz bitstream becomes a 1 kHz
            %   high-quality audio signal.
            arguments
                obj
                y      (1,:) double   % Input bitstream at fs_in
                fs_in  (1,1) double {mustBePositive}   % Bitstream sample rate
                fs_out (1,1) double {mustBePositive}   % Target output rate
                options.NTaps (1,1) double = 256       % Reconstruction filter length
            end
            % Decimation factor
            M_dec = round(fs_in / fs_out);
            assert(M_dec >= 1, 'SigmaDeltaModulator:RateError', ...
                'fs_in must be >= fs_out.');

            % Design lowpass FIR reconstruction filter
            % Cutoff = (fs_out/2) / (fs_in/2) = fs_out/fs_in (normalized to Nyquist of fs_in)
            fc_norm = fs_out / fs_in;   % normalized cutoff in [0, 1] (1 = Nyquist)
            n_taps  = options.NTaps;
            n_vec   = -(n_taps-1)/2 : (n_taps-1)/2;
            h_ideal = fc_norm * sinc(fc_norm * n_vec);
            % Hamming window to control stopband attenuation
            win     = 0.54 - 0.46*cos(2*pi*(0:n_taps-1)/(n_taps-1));
            h_rec   = h_ideal .* win;
            h_rec   = h_rec / sum(h_rec);

            % Apply reconstruction filter
            y_filt = conv(y, h_rec, 'same');

            % Decimate to fs_out
            x_rec = y_filt(1 : M_dec : end);
        end

        % ── Analysis ──────────────────────────────────────────────────────────────

        function sqnr_db = measureSQNR(obj, x_ref, y, fs_in)
            % MEASURESQNR  Compute in-band SQNR of the sigma-delta output.
            %
            %   sqnr_db = sdm.measureSQNR(x_ref, y, fs_in)
            %
            %   Reconstructs x_rec from y, then computes SQNR against x_ref.
            %   Only the in-band portion (f <= fs_in/OSR) is compared.
            %
            %   Allows plotting SQNR vs OSR to verify the 9 dB/octave law for
            %   1st-order sigma-delta (Oppenheim §4.9.2, Eq. 4.115).
            arguments
                obj
                x_ref (1,:) double   % Reference input signal at fs_in
                y     (1,:) double   % Modulator output bitstream at fs_in
                fs_in (1,1) double {mustBePositive}
            end
            fs_out  = fs_in / obj.OSR;
            x_rec   = obj.reconstruct(y, fs_in, fs_out);
            % Align lengths (reconstruction may differ by a few samples due to filter)
            n_cmp   = min(length(x_ref), length(x_rec) * obj.OSR);
            x_ref_d = x_ref(1 : obj.OSR : n_cmp);   % decimate ref to same rate
            n_cmp2  = min(length(x_ref_d), length(x_rec));
            sqnr_db = OceanMath.Signal.SpectralAnalyzer.sqnr( ...
                x_rec(1:n_cmp2), x_ref_d(1:n_cmp2));
        end

        function sqnr_theory = theoreticalSQNR(obj)
            % THEORETICALSQNR  Theoretical in-band SQNR for this OSR and order.
            %
            %   sqnr_theory = sdm.theoreticalSQNR()
            %
            %   1st-order (Oppenheim §4.9.2, Eq. 4.115):
            %     SQNR_1 ≈ 10*log10( 9*OSR^3 / pi^2 )   [dB above Nyquist]
            %
            %   2nd-order (§4.9.3):
            %     SQNR_2 ≈ 10*log10( 15*OSR^5 / (2*pi^4) )
            %
            %   For a full-scale sinusoid input.
            if obj.Order == 1
                % In-band noise power: sigma_1^2 = (Delta^2/12) * pi^2 / (3*OSR^3)
                % Signal power (full-scale sinusoid): A^2/2 where A = range/2
                % SQNR = 10*log10(9*OSR^3 / pi^2)  +  SQNR_ref_1bit
                sqnr_theory = 10 * log10(9 * obj.OSR^3 / pi^2);
            else
                % 2nd order: steeper noise shaping, OSR^5 dependence
                sqnr_theory = 10 * log10(15 * obj.OSR^5 / (2 * pi^4));
            end
        end

        % ── Metadata ─────────────────────────────────────────────────────────────

        function s = getMetadata(obj)
            % GETMETADATA  Return description struct for agents and OceanLearn.
            s.ClassName          = 'SigmaDeltaModulator';
            s.ConceptNodeId      = obj.ConceptNodeId;
            s.TextbookRef        = obj.TextbookRef;
            s.Description        = obj.ClassDescription;
            s.Order              = obj.Order;
            s.OSR                = obj.OSR;
            s.BitDepth           = obj.Quant.BitDepth;
            s.SQNR_theory_dB     = obj.theoreticalSQNR();
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print sigma-delta configuration and theoretical SQNR.
            fprintf('=== SigmaDeltaModulator ===\n');
            fprintf('  Loop order         : %d\n',   obj.Order);
            fprintf('  Oversampling ratio : %d x\n', obj.OSR);
            fprintf('  Quantizer bits     : %d\n',   obj.Quant.BitDepth);
            fprintf('  Theoretical SQNR   : %.1f dB\n', obj.theoreticalSQNR());
            if obj.Order == 1
                fprintf('  NTF                : 1 - z^{-1}  (highpass, 1st order)\n');
                fprintf('  SQNR gain/octave   : ~9 dB\n');
            else
                fprintf('  NTF                : (1 - z^{-1})^2  (steeper, 2nd order)\n');
                fprintf('  SQNR gain/octave   : ~15 dB\n');
            end
            fprintf('  Concept node       : %s\n', obj.ConceptNodeId);
        end
    end
end
