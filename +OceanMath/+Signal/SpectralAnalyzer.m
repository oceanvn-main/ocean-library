classdef SpectralAnalyzer
    % SPECTRALANALYZER  Static spectral analysis utilities for 1D discrete signals.
    %
    %   PEDAGOGICAL PURPOSE:
    %     Spectral analysis is the bridge between the time-domain samples x[n] and
    %     the frequency-domain picture X(e^{j*omega}) that explains everything that
    %     happens during sampling, aliasing, reconstruction, and noise shaping.
    %
    %     Common DSP bugs arise from incorrect frequency axis construction:
    %       - Forgetting fftshift (spectrum appears to start at DC, not -fs/2)
    %       - Wrong normalization (magnitude off by N or fs)
    %       - Confusing normalized angular (rad/sample) vs physical (Hz) frequency
    %
    %     This class centralizes all spectral computations with explicit axis
    %     conventions so every experiment uses identical, correct representations.
    %
    %   FREQUENCY AXIS CONVENTIONS (defined once, used everywhere):
    %     Two-sided physical (Hz):    f in [-fs/2, +fs/2]
    %     One-sided physical (Hz):    f in [0, fs/2]
    %     Normalized angular (rad/s): omega in [-pi, +pi]  (omega = 2*pi*f/fs)
    %
    %   ALL METHODS ARE STATIC — no instantiation required:
    %     [f, X] = OceanMath.Signal.SpectralAnalyzer.spectrumTwoSided(x, fs)
    %     [f, X] = OceanMath.Signal.SpectralAnalyzer.spectrumOneSided(x, fs)
    %     [t_c, f_c, S] = OceanMath.Signal.SpectralAnalyzer.stft(x, fs, ...)
    %     peaks = OceanMath.Signal.SpectralAnalyzer.findPeaks(f, X, n)
    %     f_alias = OceanMath.Signal.SpectralAnalyzer.aliasFormula(f0, fs)
    %     sqnr = OceanMath.Signal.SpectralAnalyzer.sqnr(x_q, x_ref)
    %
    %   REFERENCE: Oppenheim & Schafer 3rd Ed.
    %              §4.2 — Frequency-Domain Representation of Sampling
    %              §4.8.3 — Analysis of Quantization Errors
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    properties (Constant, Hidden)
        ClassDescription = ['Static spectral analysis utilities. Centralizes FFT '...
            'axis construction, STFT spectrogram, peak detection, alias frequency '...
            'prediction, and SQNR measurement. Eliminates frequency-axis bugs across '...
            'all Ch.4 experiments. See Oppenheim & Schafer §4.2, §4.8.3.'];
        ConceptNodeId = 'ch4-spectral-analysis';
        TextbookRef   = 'Oppenheim & Schafer 3rd Ed., §4.2 Eq. 4.23–4.28';
        PackagePath   = '+OceanMath/+Signal/SpectralAnalyzer.m';
    end

    methods (Static)

        % ── Core Spectrum ─────────────────────────────────────────────────────────

        function [f, X_mag] = spectrumTwoSided(x, fs)
            % SPECTRUMTWOSIDED  Compute the two-sided magnitude spectrum of x.
            %
            %   [f, X_mag] = SpectralAnalyzer.spectrumTwoSided(x, fs)
            %
            %   Math (DTFT sampled at N points):
            %     X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
            %     |X(f)| = |fft(x)| / N     (amplitude normalization)
            %
            %   Frequency axis (Eq. 4.23):
            %     f = (-N/2 : N/2-1) * (fs / N)   [Hz, centered at DC]
            %
            %   Output:
            %     f     — frequency vector in Hz,   length N, range [-fs/2, +fs/2)
            %     X_mag — magnitude spectrum,       length N, normalized by N
            %
            %   Note: fftshift moves the DC component to the center.
            arguments
                x  (1,:) double
                fs (1,1) double {mustBePositive}
            end
            N     = length(x);
            % Raw FFT, then shift DC to center
            X_raw = fftshift(fft(x, N));
            % Normalize so single-tone amplitude = A (not A*N)
            X_mag = abs(X_raw) / N;
            % Physical frequency axis: bin k -> f = k * fs/N, shifted by N/2
            f     = (-floor(N/2) : ceil(N/2)-1) * (fs / N);
        end

        function [f, X_mag] = spectrumOneSided(x, fs)
            % SPECTRUMONESIDED  Compute the one-sided (positive-frequency) spectrum.
            %
            %   [f, X_mag] = SpectralAnalyzer.spectrumOneSided(x, fs)
            %
            %   One-sided spectrum: keep only f in [0, fs/2].
            %   Negative-frequency energy is folded into positive side (x2 factor),
            %   except at DC (f=0) and Nyquist (f=fs/2) which are not doubled.
            %
            %   Used when: signal is real-valued and we only care about [0, fs/2].
            arguments
                x  (1,:) double
                fs (1,1) double {mustBePositive}
            end
            N       = length(x);
            X_full  = fft(x, N);
            X_mag   = abs(X_full) / N;
            % Keep DC through Nyquist
            n_keep  = floor(N/2) + 1;
            X_mag   = X_mag(1:n_keep);
            % Double interior bins (fold in negative-frequency mirror)
            X_mag(2:end-1) = 2 * X_mag(2:end-1);
            f = (0 : n_keep-1) * (fs / N);
        end

        function [f, X_norm] = spectrumNormalized(x)
            % SPECTRUMNORMALIZED  Two-sided spectrum with normalized angular frequency.
            %
            %   [f, X_norm] = SpectralAnalyzer.spectrumNormalized(x)
            %
            %   Frequency axis: omega in [-pi, +pi] (rad/sample)
            %   Used when: comparing discrete-time filter responses against
            %              analog prototype responses (Exp 04, Exp 05).
            arguments
                x (1,:) double
            end
            N      = length(x);
            X_raw  = fftshift(fft(x, N));
            X_norm = abs(X_raw) / N;
            % Normalized angular frequency axis: [-pi, +pi)
            f      = (-floor(N/2) : ceil(N/2)-1) * (2*pi / N);
        end

        % ── STFT Spectrogram ─────────────────────────────────────────────────────

        function [t_centers, f_bins, S_mag] = stft(x, fs, options)
            % STFT  Short-Time Fourier Transform (spectrogram).
            %
            %   [t_c, f_c, S] = SpectralAnalyzer.stft(x, fs)
            %   [t_c, f_c, S] = SpectralAnalyzer.stft(x, fs, 'WinLen', 512, 'Hop', 64)
            %
            %   Math (for each frame starting at sample m):
            %     X_m(e^{j*omega}) = sum_{n=0}^{L-1} x[m+n] * w[n] * exp(-j*omega*n)
            %
            %   where w[n] is the Hann analysis window of length WinLen.
            %
            %   Output:
            %     t_centers — center time of each STFT frame (seconds)
            %     f_bins    — frequency bins in Hz,  range [0, fs/2]
            %     S_mag     — magnitude spectrogram, size [n_freqs x n_frames]
            %
            %   Used in Exp 02 to visualize the "bouncing" alias trajectory
            %   as an oversampled chirp is downsampled below Nyquist.
            arguments
                x   (1,:) double
                fs  (1,1) double {mustBePositive}
                options.WinLen (1,1) double {mustBePositive, mustBeInteger} = 512
                options.Hop    (1,1) double {mustBePositive, mustBeInteger} = 64
                options.NFFT   (1,1) double = 0   % 0 = use WinLen
            end
            L    = options.WinLen;
            hop  = options.Hop;
            nfft = options.NFFT;
            if nfft == 0, nfft = L; end

            % Hann window — reduces spectral leakage at frame boundaries
            win = 0.5 * (1 - cos(2*pi * (0:L-1)' / (L-1)));

            N = length(x);
            % Frame start indices
            starts = 1 : hop : (N - L + 1);
            n_frames  = length(starts);
            n_freqs   = floor(nfft/2) + 1;

            S_mag = zeros(n_freqs, n_frames);
            for k = 1:n_frames
                seg          = x(starts(k) : starts(k)+L-1)';
                frame_fft    = fft(seg .* win, nfft);
                % One-sided magnitude, normalized by window energy
                S_mag(:, k)  = abs(frame_fft(1:n_freqs)) / sum(win);
            end

            % Time axis: center of each frame
            t_centers = ((starts - 1) + (L-1)/2) / fs;
            % Frequency axis: [0, fs/2]
            f_bins    = (0 : n_freqs-1) * (fs / nfft);
        end

        % ── Peak Detection ────────────────────────────────────────────────────────

        function peaks = findPeaks(f, X_mag, n_peaks, min_prominence)
            % FINDPEAKS  Locate the N strongest peaks in a magnitude spectrum.
            %
            %   peaks = SpectralAnalyzer.findPeaks(f, X_mag, n_peaks)
            %
            %   Returns a struct array with fields:
            %     peaks(k).frequency   — frequency of k-th peak (Hz)
            %     peaks(k).magnitude   — magnitude at peak
            %     peaks(k).bin_index   — FFT bin index
            %
            %   Used in Exp 01 to verify that the observed alias frequency
            %   matches the theoretical prediction.
            arguments
                f             (1,:) double
                X_mag         (1,:) double
                n_peaks       (1,1) double {mustBePositive} = 3
                min_prominence (1,1) double = 0
            end
            % Simple local maxima detection (no toolbox required)
            is_peak = [false, X_mag(2:end-1) > X_mag(1:end-2) & ...
                               X_mag(2:end-1) > X_mag(3:end), false];
            peak_idx = find(is_peak);

            % Apply prominence threshold
            if min_prominence > 0
                peak_idx = peak_idx(X_mag(peak_idx) >= min_prominence);
            end

            % Sort by magnitude descending
            [~, sort_ord] = sort(X_mag(peak_idx), 'descend');
            peak_idx = peak_idx(sort_ord);

            % Return top n_peaks
            n_ret = min(n_peaks, length(peak_idx));
            peaks = struct('frequency', cell(1,n_ret), ...
                           'magnitude', cell(1,n_ret), ...
                           'bin_index', cell(1,n_ret));
            for k = 1:n_ret
                peaks(k).frequency = f(peak_idx(k));
                peaks(k).magnitude = X_mag(peak_idx(k));
                peaks(k).bin_index = peak_idx(k);
            end
        end

        % ── Alias Frequency Prediction ────────────────────────────────────────────

        function f_alias = aliasFormula(f0, fs)
            % ALIASFORMULA  Predict the aliased frequency of a tone sampled below Nyquist.
            %
            %   f_alias = SpectralAnalyzer.aliasFormula(f0, fs)
            %
            %   Math (spectral folding formula, Oppenheim §4.2):
            %     f_alias = | mod(f0 + fs/2, fs) - fs/2 |
            %
            %   This maps any input frequency f0 to its baseband alias in [0, fs/2].
            %   The formula produces the characteristic "sawtooth bounce" pattern
            %   that reverses direction every time f0 crosses a multiple of fs/2.
            %
            %   Examples (fs = 400 Hz):
            %     f0 = 300 Hz  ->  f_alias = |mod(500, 400) - 200| = |100 - 200| = 100 Hz
            %     f0 = 100 Hz  ->  f_alias = |mod(300, 400) - 200| = |300 - 200| = 100 Hz
            %     f0 = 200 Hz  ->  f_alias = |mod(400, 400) - 200| = |0 - 200|   = 200 Hz (Nyquist)
            %
            %   Works on vectors: f0 may be a vector of frequencies.
            arguments
                f0 (1,:) double {mustBeNonnegative}
                fs (1,1) double {mustBePositive}
            end
            f_alias = abs(mod(f0 + fs/2, fs) - fs/2);
        end

        % ── Signal Quality Metrics ────────────────────────────────────────────────

        function sqnr_db = sqnr(x_quantized, x_reference)
            % SQNR  Signal-to-Quantization-Noise Ratio in dB.
            %
            %   sqnr_db = SpectralAnalyzer.sqnr(x_quantized, x_reference)
            %
            %   Math (Oppenheim §4.8.3, Eq. 4.93):
            %     e[n]    = x_quantized[n] - x_reference[n]   (error signal)
            %     SQNR    = 10 * log10( mean(x_ref^2) / mean(e^2) )   [dB]
            %
            %   The theoretical value for a B-bit quantizer with a full-scale
            %   sinusoid is: SQNR_theory = 6.02*B + 1.76 dB.
            arguments
                x_quantized (1,:) double
                x_reference (1,:) double
            end
            assert(length(x_quantized) == length(x_reference), ...
                'SpectralAnalyzer:LengthMismatch', ...
                'x_quantized and x_reference must have the same length.');
            e          = x_quantized - x_reference;
            sig_power  = mean(x_reference .^ 2);
            noise_power = mean(e .^ 2);
            if noise_power < eps
                sqnr_db = Inf;
            else
                sqnr_db = 10 * log10(sig_power / noise_power);
            end
        end

        function psd_db = powerSpectralDensity(x, fs)
            % POWERSPECTRALDENSITY  One-sided Power Spectral Density (PSD) in dB/Hz.
            %
            %   psd_db = SpectralAnalyzer.powerSpectralDensity(x, fs)
            %
            %   PROBLEM MOTIVATION (Why Raw FFT Fails for Random Signals / Noise):
            %     1. Deterministic signals (sinusoids) have discrete line spectra where
            %        raw FFT magnitude |X[k]|/N accurately reports discrete peak amplitudes.
            %     2. Stochastic / random signals (quantization noise e[n], thermal noise,
            %        sigma-delta bitstream noise) have continuous power distributions.
            %     3. Taking raw FFT of a random signal produces a spectrum whose variance
            %        does NOT decrease as N -> inf (it fluctuates erratically at every bin).
            %     4. Solution: We must compute Power Spectral Density (PSD), normalizing
            %        the squared magnitude by sampling rate fs and window energy:
            %          P(f) = |FFT(x .* win)|^2 / (fs * sum(win.^2))   [V^2 / Hz]
            %        This measures how noise power is distributed per unit bandwidth (Hz),
            %        allowing accurate visualization of quantization noise floors (Exp 09)
            %        and sigma-delta noise shaping curves (Exp 10).
            %
            %   Math (Oppenheim §4.8.3, §4.9.2):
            %     win       = Hann(N)
            %     P_raw(f)  = |FFT(x .* win)|^2 / (fs * sum(win^2))
            %     psd_db(f) = 10 * log10( P_onesided(f) + eps )  [dB/Hz]
            %
            %   Returns:
            %     psd_db — One-sided PSD array in dB/Hz for positive frequencies [0, fs/2]
            arguments
                x  (1,:) double
                fs (1,1) double {mustBePositive}
            end
            N       = length(x);
            win     = hann(N)';
            win_power = sum(win.^2);             % window normalization
            X       = fft(x .* win, N);
            % ── One-Sided Conversion & Energy Conservation ──────────────────────
            % Step 1: Keep positive frequency bins [0, fs/2] (DC through Nyquist)
            n_keep  = floor(N/2) + 1;
            P       = P(1:n_keep);             % Discard negative frequency upper half
            
            % Step 2: Double interior bins (2 to end-1) to conserve total signal power.
            % Note: DC (index 1) and Nyquist (index end) sit on symmetry axes and have
            % no negative frequency counterparts, so they must NOT be doubled.
            P(2:end-1) = 2 * P(2:end-1);
            
            % Step 3: Convert Power/Hz to dB/Hz scale (eps prevents log10(0) crash)
            psd_db  = 10 * log10(P + eps);
        end

        function P = framePowerSpectrum(frames, nfft)
            % FRAMEPOWERSPECTRUM  One-sided |DFT|^2 of each column of an (already windowed) frame matrix.
            %
            %   P = SpectralAnalyzer.framePowerSpectrum(frames, nfft)
            %
            %   frames : L x T real matrix, one (windowed) analysis frame per column
            %   nfft   : DFT length, nfft >= L; each frame is zero-padded AT THE END
            %   P      : (floor(nfft/2)+1) x T,
            %              P(k+1, t) = | sum_{n=0}^{L-1} frames(n+1,t) e^{-j 2 pi k n / nfft} |^2,
            %            k = 0..floor(nfft/2), bin frequency f_k = k fs / nfft.
            %   No one-sided doubling and no normalisation (raw periodogram numerator,
            %   the quantity mel filter banks act on; same as |numpy.fft.rfft|^2).
            %   Odd nfft has no Nyquist bin. T = 0 gives a (floor(nfft/2)+1) x 0 result.
            arguments
                frames (:,:) double {mustBeReal}
                nfft (1,1) double {mustBeInteger, mustBePositive}
            end
            L = size(frames, 1);
            if nfft < L
                error('OceanMath:Signal:SpectralAnalyzer:NFFTTooSmall', ...
                    'nfft (%d) must be at least the frame length (%d).', nfft, L);
            end
            K = floor(nfft / 2) + 1;
            if size(frames, 2) == 0
                P = zeros(K, 0);
                return;
            end
            X = fft(frames, nfft, 1);
            X = X(1:K, :);
            P = real(X) .^ 2 + imag(X) .^ 2;
        end

        % ── Metadata ─────────────────────────────────────────────────────────────

        function s = getMetadata()
            % GETMETADATA  Return description struct for agents and OceanLearn.
            s.ClassName     = 'SpectralAnalyzer';
            s.ConceptNodeId = OceanMath.Signal.SpectralAnalyzer.ConceptNodeId;
            s.TextbookRef   = OceanMath.Signal.SpectralAnalyzer.TextbookRef;
            s.Description   = OceanMath.Signal.SpectralAnalyzer.ClassDescription;
            s.Methods       = {'spectrumTwoSided', 'spectrumOneSided', ...
                               'spectrumNormalized', 'stft', 'findPeaks', ...
                               'aliasFormula', 'sqnr', 'powerSpectralDensity'};
        end
    end
end
