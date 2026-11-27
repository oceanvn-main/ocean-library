classdef DifferenceSpectrumFilter < handle
    % DIFFERENCESPECTRUMFILTER  Nth-order differencing for spectral filtering.
    %
    %   Applies the backshift-difference operator (1 - B)^d to a signal,
    %   eliminating polynomial trends of degree up to d-1 and high-passing
    %   the power spectrum.  Before and after one-sided spectra are computed
    %   using Welch's method (via pwelch) and stored for inspection.
    %
    %   Spectral interpretation
    %     - 1st-order difference: transfer function H(z) = 1 - z^{-1},
    %       frequency response |H(f)| = 2|sin(pi*f/fs)|.
    %       Attenuates DC and low frequencies; passes high frequencies.
    %     - d-th order:           |H(f)|^d = [2|sin(pi*f/fs)|]^d
    %       Steeper high-pass roll-off as d increases.
    %
    %   Parseval verification (optional)
    %     Total power from Welch PSD must integrate to signal variance.
    %     A Parseval error > 1 % triggers a warning.
    %
    %   Usage
    %     dsf = OceanPipeline.Transforms.DifferenceSpectrumFilter('Order', 1);
    %     report = dsf.filter(x, fs);
    %
    %     report fields:
    %       .x_orig       — original signal (column vector)
    %       .x_diff       — differenced signal (length N-d)
    %       .order        — differencing order d
    %       .fs           — sampling frequency
    %       .f_orig       — frequency axis for original PSD (Hz)
    %       .psd_orig     — one-sided PSD of original signal
    %       .f_diff       — frequency axis for differenced PSD (Hz)
    %       .psd_diff     — one-sided PSD of differenced signal
    %       .tf_gain      — theoretical |H(f)|^d for every f_orig point
    %       .parseval_err_orig  — Parseval % error (original)
    %       .parseval_err_diff  — Parseval % error (differenced)
    %       .var_orig     — variance of original
    %       .var_diff     — variance of differenced
    %       .summary      — printable text summary
    %
    %   See also: OceanMath.Operators.DifferenceOperator1D, pwelch

    properties
        Order   (1,1) double {mustBePositive, mustBeInteger} = 1
        % Number of Welch segments ([] = auto from sqrt(N))
        NumSegments (1,1) double = 0
        % Fraction overlap for Welch (0 = none, 0.5 = 50 %)
        Overlap (1,1) double {mustBeInRange(Overlap,0,0.99)} = 0.5
        % Window type for Welch: 'hann', 'hamming', 'blackman', 'rect'
        WindowType (1,1) string = "hann"
        % Verify Parseval's theorem after PSD estimation
        VerifyParseval (1,1) logical = true
    end

    % -----------------------------------------------------------------------
    methods
        function obj = DifferenceSpectrumFilter(options)
            % DIFFERENCESPECTRUMFILTER  Constructor.
            arguments
                options.Order          (1,1) double  = 1
                options.NumSegments    (1,1) double  = 0
                options.Overlap        (1,1) double  = 0.5
                options.WindowType     (1,1) string  = "hann"
                options.VerifyParseval (1,1) logical = true
            end

            obj.Order          = options.Order;
            obj.NumSegments    = options.NumSegments;
            obj.Overlap        = options.Overlap;
            obj.WindowType     = options.WindowType;
            obj.VerifyParseval = options.VerifyParseval;
        end

        % -------------------------------------------------------------------
        function report = filter(obj, x, fs)
            % FILTER  Apply d-th order differencing and compute before/after spectra.
            %
            %   x  — real column vector of N samples
            %   fs — sampling frequency in Hz (default = 1.0)
            %
            %   Returns a struct with fields documented in the class header.

            arguments
                obj
                x  (:,1) double
                fs (1,1) double = 1.0
            end

            x = x(:);   % enforce column
            N = numel(x);
            d = obj.Order;

            % ---- 1. Apply (1-B)^d iteratively ----------------------------
            % Each application reduces length by 1.
            x_diff = x;
            for k = 1:d
                x_diff = diff(x_diff);
            end

            % ---- 2. Welch parameters -------------------------------------
            winLen = obj.computeWindowLength(N);
            noverlap = floor(winLen * obj.Overlap);
            win = obj.makeWindow(winLen);

            winLen_d = obj.computeWindowLength(numel(x_diff));
            noverlap_d = floor(winLen_d * obj.Overlap);
            win_d = obj.makeWindow(winLen_d);

            % ---- 3. PSD via pwelch ---------------------------------------
            [psd_orig, f_orig] = pwelch(x,       win,   noverlap,   [], fs);
            [psd_diff, f_diff] = pwelch(x_diff,  win_d, noverlap_d, [], fs);

            % ---- 4. Theoretical transfer function gain |H(f)|^d ----------
            % |H(f)| = 2*|sin(pi*f/fs)|  for one-sided frequencies
            tf_gain = (2 * abs(sin(pi * f_orig / fs))) .^ d;

            % ---- 5. Parseval verification ---------------------------------
            df_orig = f_orig(2) - f_orig(1);
            df_diff = f_diff(2) - f_diff(1);

            % One-sided Welch integrates to variance (var = sigma^2 for zero-mean)
            psd_power_orig = sum(psd_orig) * df_orig;
            psd_power_diff = sum(psd_diff) * df_diff;

            var_orig = var(x);
            var_diff = var(x_diff);

            parseval_err_orig = abs(psd_power_orig - var_orig) / (var_orig + eps) * 100;
            parseval_err_diff = abs(psd_power_diff - var_diff) / (var_diff + eps) * 100;

            if obj.VerifyParseval
                if parseval_err_orig > 1.0
                    warning("DifferenceSpectrumFilter:ParsevalOrig", ...
                        "Parseval error (original) = %.2f %%. PSD integral may be inaccurate.", ...
                        parseval_err_orig);
                end
                if parseval_err_diff > 1.0
                    warning("DifferenceSpectrumFilter:ParsevalDiff", ...
                        "Parseval error (differenced) = %.2f %%. PSD integral may be inaccurate.", ...
                        parseval_err_diff);
                end
            end

            % ---- 6. Build summary text -----------------------------------
            summary = sprintf( ...
                "DifferenceSpectrumFilter | Order=%d  fs=%.4g Hz\n" + ...
                "  Original  : N=%d  var=%.6g  Parseval err=%.2f%%\n" + ...
                "  Differenced: N=%d  var=%.6g  Parseval err=%.2f%%\n" + ...
                "  Variance reduction factor: %.4f  (%.1f dB)", ...
                d, fs, ...
                N, var_orig, parseval_err_orig, ...
                numel(x_diff), var_diff, parseval_err_diff, ...
                var_diff / (var_orig + eps), ...
                10*log10(var_diff / (var_orig + eps) + eps));

            % ---- 7. Pack report ------------------------------------------
            report.x_orig            = x;
            report.x_diff            = x_diff;
            report.order             = d;
            report.fs                = fs;
            report.f_orig            = f_orig;
            report.psd_orig          = psd_orig;
            report.f_diff            = f_diff;
            report.psd_diff          = psd_diff;
            report.tf_gain           = tf_gain;
            report.parseval_err_orig = parseval_err_orig;
            report.parseval_err_diff = parseval_err_diff;
            report.var_orig          = var_orig;
            report.var_diff          = var_diff;
            report.summary           = summary;
        end

        % -------------------------------------------------------------------
        function printSummary(~, report)
            % PRINTSUMMARY  Print the text summary from a filter report.
            fprintf("%s\n", report.summary);
        end
    end

    % -----------------------------------------------------------------------
    methods (Access = private)

        function winLen = computeWindowLength(obj, N)
            % COMPUTEWINDOWLENGTH  Determine Welch segment length.
            %   If NumSegments == 0 (auto): segment length = floor(sqrt(N))
            %   rounded up to the next power of 2 for efficient FFT.
            if obj.NumSegments == 0
                raw = floor(sqrt(N));
                winLen = 2^nextpow2(raw);
                winLen = min(winLen, N);   % can't exceed signal length
            else
                winLen = floor(N / obj.NumSegments);
                winLen = max(winLen, 8);   % minimum useful segment
            end
        end

        function w = makeWindow(obj, winLen)
            % MAKEWINDOW  Return a window vector of length winLen.
            switch lower(obj.WindowType)
                case 'hann'
                    w = hann(winLen);
                case 'hamming'
                    w = hamming(winLen);
                case 'blackman'
                    w = blackman(winLen);
                case {'rect', 'rectangular', 'boxcar'}
                    w = rectwin(winLen);
                otherwise
                    warning("DifferenceSpectrumFilter:UnknownWindow", ...
                        "Unknown window '%s'; defaulting to Hann.", obj.WindowType);
                    w = hann(winLen);
            end
        end
    end
end
