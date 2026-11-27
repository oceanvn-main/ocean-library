classdef FourierFeatureMap < OceanStats.Preprocessing.IPreprocessor
    % FOURIERFEATUREMAP  Fourier Feature Mapping preprocessor for MLPs.
    %
    %   Projects raw low-dimensional input coordinates into a high-dimensional
    %   Fourier space to overcome the spectral bias of standard MLPs:
    %
    %     gamma(X) = [ cos(2*pi * B * X); sin(2*pi * B * X) ]
    %
    %   where B ~ N(0, sigma^2 * I) is a FROZEN (non-trainable) random matrix,
    %   sampled once during fit() and saved with the checkpoint.
    %   Each row in B is a distinct frequency vector w_m — a Monte Carlo
    %   approximation of the infinite kernel integral (Bochner's Theorem).
    %
    %   Theory:
    %     Under Bochner's thm, the inner product of these features approximates
    %     a stationary Gaussian kernel with bandwidth controlled by sigma.
    %     Higher sigma -> narrower kernel -> network behaves as a high-pass
    %     filter, learning fine-grained, high-frequency structures.
    %
    %   Choosing sigma:
    %     Use  FourierFeatureMap.recommendSigma(X)  for an analytically safe
    %     starting point based on the Nyquist limit of your grid.
    %     Use  FourierFeatureMap.sweepSigma(X, Y)   for empirical U-curve
    %     tuning (the "industry standard" method from Tancik et al.).
    %
    %   Reference:
    %     Tancik et al., "Fourier Features Let Networks Learn High Frequency
    %     Functions in Low Dimensional Domains", NeurIPS 2020.
    %     https://arxiv.org/abs/2006.10739
    %
    %   USAGE:
    %     ffm = OceanStats.Preprocessing.FourierFeatureMap();
    %     sigma = FourierFeatureMap.recommendSigma(X_raw);  % data-driven
    %     Xf  = ffm.fitTransform(X_raw, 'M', 64, 'Sigma', sigma);
    %     % ... train NN on Xf instead of X_raw ...
    %     Xf_new = ffm.transform(X_new);
    %
    % See also: OceanStats.Preprocessing

    properties (SetAccess = private)
        M        (1,1) double = 64    % Number of frequency components (output: 2M)
        Sigma    (1,1) double = 10    % Bandwidth: higher = finer detail
        B        double               % Frozen random matrix [M x InputDim] — Monte Carlo approx
        InputDim (1,1) double = 0     % Detected from input during fit
        IsFitted (1,1) logical = false
    end

    % =====================================================================
    methods   % Instance methods
    % =====================================================================

        function obj = FourierFeatureMap(opts)
            % FOURIERFEATUREMAP  Construct with optional M and Sigma defaults.
            %
            %   ffm = FourierFeatureMap()
            %   ffm = FourierFeatureMap('M', 128, 'Sigma', 5)
            %
            %   Call encode(X) to sample B and map data.
            arguments
                opts.M     (1,1) double = 64
                opts.Sigma (1,1) double = 10
            end
            obj.M     = opts.M;
            obj.Sigma = opts.Sigma;
        end

        function fit(obj, InputDim, opts)
            % FIT  Sample and freeze the random projection matrix B.
            %
            %   ffm.fit(InputDim)
            %   ffm.fit(InputDim, 'M', 128, 'Sigma', 5)
            %
            %   InputDim - number of raw input features (e.g. 2 for (a,b))
            arguments
                obj
                InputDim  (1,1) double
                opts.M    (1,1) double = obj.M
                opts.Sigma(1,1) double = obj.Sigma
            end
            obj.InputDim = InputDim;
            obj.M        = opts.M;
            obj.Sigma    = opts.Sigma;
            obj.B        = randn(obj.M, InputDim) * obj.Sigma;  % Monte Carlo approx
            obj.IsFitted = true;
        end

        function X_enc = encodeNew(obj, X_raw)
            % ENCODENEW  Apply the frozen B mapping to new data (e.g. Test/Val).
            %
            %   X_enc = ffm.encodeNew(X_raw)
            if ~obj.IsFitted
                error('OceanStats:Preprocessing:FourierFeatureMap:NotFitted', ...
                    'Call fit() or encode() before calling encodeNew().');
            end
            if size(X_raw, 1) ~= obj.InputDim
                error('OceanStats:Preprocessing:FourierFeatureMap:DimMismatch', ...
                    'Input dimension %d does not match fitted dimension %d.', ...
                    size(X_raw, 1), obj.InputDim);
            end
            
            proj  = 2 * pi * obj.B * X_raw;     % [M x N] 
            X_enc = [cos(proj); sin(proj)];     % [2M x N]
        end

        function X_enc = encode(obj, X_raw, opts)
            % ENCODE  IPreprocessor contract: fit configuration and transform in one call.
            %
            %   X_enc = ffm.encode(X_raw)
            %   X_enc = ffm.encode(X_raw, 'M', 64, 'Sigma', 10)
            arguments
                obj
                X_raw      double
                opts.M     (1,1) double = obj.M
                opts.Sigma (1,1) double = obj.Sigma
            end
            obj.fit(size(X_raw, 1), 'M', opts.M, 'Sigma', opts.Sigma);
            X_enc = obj.encodeNew(X_raw);
        end

        function d = outputDim(obj)
            % OUTPUTDIM  Returns 2*M — the dimension of the mapped features.
            d = 2 * obj.M;
        end

        function s = name(obj)
            s = 'fourier_features';
        end

        function s = describe(obj)
            % DESCRIBE  Summary string for display/logging.
            if ~obj.IsFitted
                s = sprintf('FourierFeatureMap (not fitted, M=%d, sigma=%.2f)', obj.M, obj.Sigma);
            else
                s = sprintf('FourierFeatureMap  %d -> %d  (M=%d, sigma=%.2f)', ...
                    obj.InputDim, obj.outputDim(), obj.M, obj.Sigma);
            end
        end

    end % instance methods

    % =====================================================================
    methods (Static)   % Data-driven sigma selection
    % =====================================================================

        function sigma = recommendSigma(X, opts)
            % RECOMMENDSIGMA  Analytically safe sigma from the Nyquist limit.
            %
            %   Derivation (Part 2 & 3 of Tancik analysis):
            %     Step 1 — Compute the Nyquist frequency of the discrete grid:
            %
            %       f_N = 1 / (2 * delta_x)
            %
            %     where delta_x = median nearest-neighbor spacing between
            %     training points. Median is robust to boundary outliers.
            %
            %     Step 2 — Apply the 3-sigma safety bound (99.7% rule):
            %     99.7% of samples from N(0, sigma^2) fall within [-3*sigma, +3*sigma].
            %     To guarantee virtually no frequency exceeds f_N:
            %
            %       3 * sigma <= f_N   =>   sigma = f_N / 3
            %
            %     Setting sigma above this threshold risks aliasing artifacts.
            %
            %   USAGE:
            %     sigma = OceanStats.Preprocessing.FourierFeatureMap.recommendSigma(X_raw)
            %     sigma = OceanStats.Preprocessing.FourierFeatureMap.recommendSigma(X_raw, ...
            %                 'SafetyFactor', 2)   % 95.4% safe, richer but riskier
            %
            %   INPUTS:
            %     X            [d x N] raw input coordinates (not Fourier-mapped)
            %     SafetyFactor Divisor for f_N. 3 = 99.7% safe (default).
            %                  Lower = richer frequencies, higher aliasing risk.
            %
            %   OUTPUT:
            %     sigma        Recommended bandwidth. Pass as 'Sigma' in fitTransform.
            arguments
                X                    double
                opts.SafetyFactor    (1,1) double  = 3      % 3-sigma bound (99.7% rule)
                opts.Verbose         (1,1) logical = true
                opts.SubsampleN      (1,1) double  = 2000   % Cap for speed
            end

            N  = size(X, 2);
            K  = min(N, opts.SubsampleN);
            Xs = X(:, randperm(N, K));          % [d x K] subsample

            % Nearest-neighbour distances (vectorized, O(K^2) in memory)
            % For large K, process in blocks to avoid OOM
            deltas = inf(1, K);
            block  = 200;
            for start = 1:block:K
                cols = start : min(start+block-1, K);
                D = sqrt(sum((Xs - permute(Xs(:, cols), [1 3 2])).^2, 1)); % [1 x K x B]
                D = squeeze(D);            % [K x B]
                if numel(cols) == 1, D = D'; end
                D(sub2ind(size(D), cols, 1:numel(cols))) = Inf;  % exclude self
                deltas(cols) = min(D, [], 1);
            end

            delta_x = median(deltas); % median nearest-neighbor spacing. Spatial resolution of the grid.
            f_N     = 1 / (2 * delta_x); % Nyquist frequency
            sigma   = f_N / opts.SafetyFactor; % safe sigma

            if opts.Verbose
                fprintf('[FourierFeatureMap.recommendSigma]\n');
                fprintf('  Grid spacing  delta_x = %.5f  (median nearest-neighbour)\n', delta_x);
                fprintf('  Nyquist freq  f_N     = %.4f  cycles/unit\n', f_N);
                fprintf('  Safe sigma    f_N/%d   = %.4f\n', round(opts.SafetyFactor), sigma);
                fprintf('  Aliasing risk above  sigma > %.4f\n\n', f_N / 2);
            end
        end

        function [bestSigma, results] = sweepSigma(X, Y, opts)
            % SWEEPSIGMA  Empirical U-curve sweep to find optimal sigma.
            %
            %   Geometrically scans sigma over [SigmaMin, SigmaMax], trains a
            %   lightweight 1-hidden-layer probe network for each value, and
            %   records validation MSE. The optimal sigma is at the minimum.
            %
            %   Interpretation of the resulting U-curve:
            %     sigma too low  -> high MSE (underfitting: too smooth, low-freq only)
            %     sigma too high -> high MSE (overfitting gaps: aliasing hallucinations)
            %     bottom of U   -> optimal: best fit without aliasing
            %
            %   USAGE:
            %     [sig, res] = OceanStats.Preprocessing.FourierFeatureMap.sweepSigma(X, Y);
            %     [sig, res] = OceanStats.Preprocessing.FourierFeatureMap.sweepSigma(X, Y, ...
            %         'SigmaMin', 0.5, 'SigmaMax', 30, 'NumSteps', 10);
            %
            %   INPUTS:
            %     X, Y         Training inputs [d x N] and targets [1 x N]
            %     SigmaMin     Lower sweep bound (default: 0.1)
            %     SigmaMax     Upper sweep bound (default: 50)
            %     NumSteps     Number of sigma values to test (default: 8)
            %     M            Fourier components for probe net (default: 32)
            %     HiddenDim    Hidden units in probe net (default: 64)
            %     Epochs       Training steps per probe net (default: 200)
            %     ValFrac      Validation fraction (default: 0.2)
            %     PlotResult   Show U-curve plot (default: true)
            %
            %   OUTPUTS:
            %     bestSigma    sigma with lowest validation MSE
            %     results      struct: .Sigmas .ValMSE .TrainMSE .BestSigma
            arguments
                X               double
                Y               double
                opts.SigmaMin   (1,1) double  = 0.1
                opts.SigmaMax   (1,1) double  = 50.0
                opts.NumSteps   (1,1) double  = 8
                opts.M          (1,1) double  = 32
                opts.HiddenDim  (1,1) double  = 64
                opts.Epochs     (1,1) double  = 200
                opts.ValFrac    (1,1) double  = 0.2
                opts.PlotResult (1,1) logical = true
            end

            N      = size(X, 2);
            nVal   = round(N * opts.ValFrac);
            perm   = randperm(N);
            valIdx = perm(1:nVal);
            trIdx  = perm(nVal+1:end);

            % Geometric (log-spaced) sigma sequence
            sigmas   = exp(linspace(log(opts.SigmaMin), log(opts.SigmaMax), opts.NumSteps));
            trainMSE = zeros(1, opts.NumSteps);
            valMSE   = zeros(1, opts.NumSteps);

            fprintf('[FourierFeatureMap.sweepSigma]  Probe: %d->%d->1  Epochs: %d\n', ...
                2*opts.M, opts.HiddenDim, opts.Epochs);
            fprintf('  Sigma range: %.2f to %.2f  (%d steps, geometric)\n\n', ...
                opts.SigmaMin, opts.SigmaMax, opts.NumSteps);

            for si = 1:opts.NumSteps 
                ffm_probe = OceanStats.Preprocessing.FourierFeatureMap(); 
                Xf_all    = ffm_probe.encode(X, 'M', opts.M, 'Sigma', sigmas(si)); %One hidden layer probe net
                Xtr = Xf_all(:, trIdx);  Ytr = Y(trIdx);
                Xvl = Xf_all(:, valIdx); Yvl = Y(valIdx);

                [trainMSE(si), valMSE(si)] = OceanStats.Preprocessing.FourierFeatureMap.probeNet_( ...
                    Xtr, Ytr, Xvl, Yvl, opts.HiddenDim, opts.Epochs);

                fprintf('  [%d/%d] sigma=%-7.3f  TrainMSE=%.5f  ValMSE=%.5f%s\n', ...
                    si, opts.NumSteps, sigmas(si), trainMSE(si), valMSE(si), ...
                    repmat(' <-- best', 1, valMSE(si) == min(valMSE(1:si))));
            end

            [~, bestIdx] = min(valMSE);
            bestSigma = sigmas(bestIdx);
            fprintf('\n  Best sigma = %.4f  (ValMSE = %.6f)\n\n', bestSigma, valMSE(bestIdx));

            results = struct('Sigmas', sigmas, 'ValMSE', valMSE, ...
                             'TrainMSE', trainMSE, 'BestSigma', bestSigma);

            if opts.PlotResult
                OceanStats.Preprocessing.FourierFeatureMap.plotSweep_(results);
            end
        end

    end % static public methods

    % =====================================================================
    methods (Static, Access = private)
    % =====================================================================

        function [trainMSE, valMSE] = probeNet_(Xtr, Ytr, Xval, Yval, H, epochs)
            %Objective: Li = -[ y_i*log(a_i) + (1-y_i)*log(1-a_i)]
            % dL / dZ = dL / dA * dA / dZ
            % Tiny 1-hidden-layer probe NN for sweepSigma.
            % Uses AdamOptimizer — no inline moment bookkeeping needed.
            d  = size(Xtr, 1);
            BATCH = min(512, size(Xtr, 2));
            N     = size(Xtr, 2);

            W1 = randn(H, d) * sqrt(2/d);  b1 = zeros(H, 1);
            W2 = randn(1, H) * sqrt(2/H);  b2 = zeros(1, 1);

            % One AdamOptimizer instance per parameter
            opts_  = OceanStats.Optim.AdamOptimizer.forParams(4, 'LR', 0.005);
            [oW1, ob1, oW2, ob2] = deal(opts_{1}, opts_{2}, opts_{3}, opts_{4});

            for ep = 1:epochs
                idx = randperm(N, BATCH); %Stochastic gradient descent
                Xb = Xtr(:, idx);  Yb = Ytr(idx);

                % Forward
                A1 = tanh(W1*Xb + b1);
                A2 = 1./(1 + exp(-(W2*A1 + b2)));

                % Gradients
                e   = (A2 - Yb) / BATCH;
                dW2 = e * A1';      db2 = sum(e, 2);
                dA1 = (W2' * e) .* (1 - A1.^2);
                dW1 = dA1 * Xb';   db1 = sum(dA1, 2);

                % Adam updates (clean, no manual t / m / v management)
                W1 = oW1.step(W1, dW1);
                b1 = ob1.step(b1, db1);
                W2 = oW2.step(W2, dW2);
                b2 = ob2.step(b2, db2);
            end

            A1v = tanh(W1*Xval + b1);  Pv = 1./(1+exp(-(W2*A1v + b2)));
            valMSE   = mean((Pv - Yval).^2);
            A1t = tanh(W1*Xtr  + b1);  Pt = 1./(1+exp(-(W2*A1t + b2)));
            trainMSE = mean((Pt - Ytr).^2);
        end


        function plotSweep_(results)
            figure('Name', 'FourierFeatureMap — Sigma U-Curve', ...
                'Color', [0.12 0.12 0.16], 'Position', [200 200 700 380]);
            ax = axes('Color', [0.08 0.08 0.12], 'XColor', [0.6 0.6 0.6], ...
                'YColor', [0.6 0.6 0.6], 'XScale', 'log', ...
                'GridColor', [0.3 0.3 0.35], 'GridAlpha', 0.4);
            hold(ax, 'on'); grid(ax, 'on');

            plot(ax, results.Sigmas, results.TrainMSE, '--o', ...
                'Color', [0.4 0.7 1.0], 'LineWidth', 1.5, 'DisplayName', 'Train MSE');
            plot(ax, results.Sigmas, results.ValMSE, '-o', ...
                'Color', [1.0 0.4 0.4], 'LineWidth', 2.5, 'DisplayName', 'Val MSE');
            xline(ax, results.BestSigma, 'Color', [0.3 1.0 0.5], 'LineWidth', 2, ...
                'Label', sprintf('\\sigma*=%.3f', results.BestSigma), ...
                'LabelColor', [0.3 1.0 0.5], 'FontSize', 9);

            legend(ax, 'Location', 'northeast', 'TextColor', [0.8 0.8 0.8], ...
                'Color', [0.1 0.1 0.15]);
            xlabel(ax, '\sigma  (log scale)', 'Color', [0.85 0.85 0.85], 'FontSize', 11);
            ylabel(ax, 'MSE', 'Color', [0.85 0.85 0.85], 'FontSize', 11);
            title(ax, 'FourierFeatureMap — U-Curve \sigma Sweep', ...
                'Color', [1 1 1], 'FontSize', 13, 'FontWeight', 'bold');
        end

    end % static private methods

end
