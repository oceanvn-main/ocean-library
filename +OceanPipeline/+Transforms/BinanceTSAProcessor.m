classdef BinanceTSAProcessor < handle
    % BINANCETSAPROCESSOR  Time Series Analysis (TSA) Preprocessing and EDA Engine.
    %
    %   Performs end-to-end data cleaning, causal regularisation, feature
    %   engineering, exploratory data analysis (EDA), stationarity testing,
    %   and strict anti-leakage temporal train/val/test splitting for time series models.
    %
    %   Workflow:
    %     1. Causal Regularisation: retime() to fixed interval & forward-fill missing values
    %     2. Feature Engineering: Log returns, rolling volatility, volume z-score, taker ratio, VWAP
    %     3. Exploratory Data Analysis: ADF stationarity test, ACF/PACF, spectral density, normality
    %     4. Anti-Leakage Temporal Split: Chronological Train/Val/Test split & train-only scaling
    %     5. Medallion Export: Bronze (raw) and Silver (engineered) Parquet data lake persistence
    %
    %   Syntax:
    %     processor = OceanPipeline.Transforms.BinanceTSAProcessor();
    %     TT_reg = processor.regularizeTimetable(TT_raw, '1h');
    %     TT_feat = processor.engineerFeatures(TT_reg, 'VolWindow', 24);
    %     edaResults = processor.runEDA(TT_feat, 'Close', 'maxLag', 40);
    %     splits = processor.splitTimeSeries(TT_feat, 'TrainRatio', 0.7, 'ValRatio', 0.15);
    %
    %   See also: OceanPipeline.Ingestion.BinanceDataReader, timetable, adftest, autocorr, parcorr

    properties
        TimeStep    (1,1) string = "1h"
        VolWindow   (1,1) double = 24
        MaxLag      (1,1) double = 40
        TrainRatio  (1,1) double = 0.70
        ValRatio    (1,1) double = 0.15
        TestRatio   (1,1) double = 0.15
    end

    methods
        function obj = BinanceTSAProcessor(options)
            % BINANCETSAPROCESSOR  Constructor.
            arguments
                options.TimeStep   (1,1) string = "1h"
                options.VolWindow  (1,1) double = 24
                options.MaxLag     (1,1) double = 40
                options.TrainRatio (1,1) double = 0.70
                options.ValRatio   (1,1) double = 0.15
                options.TestRatio  (1,1) double = 0.15
            end

            obj.TimeStep   = options.TimeStep;
            obj.VolWindow  = options.VolWindow;
            obj.MaxLag     = options.MaxLag;
            obj.TrainRatio = options.TrainRatio;
            obj.ValRatio   = options.ValRatio;
            obj.TestRatio  = 1.0 - (options.TrainRatio + options.ValRatio);
        end

        function TT_reg = regularizeTimetable(obj, TT, timeStep)
            % REGULARIZETIMETABLE  Enforce regular grid & causal missing value imputation.
            arguments
                obj
                TT timetable
                timeStep (1,1) string = ""
            end

            if timeStep == ""
                timeStep = obj.TimeStep;
            end

            dur = obj.parseTimeStep(timeStep);

            % Retime to regular grid using linear interpolation for prices
            TT_reg = retime(TT, 'regular', 'linear', 'TimeStep', dur);

            % Apply causal forward-fill ('previous') to prevent looking into the future
            varNames = TT_reg.Properties.VariableNames;
            for i = 1:length(varNames)
                col = varNames{i};
                if any(isnan(TT_reg.(col))) || any(ismissing(TT_reg.(col)))
                    TT_reg.(col) = fillmissing(TT_reg.(col), 'previous');
                    % If leading elements are NaN, fill with next valid value
                    TT_reg.(col) = fillmissing(TT_reg.(col), 'next');
                end
            end

            TT_reg = sortrows(TT_reg, 'Time');
        end

        function TT_feat = engineerFeatures(obj, TT, options)
            % ENGINEERFEATURES  Compute log returns, realized volatility, VWAP, and taker ratio.
            arguments
                obj
                TT timetable
                options.VolWindow (1,1) double = 0
            end

            volWin = obj.VolWindow;
            if options.VolWindow > 0
                volWin = options.VolWindow;
            end

            TT_feat = TT;
            P = TT_feat.Close;

            % 1. Log Returns: r_t = ln(P_t / P_{t-1})
            logRet = [0; diff(log(P))];
            TT_feat.LogReturn = logRet;

            % 2. Price Difference: Delta P_t = P_t - P_{t-1}
            TT_feat.PriceDiff = [0; diff(P)];

            % 3. Log Price
            TT_feat.LogPrice = log(P);

            % 4. Rolling Volatility (Rolling standard deviation of log returns)
            % Econometric note: Econometrically, "Realized Volatility" specifically refers to RV = sum(r_i^2) 
            % constructed from high-frequency intraday ticks. This metric is a rolling standard deviation.
            rollingVol = movstd(logRet, [volWin-1, 0], 'omitnan');
            TT_feat.RollingVol  = rollingVol;
            TT_feat.RealizedVol = rollingVol; % Alias for backward compatibility

            % 5. Rolling Volume Z-Score
            vol = TT_feat.Volume;
            movMuVol  = movmean(vol, [volWin-1, 0], 'omitnan');
            movSigVol = movstd(vol,  [volWin-1, 0], 'omitnan');
            TT_feat.VolumeZScore = (vol - movMuVol) ./ (movSigVol + 1e-8);

            % 6. Taker Buy Ratio (if TakerBuyBaseVolume is present)
            if ismember('TakerBuyBaseVolume', TT_feat.Properties.VariableNames)
                takerRatio = TT_feat.TakerBuyBaseVolume ./ (TT_feat.Volume + 1e-8);
                takerRatio(takerRatio > 1) = 1;
                TT_feat.TakerBuyRatio = takerRatio;
            end

            % 7. Rolling Volume-Weighted Average Price (VWAP)
            % Econometric distinction:
            %  - Exact Exchange VWAP = sum_i (trade_price_i * trade_qty_i) / sum_i (trade_qty_i). 
            %    Binance provides sum_i (price * qty) as QuoteAssetVolume for each candle.
            %  - Typical Price VWAP = sum(TP_t * V_t) / sum(V_t) where TP_t = (High_t + Low_t + Close_t)/3.
            %  - Close-Weighted VWAP = sum(Close_t * V_t) / sum(V_t).
            denVwap = movsum(vol, [volWin-1, 0], 'omitnan') + 1e-8;
            
            % Typical Price (H + L + C) / 3
            if ismember('High', TT_feat.Properties.VariableNames) && ismember('Low', TT_feat.Properties.VariableNames)
                tp = (TT_feat.High + TT_feat.Low + TT_feat.Close) / 3;
            else
                tp = P;
            end
            TT_feat.TypicalVWAP = movsum(tp .* vol, [volWin-1, 0], 'omitnan') ./ denVwap;
            TT_feat.CloseVWAP   = movsum(P  .* vol, [volWin-1, 0], 'omitnan') ./ denVwap;

            % Set primary VWAP: Exact Exchange VWAP if QuoteAssetVolume exists, else TypicalVWAP
            if ismember('QuoteAssetVolume', TT_feat.Properties.VariableNames)
                TT_feat.VWAP = movsum(TT_feat.QuoteAssetVolume, [volWin-1, 0], 'omitnan') ./ denVwap;
            else
                TT_feat.VWAP = TT_feat.TypicalVWAP;
            end

            % Trim first row where return diff is zero-padded
            TT_feat = TT_feat(2:end, :);
        end

        function eda = runEDA(obj, TT, varName, options)
            % RUNEDA  Exploratory Data Analysis, stationarity tests (ADF), ACF/PACF, & spectrum.
            arguments
                obj
                TT timetable
                varName (1,1) string = "Close"
                options.MaxLag (1,1) double = 0
            end

            maxLag = obj.MaxLag;
            if options.MaxLag > 0
                maxLag = options.MaxLag;
            end

            y = TT.(varName);
            y = y(~isnan(y));
            N = length(y);

            % 1. Summary Statistics
            eda.VarName  = varName;
            eda.N        = N;
            eda.Mean     = mean(y);
            eda.Std      = std(y);
            eda.Variance = var(y);
            eda.Min      = min(y);
            eda.Max      = max(y);
            eda.Skewness = skewness(y);
            eda.Kurtosis = kurtosis(y);

            % Jarque-Bera Test for Normality
            % JB = (N/6) * (S^2 + (1/4)*(K-3)^2)
            S = eda.Skewness;
            K = eda.Kurtosis;
            eda.JarqueBeraStat = (N / 6) * (S^2 + 0.25 * (K - 3)^2);
            eda.JarqueBeraPVal = 1 - chi2cdf(eda.JarqueBeraStat, 2);

            % 2. Augmented Dickey-Fuller (ADF) Stationarity Test
            try
                [hADF, pValADF, statADF] = adftest(y);
                eda.ADF_IsStationary   = logical(hADF);
                eda.ADF_PValue        = pValADF;
                if isstruct(statADF) && isfield(statADF, 'stat')
                    eda.ADF_TestStatistic = statADF.stat;
                else
                    eda.ADF_TestStatistic = statADF;
                end
            catch
                % Fallback if adftest is unavailable or fails
                eda.ADF_IsStationary   = false;
                eda.ADF_PValue        = NaN;
                eda.ADF_TestStatistic = NaN;
            end

            % 3. Autocorrelation (ACF) & Partial Autocorrelation (PACF)
            lagLimit = min(maxLag, N - 2);
            try
                [acfVec, acfLags] = autocorr(y, 'NumLags', lagLimit);
                [pacfVec, pacfLags] = parcorr(y, 'NumLags', lagLimit);
                if ~isempty(pacfLags) && pacfLags(1) == 0
                    pacfVec = pacfVec(2:end);
                    pacfLags = pacfLags(2:end);
                end
            catch
                % Manual fallback computation
                acfVec = obj.computeACF(y, lagLimit);
                acfLags = (0:lagLimit)';
                pacfVec = obj.computePACF(y, lagLimit);
                pacfLags = (1:lagLimit)';
            end

            eda.ACF      = acfVec';
            eda.ACFLags  = acfLags';
            eda.PACF     = pacfVec';
            eda.PACFLags = pacfLags';
            eda.ConfBound = 1.96 / sqrt(N);

            % 4. Spectral Power Density Estimation (Welch periodogram & Raw Periodogram)
            y_centered = y - mean(y);
            
            % 4a. Raw FFT Periodogram (Parseval exact density)
            N_fft = length(y_centered);
            Y_fft = fft(y_centered);
            n_half = floor(N_fft / 2) + 1;
            raw_power = (abs(Y_fft(1:n_half)).^2) / (N_fft * 2 * pi);
            
            if n_half > 1
                if mod(N_fft, 2) == 0
                    raw_power(2:end-1) = 2 * raw_power(2:end-1); % Even N: Nyquist bin at end is unique
                else
                    raw_power(2:end)   = 2 * raw_power(2:end);   % Odd N: No exact Nyquist bin, double all positive freqs
                end
            end
            
            dw_raw = 2 * pi / N_fft;
            w_raw  = (0:n_half-1)' * dw_raw;
            
            eda.Periodogram_Power = raw_power';
            eda.Periodogram_Freqs = w_raw';
            
            % 4b. Welch PSD estimate
            try
                winLen = min(256, N);
                [psdVal, freqs] = pwelch(y_centered, winLen, [], [], 1);
                eda.PSD_Freqs = freqs'; % Normalized freq [0, 0.5] Hz or cycles/sample
                eda.PSD_Power = psdVal';
            catch
                eda.PSD_Freqs = (w_raw / (2 * pi))';
                eda.PSD_Power = raw_power';
            end
            
            % Convert frequencies to normalized angular frequency w in [0, pi] and Period (Hours)
            eda.PSD_AngularFreq = eda.PSD_Freqs * 2 * pi;
            % Period in samples (hours for 1h data)
            validIdx = eda.PSD_AngularFreq > 0;
            eda.PSD_PeriodHours = zeros(size(eda.PSD_AngularFreq));
            eda.PSD_PeriodHours(validIdx) = (2 * pi) ./ eda.PSD_AngularFreq(validIdx);
            
            % Parseval's Theorem Verification: Integral over [0, pi] of one-sided spectral density = var(y, 1)
            eda.Parseval_TotalArea = sum(raw_power) * dw_raw;
            eda.Parseval_Variance  = var(y_centered, 1); % Population variance (N normalization)
            
            % Dominant Peak Identification
            [maxPwr, maxIdx] = max(eda.PSD_Power(validIdx));
            validPeriods = eda.PSD_PeriodHours(validIdx);
            if ~isempty(validPeriods)
                eda.DominantPeriodHours = validPeriods(maxIdx);
                eda.DominantPower       = maxPwr;
            else
                eda.DominantPeriodHours = NaN;
                eda.DominantPower       = NaN;
            end
        end

        function splits = splitTimeSeries(obj, TT, options)
            % SPLITTIMESERIES  Chronological anti-leakage train/val/test split.
            arguments
                obj
                TT timetable
                options.TrainRatio (1,1) double = 0
                options.ValRatio   (1,1) double = 0
            end

            rTr = obj.TrainRatio; if options.TrainRatio > 0, rTr = options.TrainRatio; end
            rVal = obj.ValRatio;   if options.ValRatio > 0,   rVal = options.ValRatio;   end
            rTe = 1.0 - (rTr + rVal);

            nTotal = height(TT);
            nTrain = floor(nTotal * rTr);
            nVal   = floor(nTotal * rVal);

            idxTrain = 1:nTrain;
            idxVal   = (nTrain + 1):(nTrain + nVal);
            idxTest  = (nTrain + nVal + 1):nTotal;

            TT_train = TT(idxTrain, :);
            TT_val   = TT(idxVal, :);
            TT_test  = TT(idxTest, :);

            % Fit normalization scaling strictly on Train set
            numericVars = {};
            varNames = TT.Properties.VariableNames;
            for i = 1:length(varNames)
                if isnumeric(TT.(varNames{i}))
                    numericVars{end+1} = varNames{i}; %#ok<AGROW>
                end
            end

            muStats = struct();
            sigStats = struct();

            TT_train_scaled = TT_train;
            TT_val_scaled   = TT_val;
            TT_test_scaled  = TT_test;

            for i = 1:length(numericVars)
                vName = numericVars{i};
                trVec = TT_train.(vName);
                mu = mean(trVec, 'omitnan');
                sig = std(trVec, 'omitnan');
                if sig == 0 || isnan(sig), sig = 1; end

                muStats.(vName)  = mu;
                sigStats.(vName) = sig;

                TT_train_scaled.(vName) = (TT_train.(vName) - mu) ./ sig;
                TT_val_scaled.(vName)   = (TT_val.(vName)   - mu) ./ sig;
                TT_test_scaled.(vName)  = (TT_test.(vName)  - mu) ./ sig;
            end

            splits.TT_Total      = TT;
            splits.TT_Train      = TT_train;
            splits.TT_Val        = TT_val;
            splits.TT_Test       = TT_test;
            splits.TT_TrainScale = TT_train_scaled;
            splits.TT_ValScale   = TT_val_scaled;
            splits.TT_TestScale  = TT_test_scaled;
            splits.TrainMu       = muStats;
            splits.TrainSigma    = sigStats;
            splits.Indices       = struct('Train', idxTrain, 'Val', idxVal, 'Test', idxTest);
        end

        function lakeResult = exportToLake(obj, splits, datasetName, rootDir)
            % EXPORTTOLAKE  Export datasets into Medallion DataLake as Parquet files.
            arguments
                obj %#ok<INUSD>
                splits struct
                datasetName (1,1) string = "binance_btc"
                rootDir (1,1) string = "./lake/"
            end

            try
                lake = OceanData.DataArch.DataLake('rootDir', rootDir);

                % Bronze (Raw/Total)
                pathRaw = sprintf("raw/%s/total", datasetName);
                lake.put(pathRaw, timetable2table(splits.TT_Total), struct('Dataset', datasetName), 'asParquet', true);

                % Silver (Processed Train, Val, Test)
                lake.put(sprintf("processed/%s/train", datasetName), timetable2table(splits.TT_Train), struct('Split', 'Train'), 'asParquet', true);
                lake.put(sprintf("processed/%s/val",   datasetName), timetable2table(splits.TT_Val),   struct('Split', 'Val'),   'asParquet', true);
                lake.put(sprintf("processed/%s/test",  datasetName), timetable2table(splits.TT_Test),  struct('Split', 'Test'),  'asParquet', true);

                lakeResult.Status = "Success";
                lakeResult.Path = rootDir;
            catch ME
                lakeResult.Status = "Skipped (" + ME.message + ")";
                lakeResult.Path = rootDir;
            end
        end
    end

    methods (Static)
        function r = computeACF(x, maxLag)
            x = x - mean(x);
            varX = sum(x.^2);
            r = zeros(maxLag + 1, 1);
            r(1) = 1;
            for k = 1:maxLag
                r(k+1) = sum(x(1:end-k) .* x(k+1:end)) / varX;
            end
        end

        function phi = computePACF(x, maxLag)
            r = OceanPipeline.Transforms.BinanceTSAProcessor.computeACF(x, maxLag);
            phi = zeros(maxLag, 1);
            for k = 1:maxLag
                R = toeplitz(r(1:k));
                rhs = r(2:k+1);
                sol = R \ rhs;
                phi(k) = sol(end);
            end
        end
    end

    methods (Access = private)

        function dur = parseTimeStep(~, str)
            strChar = char(str);
            numMatch = regexp(strChar, '\d+', 'match', 'once');
            if isempty(numMatch), numVal = 1; else, numVal = str2double(numMatch); end
            unitMatch = regexp(strChar, '[a-zA-Z]+', 'match', 'once');
            switch lower(unitMatch)
                case 'm', dur = minutes(numVal);
                case 'h', dur = hours(numVal);
                case 'd', dur = days(numVal);
                case 's', dur = seconds(numVal);
                otherwise, dur = hours(numVal);
            end
        end
    end
end
