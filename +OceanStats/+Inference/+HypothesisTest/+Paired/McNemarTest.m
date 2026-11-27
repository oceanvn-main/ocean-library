classdef McNemarTest < OceanStats.Inference.HypothesisTest.Base.IStatisticalTest
    % MCNEMARTEST  McNemar's test for comparing two classifiers on paired data.
    %
    %   Tests whether two classifiers have significantly different error rates
    %   when evaluated on the same held-out test set (matched-pair design).
    %
    %   NULL HYPOTHESIS (H0):
    %     The two classifiers make errors in the same proportion.
    %     Formally: P(Model1 wrong, Model2 right) = P(Model1 right, Model2 wrong)
    %     i.e. n01 = n10  (the off-diagonal cells of the contingency table)
    %
    %   TEST STATISTIC:
    %     Standard (asymptotic):    chi2 = (n01 - n10)^2 / (n01 + n10)
    %     With continuity correction: chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
    %
    %     chi2 ~ chi-squared(df=1) under H0
    %
    %   SMALL SAMPLE:
    %     When n01 + n10 < 25, the chi-squared approximation is unreliable.
    %     The test automatically switches to the exact binomial test and
    %     sets a warning in TestResult.Notes.
    %
    %   INPUT (two formats accepted):
    %
    %   Format 1 — raw predictions:
    %     data.Yhat1    (N×1) logical/binary — predictions from model 1
    %     data.Yhat2    (N×1) logical/binary — predictions from model 2
    %     data.Ytrue    (N×1) logical/binary — ground truth labels
    %     data.Alpha    (1×1) double         — significance level (default 0.05)
    %     data.Correction (1×1) logical      — use continuity correction (default true)
    %
    %   Format 2 — pre-computed contingency counts:
    %     data.n01      scalar — model1 wrong, model2 correct
    %     data.n10      scalar — model1 correct, model2 wrong
    %     data.Alpha    (1×1) double         — significance level (default 0.05)
    %     data.Correction (1×1) logical      — use continuity correction (default true)
    %
    %   USAGE:
    %     t = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
    %
    %     % Format 1 — from raw predictions
    %     data   = struct('Yhat1', Yhat1, 'Yhat2', Yhat2, 'Ytrue', Ytrue);
    %     result = t.test(data);
    %     result.summary();
    %
    %     % Format 2 — from contingency counts
    %     result = t.test(struct('n01', 12, 'n10', 3));
    %     result.Reject   % true/false
    %     result.PValue   % p-value
    %
    %   REFERENCE:
    %     McNemar, Q. (1947). Note on the sampling error of the difference
    %     between correlated proportions or percentages.
    %     Psychometrika, 12(2), 153–157.
    %
    %     Dietterich, T. G. (1998). Approximate statistical tests for
    %     comparing supervised classification learning algorithms.
    %     Neural Computation, 10(7), 1895–1923.
    %
    %   See also: OceanStats.Inference.HypothesisTest.Base.IStatisticalTest
    %             OceanStats.Inference.HypothesisTest.Base.TestResult

    % ================================================================== %
    methods

        function result = test(obj, data)
            % TEST  Run McNemar's test on paired classifier predictions.
            %
            %   result = t.test(data)
            %
            %   See class header for data format options.

            % ---- Parse inputs ---------------------------------------- %
            alpha      = obj.getField_(data, 'Alpha',      0.05);
            correction = obj.getField_(data, 'Correction', true);

            % Build contingency counts
            if isfield(data, 'n01') && isfield(data, 'n10')
                n01 = data.n01;
                n10 = data.n10;
            elseif isfield(data, 'Yhat1') && isfield(data, 'Yhat2') && isfield(data, 'Ytrue')
                [n01, n10] = obj.buildCounts_(data.Yhat1, data.Yhat2, data.Ytrue);
            else
                error('OceanStats:Inference:McNemarTest:InvalidInput', ...
                    ['McNemarTest.test() requires either:\n' ...
                     '  (a) data.Yhat1, data.Yhat2, data.Ytrue  — raw predictions\n' ...
                     '  (b) data.n01, data.n10                  — contingency counts']);
            end

            % ---- Validate counts ------------------------------------- %
            obj.validateCounts_(n01, n10);

            nDisagree = n01 + n10;
            notes     = "";

            % ---- Compute test statistic ----------------------------- %
            if nDisagree < 25
                % Small sample: exact binomial test
                [chi2val, pval] = obj.exactBinomial_(n01, nDisagree);
                df    = NaN;
                notes = "Small sample (n01+n10 < 25): exact binomial test used.";
            else
                % Asymptotic chi-squared test
                if correction
                    chi2val = (abs(n01 - n10) - 1)^2 / nDisagree;
                    notes   = "Continuity correction applied.";
                else
                    chi2val = (n01 - n10)^2 / nDisagree;
                end
                pval = 1 - chi2cdf(chi2val, 1); %Upper tail
                df   = 1;
            end

            % ---- Pack result (full qualified path — MATLAB disallows package aliases) %
            result = OceanStats.Inference.HypothesisTest.Base.TestResult( ...
                'Statistic', chi2val, ...
                'PValue',    pval,    ...
                'Df',        df,      ...
                'Alpha',     alpha,   ...
                'TestName',  obj.name(), ...
                'Notes',     notes);
        end

        % -------------------------------------------------------------- %
        function str = name(~)
            str = "McNemar's Test";
        end

        % -------------------------------------------------------------- %
        function list = assumptions(~)
            list = { ...
                'Binary (dichotomous) outcomes — each observation is right or wrong', ...
                'Matched pairs — both classifiers evaluated on the same N samples', ...
                'Independence of pairs — each sample is independent', ...
                'n01 + n10 >= 25 for chi-squared approximation (auto-switches to exact otherwise)', ...
            };
        end

    end

    % ================================================================== %
    methods (Access = private)

        function [n01, n10] = buildCounts_(~, Yhat1, Yhat2, Ytrue)
            % BUILDCOUNTS_  Compute contingency off-diagonal counts.
            %
            %   n01 = model1 WRONG, model2 RIGHT
            %   n10 = model1 RIGHT, model2 WRONG
            Yhat1 = logical(Yhat1(:));
            Yhat2 = logical(Yhat2(:));
            Ytrue = logical(Ytrue(:));

            if numel(Yhat1) ~= numel(Ytrue) || numel(Yhat2) ~= numel(Ytrue)
                error('OceanStats:Inference:McNemarTest:DimMismatch', ...
                    'Yhat1, Yhat2, and Ytrue must all have the same length N.');
            end

            correct1 = (Yhat1 == Ytrue);
            correct2 = (Yhat2 == Ytrue);

            n01 = sum(~correct1 &  correct2);   % model1 wrong, model2 right
            n10 = sum( correct1 & ~correct2);   % model1 right, model2 wrong
        end

        % -------------------------------------------------------------- %
        function validateCounts_(~, n01, n10)
            % VALIDATECOUNTS_  Throw if counts are non-integer or negative.
            if n01 < 0 || n10 < 0
                error('OceanStats:Inference:McNemarTest:NegativeCounts', ...
                    'n01 and n10 must be non-negative integers. Got n01=%g, n10=%g.', n01, n10);
            end
            if n01 + n10 == 0
                error('OceanStats:Inference:McNemarTest:NoDisagreements', ...
                    'Both classifiers agree on every sample (n01=0, n10=0). Test is undefined.');
            end
        end

        % -------------------------------------------------------------- %
        function [statistic, pval] = exactBinomial_(~, n01, nDisagree)
            % EXACTBINOMIAL_  Two-sided exact binomial test under H0: p=0.5.
            %
            %   Returns a statistic compatible with chi-squared scale
            %   (for consistent display) plus the exact p-value.
            k   = min(n01, nDisagree - n01);   % the smaller tail count
            % Exact two-sided p-value: sum both tails of Bin(nDisagree, 0.5)
            pval = 2 * binocdf(k, nDisagree, 0.5);
            pval = min(pval, 1.0);  % clamp numerical overshoot
            % Report a chi-squared-like statistic for display consistency
            statistic = (n01 - nDisagree * 0.5)^2 / (nDisagree * 0.25);
        end

        % -------------------------------------------------------------- %
        function val = getField_(~, s, field, default)
            % GETFIELD_  Return struct field or default if missing.
            if isfield(s, field)
                val = s.(field);
            else
                val = default;
            end
        end

    end

end
