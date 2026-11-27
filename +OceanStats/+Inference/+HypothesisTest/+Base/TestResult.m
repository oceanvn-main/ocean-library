classdef TestResult
    % TESTRESULT  Immutable result container for any statistical hypothesis test.
    %
    %   Returned by IStatisticalTest.test(). Provides a standardised
    %   structure so callers can handle results from different tests
    %   uniformly, without knowing which test was used.
    %
    %   FIELDS:
    %     Statistic  — the computed test statistic value
    %     PValue     — two-sided p-value (or one-sided if documented by test)
    %     Df         — degrees of freedom (NaN if not applicable)
    %     Alpha      — significance level used for Reject decision
    %     Reject     — true if H0 is rejected at Alpha
    %     CI         — confidence interval [lo, hi] (NaN(1,2) if N/A)
    %     TestName   — name of the test that produced this result
    %     Notes      — any warnings, e.g. small-sample correction applied
    %
    %   USAGE:
    %     result = McNemarTest().test(data);
    %     result.Reject       % logical
    %     result.PValue       % scalar
    %     result.summary()    % prints a formatted block
    %
    %   See also: OceanStats.Inference.HypothesisTest.Base.IStatisticalTest

    % ================================================================== %
    properties (SetAccess = immutable)
        Statistic  (1,1) double   = NaN    % Test statistic value
        PValue     (1,1) double   = NaN    % p-value
        Df         (1,1) double   = NaN    % Degrees of freedom (NaN if N/A)
        Alpha      (1,1) double   = 0.05   % Significance level
        Reject     (1,1) logical  = false  % H0 rejected at Alpha?
        CI         (1,2) double   = [NaN NaN]  % Confidence interval
        TestName   (1,1) string   = ""     % Name of the test
        Notes      (1,1) string   = ""     % Warnings or extra info
    end

    % ================================================================== %
    methods

        function obj = TestResult(opts)
            % TESTRESULT  Construct a result object.
            %
            %   r = TestResult('Statistic', chi2, 'PValue', p, ...
            %                  'Df', 1, 'Alpha', 0.05, ...
            %                  'TestName', "McNemar's Test", 'Notes', "")
            %
            % All fields are optional and default to NaN/false/"".
            % Reject is computed automatically from PValue and Alpha if
            % both are provided and Reject is not explicitly supplied.
            arguments
                opts.Statistic (1,1) double  = NaN
                opts.PValue    (1,1) double  = NaN
                opts.Df        (1,1) double  = NaN
                opts.Alpha     (1,1) double  = 0.05
                opts.Reject    (1,1) logical = false
                opts.CI        (1,2) double  = [NaN NaN]
                opts.TestName  (1,1) string  = ""
                opts.Notes     (1,1) string  = ""
            end
            obj.Statistic = opts.Statistic;
            obj.PValue    = opts.PValue;
            obj.Df        = opts.Df;
            obj.Alpha     = opts.Alpha;
            % Auto-derive Reject from PValue vs Alpha if both are known
            if ~isnan(opts.PValue)
                obj.Reject = opts.PValue < opts.Alpha;
            else
                obj.Reject = opts.Reject;
            end
            obj.CI       = opts.CI;
            obj.TestName = opts.TestName;
            obj.Notes    = opts.Notes;
        end

        % -------------------------------------------------------------- %
        function summary(obj)
            % SUMMARY  Print a formatted result block.
            %
            %   result.summary()
            %
            %   Example output:
            %     ╔═══════════════════════════════╗
            %     ║  McNemar's Test               ║
            %     ╠═══════════════════════════════╣
            %     ║  Statistic : 5.0800  (df=1)  ║
            %     ║  p-value   : 0.0242           ║
            %     ║  alpha     : 0.0500           ║
            %     ║  Decision  : REJECT H0  ✓    ║
            %     ║  Notes     : —                ║
            %     ╚═══════════════════════════════╝
            w = 40;
            sep  = repmat('═', 1, w-2);
            fprintf('\n  ╔%s╗\n', sep);
            fprintf('  ║  %-*s║\n', w-3, obj.TestName);
            fprintf('  ╠%s╣\n', sep);
            if ~isnan(obj.Df)
                fprintf('  ║  Statistic : %-6.4f  (df=%g)%*s║\n', ...
                    obj.Statistic, obj.Df, w-32, '');
            else
                fprintf('  ║  Statistic : %-6.4f%*s║\n', ...
                    obj.Statistic, w-22, '');
            end
            fprintf('  ║  p-value   : %-6.4f%*s║\n', obj.PValue, w-22, '');
            fprintf('  ║  alpha     : %-6.4f%*s║\n', obj.Alpha,  w-22, '');
            if obj.Reject
                decision = 'REJECT H0  ✓';
            else
                decision = 'FAIL TO REJECT H0';
            end
            fprintf('  ║  Decision  : %-*s║\n', w-15, decision);
            if strlength(obj.Notes) > 0
                fprintf('  ║  Notes     : %-*s║\n', w-15, obj.Notes);
            end
            fprintf('  ╚%s╝\n\n', sep);
        end

    end

end
