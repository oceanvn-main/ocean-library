classdef (Abstract) IStatisticalTest < handle
    % ISTATISTICALTEST  Abstract interface for statistical hypothesis tests.
    %
    %   Strategy pattern: every hypothesis test implements this interface.
    %   The caller injects whichever test is appropriate at runtime.
    %
    %   CONTRACT:
    %     result = test(obj, data)   — run the test, return TestResult
    %     name(obj)                  — short human-readable name
    %     assumptions(obj)           — list of assumptions this test requires
    %
    %   STANDARD DATA FORMAT:
    %     'data' is a struct whose fields depend on the concrete test.
    %     Each test documents its required fields in its own header.
    %     Common conventions:
    %       data.Yhat1   — predictions from model 1  (N×1)
    %       data.Yhat2   — predictions from model 2  (N×1)
    %       data.Ytrue   — ground truth labels        (N×1)
    %       data.Alpha   — significance level (optional, default 0.05)
    %
    %   USAGE PATTERN:
    %     t      = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
    %     data   = struct('Yhat1', Yhat1, 'Yhat2', Yhat2, 'Ytrue', Ytrue);
    %     result = t.test(data);
    %     if result.Reject
    %       fprintf('Models differ significantly (p=%.4f)\n', result.PValue);
    %     end
    %
    %   See also: OceanStats.Inference.HypothesisTest.Base.TestResult
    %             OceanStats.Inference.HypothesisTest.Paired.McNemarTest

    % ================================================================== %
    methods (Abstract)

        % TEST  Run the hypothesis test on the provided data.
        %
        %   result = test(obj, data)
        %
        %   data   - struct; required fields are test-specific (see subclass)
        %   result - TestResult value object
        result = test(obj, data)

        % NAME  Return the short human-readable name of this test.
        %
        %   str = test.name()   % e.g. "McNemar's Test"
        str = name(obj)

        % ASSUMPTIONS  Return the statistical assumptions this test makes.
        %
        %   list = test.assumptions()   % (1×K) cell array of strings
        %
        %   Example: {'Binary outcomes', 'Matched pairs', 'N >= 25'}
        list = assumptions(obj)

    end

    % ================================================================== %
    methods

        % DESCRIBE  Print a formatted summary of the test and its assumptions.
        function describe(obj)
            fprintf('\n  Test : %s\n', obj.name());
            fprintf('  Assumptions:\n');
            asmp = obj.assumptions();
            for i = 1:numel(asmp)
                fprintf('    [%d] %s\n', i, asmp{i});
            end
            fprintf('\n');
        end

    end

end
