classdef ExpectedCodeLengthStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % EXPECTEDCODELENTHSTRATEGY  Compute L_avg = sum_k p_k * len_k.
    %
    % Returns the average codeword length of a source code, weighted
    % by the symbol probabilities. The context must be a CodeTable.
    %
    % Usage (via InformationEngine):
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(table) ...
    %                .usingStrategy('ExpectedLength') ...
    %                .run();
    %
    % Usage (direct):
    %   s = OceanMath.Information.Strategies.ExpectedCodeLengthStrategy();
    %   L = s.calculate(table);   % scalar double
    %
    % See also: OceanMath.Information.Coding.CodeTable
    %           OceanMath.Information.Strategies.IMetricStrategy

    methods
        function result = calculate(~, context)
            % CALCULATE  Return L_avg = sum p_k * len_k.
            result = context.expectedLength();
        end

        function tf = isCompatible(~, context)
            % ISCOMPATIBLE  True if context is a CodeTable.
            tf = isa(context, 'OceanMath.Information.Coding.CodeTable');
        end

        function name = getName(~)
            name = 'Expected Code Length L_avg';
        end
    end
end
