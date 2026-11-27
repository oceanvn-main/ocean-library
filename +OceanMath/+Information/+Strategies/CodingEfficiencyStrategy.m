classdef CodingEfficiencyStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % CODINGEFFICIENCYSTRATEGY  Compute eta = H(X) / (L_avg * log2(Base)).
    %
    % Coding efficiency is 1.0 for a perfect code (L_avg = H(X)).
    % Values less than 1 indicate unused channel capacity due to rounding.
    % For Huffman codes: eta > H(X) / (H(X) + 1).
    %
    % Usage (via InformationEngine):
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(table) ...
    %                .usingStrategy('CodingEfficiency') ...
    %                .run();
    %
    % See also: OceanMath.Information.Coding.CodeTable
    %           OceanMath.Information.Strategies.IMetricStrategy

    methods
        function result = calculate(~, context)
            % CALCULATE  Return eta in (0, 1].
            result = context.efficiency();
        end

        function tf = isCompatible(~, context)
            tf = isa(context, 'OceanMath.Information.Coding.CodeTable');
        end

        function name = getName(~)
            name = 'Coding Efficiency eta = H(X)/L_avg';
        end
    end
end
