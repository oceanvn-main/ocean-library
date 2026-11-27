classdef RedundancyStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % REDUNDANCYSTRATEGY  Compute R = L_avg - H(X)/log2(Base).
    %
    % Measures the gap between actual average codeword length and the
    % theoretical entropy lower bound. Shannon guarantees R >= 0.
    % Huffman achieves R < 1 bit.
    %
    % Usage (via InformationEngine):
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(table) ...
    %                .usingStrategy('Redundancy') ...
    %                .run();
    %
    % See also: OceanMath.Information.Coding.CodeTable
    %           OceanMath.Information.Strategies.IMetricStrategy

    methods
        function result = calculate(~, context)
            % CALCULATE  Return R = L_avg - H(X)/log2(Base).
            result = context.redundancy();
        end

        function tf = isCompatible(~, context)
            tf = isa(context, 'OceanMath.Information.Coding.CodeTable');
        end

        function name = getName(~)
            name = 'Code Redundancy R = L_avg - H(X)';
        end
    end
end
