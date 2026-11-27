classdef CodingGapStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % CODINGGAPSTRATEGY  Compute coding gap = L_avg - H(X) in bits.
    %
    % The coding gap is the number of extra bits per symbol that the code
    % uses beyond the theoretical entropy minimum. For binary Huffman codes,
    % the coding gap is always in [0, 1).
    %
    % Note: For binary codes (Base=2), this equals redundancy() exactly.
    % The distinction matters for k-ary codes where the natural unit differs.
    %
    % Usage (via InformationEngine):
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(table) ...
    %                .usingStrategy('CodingGap') ...
    %                .run();
    %
    % See also: OceanMath.Information.Coding.CodeTable
    %           OceanMath.Information.Strategies.IMetricStrategy

    methods
        function result = calculate(~, context)
            % CALCULATE  Return gap = L_avg - H(X) in bits.
            result = context.codingGap();
        end

        function tf = isCompatible(~, context)
            tf = isa(context, 'OceanMath.Information.Coding.CodeTable');
        end

        function name = getName(~)
            name = 'Coding Gap = L_avg - H(X) [bits]';
        end
    end
end
