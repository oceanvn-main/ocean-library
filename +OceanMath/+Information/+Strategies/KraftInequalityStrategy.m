classdef KraftInequalityStrategy < OceanMath.Information.Strategies.IMetricStrategy
    % KRAFTINEQUALITYSTRATEGY  Compute K = sum_k Base^(-len_k).
    %
    % The Kraft inequality states that a uniquely decodable code must
    % satisfy K <= 1. A complete prefix-free code achieves K = 1 exactly.
    % Huffman codes always produce K = 1.
    %
    % Usage (via InformationEngine):
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(table) ...
    %                .usingStrategy('KraftSum') ...
    %                .run();
    %
    % See also: OceanMath.Information.Coding.CodeTable
    %           OceanMath.Information.Strategies.IMetricStrategy

    methods
        function result = calculate(~, context)
            % CALCULATE  Return K = sum Base^(-len_k).
            result = context.kraftSum();
        end

        function tf = isCompatible(~, context)
            tf = isa(context, 'OceanMath.Information.Coding.CodeTable');
        end

        function name = getName(~)
            name = 'Kraft Sum K = sum(Base^(-len_k))';
        end
    end
end
