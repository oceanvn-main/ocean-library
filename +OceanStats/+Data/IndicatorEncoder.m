classdef IndicatorEncoder < OceanStats.Base.IDataEncoder
    % INDICATORENCODER Encodes class labels as an N×K indicator matrix.
    %
    % Used for classification tasks. Stores ClassNames after encode() so
    % decode() can map argmax scores back to original labels.
    %
    % Supports numeric, string, char, logical, and categorical Y vectors.
    
    properties (SetAccess = private, GetAccess = public)
        ClassNames  % Ordered unique class labels discovered during encode()
    end
    
    methods
        function Y_enc = encode(obj, Y)
            % ENCODE Build N×K indicator matrix and store ClassNames.
            obj.ClassNames = unique(Y);
            K     = length(obj.ClassNames);
            N     = size(Y, 1);
            Y_enc = zeros(N, K);
            for k = 1:K
                if isnumeric(Y) || islogical(Y)
                    Y_enc(:, k) = double(Y == obj.ClassNames(k));
                else
                    Y_enc(:, k) = double(strcmp(string(Y), string(obj.ClassNames(k))));
                end
            end
        end
        
        function y_pred = decode(obj, scores)
            % DECODE Map (N×K) score matrix → (N×1) class labels via argmax.
            [~, maxIdx] = max(scores, [], 2);
            y_pred = obj.ClassNames(maxIdx);
            if isrow(y_pred) && size(scores, 1) > 1
                y_pred = y_pred';
            end
        end
        
        function tf = isClassification(~)
            tf = true;
        end
        
        function tf = isFitted(obj)
            % Returns true if encode() has been called.
            tf = ~isempty(obj.ClassNames);
        end
    end
    
    methods (Static)
        function tf = canEncode(Y)
            % CANENCODE Heuristic to detect if Y looks like class labels.
            if size(Y, 2) ~= 1; tf = false; return; end
            if iscategorical(Y) || islogical(Y) || ischar(Y) || isstring(Y)
                tf = true; return;
            end
            uniqueY = unique(Y);
            tf = isnumeric(Y) && length(uniqueY) <= 15 && all(floor(uniqueY) == uniqueY);
        end
    end
end
