classdef RawEncoder < OceanStats.Base.IDataEncoder
    % RAWENCODER Pass-through encoder for regression tasks.
    %
    % encode(Y) returns Y unchanged.
    % decode(scores) returns scores unchanged.
    % isClassification() returns false.
    
    methods
        function Y_enc = encode(~, Y)
            Y_enc = Y;
        end
        
        function y_pred = decode(~, scores)
            y_pred = scores;
        end
        
        function tf = isClassification(~)
            tf = false;
        end
    end
end
