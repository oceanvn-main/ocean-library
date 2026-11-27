classdef (Abstract) IBoxSelector < handle
    % IBOXSELECTOR Abstract strategy for selecting the optimal box from a peeling sequence.
    
    methods (Abstract)
        % SELECT Given a generated sequence of Box objects and dataset, 
        % return the index of the best box.
        % Inputs:
        %   boxesSeq: Cell array of Box objects (the peeling sequence)
        %   X: Features [N x D]
        %   Y: Responses [N x 1]
        %   objectiveFn: An object implementing IObjectiveFunction
        %   primInstance: An optional reference to the parent PRIM tracking growth
        % Output:
        %   bestIdx: Integer index corresponding to the optimal Box in boxesSeq.
        bestIdx = select(obj, boxesSeq, X, Y, objectiveFn, primInstance)
    end
end
