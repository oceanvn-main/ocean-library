classdef (Abstract) IObjectiveFunction < handle
    % IOBJECTIVEFUNCTION Strategy interface for scoring a PRIM Box.
    % This abstraction allows PRIM to hunt for regions of high mean (regression)
    % or high target class probability (classification) smoothly.
    
    properties (Abstract)
        Name string
    end
    
    methods (Abstract)
        % EVALUATE Calculates the score for the data within a proposed box.
        % Inputs:
        %   Y_box: Response values for the data currently in the box (N_box x 1)
        %   X_box: Feature values for the data currently in the box (N_box x D)
        % Output:
        %   score: A scalar value indicating the quality of the box 
        %          (higher is better).
        score = evaluate(obj, Y_box, X_box)
    end
end
