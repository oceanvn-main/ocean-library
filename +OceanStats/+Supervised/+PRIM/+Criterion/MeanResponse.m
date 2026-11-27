classdef MeanResponse < OceanStats.Supervised.PRIM.Criterion.IObjectiveFunction
    % MEANRESPONSE Scores a box based on the mean of the response variable.
    % Standard strategy for regression "bump hunting".
    
    properties
        Name string = "Mean Response"
    end
    
    methods
        function obj = MeanResponse()
        end
        
        function score = evaluate(~, Y_box, ~)
            if isempty(Y_box)
                score = -Inf; % An empty box is minimally desirable
            else
                score = mean(Y_box);
            end
        end
    end
end
