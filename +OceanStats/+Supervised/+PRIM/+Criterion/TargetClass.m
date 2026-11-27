classdef TargetClass < OceanStats.Supervised.PRIM.Criterion.IObjectiveFunction
    % TARGETCLASS Scores a box based on the proportion of instances 
    % belonging to a specific target class. 
    % Standard strategy for classification bumped hunting.
    
    properties
        Name string = "Target Class Proportion"
        TargetValue
    end
    
    methods
        function obj = TargetClass(targetValue)
            arguments
                targetValue = 1 % Default target class
            end
            obj.TargetValue = targetValue;
        end
        
        function score = evaluate(obj, Y_box, ~)
            if isempty(Y_box)
                score = -Inf;
            else
                score = mean(Y_box == obj.TargetValue);
            end
        end
    end
end
