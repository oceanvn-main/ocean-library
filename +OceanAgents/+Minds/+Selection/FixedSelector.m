classdef FixedSelector < OceanAgents.Minds.Selection.IPolicySelector
    % FIXEDSELECTOR  The policy chosen by settings: always the named one (design D29, "for now we use settings").
    %
    %   mind.useSelector(OceanAgents.Minds.Selection.FixedSelector("explore"));   % or mind.usePolicy("explore")

    properties (SetAccess = private)
        Choice string = ""
    end

    methods
        function obj = FixedSelector(choice)
            arguments
                choice (1,1) string {mustBeNonzeroLengthText}
            end
            obj.Choice = choice;
        end

        function name = select(obj, ~)
            name = obj.Choice;
        end

        function d = describe(obj)
            d = struct('Name', "FixedSelector", 'Choice', obj.Choice);
        end
    end
end
