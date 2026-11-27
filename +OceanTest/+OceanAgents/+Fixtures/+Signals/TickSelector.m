classdef TickSelector < OceanAgents.Minds.Selection.IPolicySelector
    % TICKSELECTOR  Test fixture: a selector that decides from the situation it is given (here the tick: even ticks
    % use the first policy, odd ticks the second), standing in for a learned selector.
    methods
        function name = select(~, context)
            name = context.Names(1 + mod(context.Tick, 2));
        end
        function d = describe(~)
            d = struct('Name', "TickSelector");
        end
    end
end
