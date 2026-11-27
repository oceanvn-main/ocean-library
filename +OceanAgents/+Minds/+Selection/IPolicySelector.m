classdef (Abstract) IPolicySelector < handle
    % IPOLICYSELECTOR  Chooses which of the mind's policies acts now (design D29). A mind may hold several policies;
    % today the choice comes from settings (FixedSelector), later from a learned selector (for example a neural
    % network that picks a policy from the situation). The selector sees only what the mind sees.
    %
    %   name = selector.select(context)      % context: struct with Perceived, Names, EpisodeID, Tick

    methods (Abstract)
        name = select(obj, context)
        d = describe(obj)
    end
end
