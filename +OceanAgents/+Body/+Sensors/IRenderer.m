classdef (Abstract) IRenderer
    % IRENDERER  Draws what a vision sensor sees: an image of exactly Size from the view (AB2 section 4). The renderer
    % is the only part that knows how a scene looks; a scene it cannot draw is refused (OceanAgents:Signals:BadValue).

    properties (Abstract, SetAccess = immutable)
        Size double
    end

    methods (Abstract)
        img = render(obj, view)
    end
end
