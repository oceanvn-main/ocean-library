classdef VisionObservation < OceanAgents.Body.Sensors.IObservation
    % VISIONOBSERVATION  Reports an image of the scene (an "array" of the renderer's Size, e.g. H x W x 1), made by a
    % renderer from the view. The renderer is the only part that knows how a scene looks. AB2 section 4; learning
    % from images is AB3.
    %
    %   s = OceanAgents.Body.Sensors.VisionObservation(OceanAgents.Body.Sensors.GridRenderer(4, 12));

    properties (SetAccess = private)
        Renderer
    end

    methods
        function obj = VisionObservation(renderer, options)
            arguments
                renderer (1,1) OceanAgents.Body.Sensors.IRenderer
                options.ID (1,1) string = "vision"
            end
            obj.init_(options.ID, "vision", ...
                OceanAgents.Minds.Signals.SignalType('ValueKind', "array", 'Size', renderer.Size, 'Role', "sense"));
            obj.Renderer = renderer;
        end
    end

    methods (Access = protected)
        function r = read_(obj, view, ~)
            r = struct('Image', obj.Renderer.render(view));
        end

        function out = act(obj, signal)
            out = obj.emit_(signal.Value.Image);
        end
    end
end
