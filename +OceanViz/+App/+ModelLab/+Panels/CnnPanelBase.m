classdef (Abstract) CnnPanelBase < OceanViz.App.ModelLab.Panels.PanelBase
    % CNNPANELBASE  Shared layout for the CNN-interpretability panels (the
    %   tabs of the former CNNInterpretabilityStudio): controls on the left,
    %   one plot on the right, a Run button calling the method's
    %   OceanDL.Interpretability.* module and its OceanViz.Strategies.CNN.* renderer.

    properties (SetAccess = protected)
        Ax
    end

    methods
        function c = requires(~), c = "cnn-legacy"; end
    end

    methods (Access = protected)
        function setup(obj, parent, ctx)
            obj.layout(parent, ctx, 1, 1);
            obj.Ax = uiaxes(obj.Main);
            obj.Ax.Color = ctx.Theme.Background; obj.Ax.XColor = ctx.Theme.Text; obj.Ax.YColor = ctx.Theme.Text;
        end

        function img = input(obj)
            % The current image as (H x W x C x 1), forward pass done (caches filled).
            img = obj.Ctx.Model.prepare(obj.Ctx.Image);
        end

        function e = engine(obj)
            e = obj.Ctx.Model.Engine;
        end

        function dd = layerDropdown(obj, labelText, types, default)
            m = obj.Ctx.Model;
            if isempty(types), names = m.layers(); else, names = m.layers(types); end
            if isempty(names), names = {'--'}; end
            if nargin < 4 || isempty(default), default = names{1}; end
            dd = obj.Side.dropdown(labelText, names, default);
        end

        function runButton(obj, callback)
            obj.Side.button('Run', @(~, ~) obj.safely(callback), true);
        end

        function safely(obj, callback)
            try
                callback();
            catch err
                obj.Ctx.status([char(obj.title()) ' failed: ' err.message]);
            end
        end
    end
end
