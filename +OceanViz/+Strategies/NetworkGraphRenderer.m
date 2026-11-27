classdef NetworkGraphRenderer < OceanViz.Strategies.IPredictionRenderer
    % NETWORKGRAPHRENDERER  Renders an animated NNGraph (edge weights +
    %   node activations) as the main prediction panel in NNReplayEngine.
    %
    %   Builds a compact proxy graph with VIZ_INPUT representative input nodes
    %   so the 50k+ edge count of a 784-node network never hits the renderer.
    %
    %   PROPERTIES:
    %     VIZ_INPUT - number of proxy input nodes (default 16)
    %
    % See also: OceanViz.Strategies.IPredictionRenderer

    properties
        VIZ_INPUT = 16
    end

    properties (Access = private)
        GraphViz_
        ParamStoreViz_
        Layout_
        Strategy_
        VizInputIdx_
        Colors_
    end

    methods
        function setup(obj, ax, checkpoint, colors)
            obj.Colors_ = colors;
            INPUT_DIM = checkpoint.layerSizes(1);
            HID_DIM   = checkpoint.layerSizes(2);
            OUT_DIM   = checkpoint.layerSizes(end);

            obj.VizInputIdx_ = round(linspace(1, INPUT_DIM, obj.VIZ_INPUT));
            spec = OceanDL.Graph.NNTopologySpec.make( ...
                [obj.VIZ_INPUT, HID_DIM, OUT_DIM], checkpoint.activationFns);
            obj.GraphViz_      = OceanDL.Graph.NNGraph.fromSpec(spec);
            obj.ParamStoreViz_ = OceanDL.Graph.ParameterStore();
            obj.Layout_        = OceanDL.Viz.NNLayeredLayout();
            obj.Strategy_      = OceanViz.Strategies.NetworkGraphStrategy();
            axis(ax, 'off');
        end

        function render(obj, ax, checkpoint, frame, ep)
            ck        = checkpoint;
            INPUT_DIM = ck.layerSizes(1);
            HID_DIM   = ck.layerSizes(2);
            OUT_DIM   = ck.layerSizes(end);

            % Reconstruct weight matrices from flat checkpoint
            v   = ck.weightHistory{ep};
            b1  = v(1:HID_DIM);
            W1  = reshape(v(HID_DIM+1 : HID_DIM + HID_DIM*INPUT_DIM), HID_DIM, INPUT_DIM);
            off = HID_DIM + HID_DIM*INPUT_DIM;
            b2  = v(off+1 : off+OUT_DIM);
            W2  = reshape(v(off+OUT_DIM+1 : end), OUT_DIM, HID_DIM);

            % Inject subsampled weights into the proxy display graph
            W1_viz  = W1(:, obj.VizInputIdx_); %Keeping only a subset of the weights for visualization
            obj.ParamStoreViz_.setFromFlat(obj.GraphViz_, [b1(:); W1_viz(:); b2(:); W2(:)]);

            % Assign pre-computed activations to display graph nodes
            inNodes  = obj.GraphViz_.getLayer(1).getNodes();
            for ni = 1:obj.VIZ_INPUT
                inNodes{ni}.Value_ = frame.X(obj.VizInputIdx_(ni));
            end
            hidNodes = obj.GraphViz_.getLayer(2).getNodes();
            for ni = 1:HID_DIM, hidNodes{ni}.Value_ = frame.A1(ni); end
            outNodes = obj.GraphViz_.getLayer(3).getNodes();
            for ni = 1:OUT_DIM, outNodes{ni}.Value_ = frame.A2(ni); end

            % Render
            data = obj.Layout_.toData(obj.GraphViz_, struct('Colors', obj.Colors_));
            cla(ax);
            obj.Strategy_.plot(ax, data, struct('Colors', obj.Colors_, 'NodeDisplay', 'Both'));
            title(ax, sprintf('Epoch %d / %d   |   Accuracy: %.1f%%   |   [%d proxy inputs]', ...
                ep, ck.numEpochs, ck.accHistory(ep)*100, obj.VIZ_INPUT), ...
                'Color', obj.Colors_.Text, 'FontSize', 10);
        end
    end
end
