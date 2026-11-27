classdef DecisionBoundaryRenderer < OceanViz.Strategies.IPredictionRenderer
    % DECISIONBOUNDARYRENDERER  Renders the NN's learned approximation of a
    %   2D function into the prediction panel of NNReplayEngine.
    %
    %   Evaluates the current epoch's weights on a pre-baked dense grid
    %   (from the checkpoint) and renders the output as a heatmap.
    %   Supports variable-depth networks by reading layerSizes from the
    %   checkpoint and reconstructing weights generically.
    %
    %   PROPERTIES:
    %     Colormap - MATLAB colormap name (default 'hot')
    %
    % See also: OceanViz.Strategies.IPredictionRenderer

    properties
        Colormap = 'hot'
    end

    properties (Access = private)
        Colors_
        ImageHandle_    % imagesc handle — updated in-place each frame
        DisplayRes_
    end

    methods
        function setup(obj, ax, checkpoint, colors)
            obj.Colors_     = colors;
            obj.DisplayRes_ = checkpoint.DISPLAY_RES;

            G = obj.DisplayRes_;
            set(ax, 'Color', 'k', 'XColor', colors.Grid, 'YColor', colors.Grid);

            obj.ImageHandle_ = imagesc(ax, checkpoint.a_d, checkpoint.b_d, zeros(G, G));
            colormap(ax, obj.Colormap);
            clim(ax, [0 1]);
            set(ax, 'YDir', 'normal');
            hold(ax, 'on');
            xline(ax, 0, 'w:', 'Alpha', 0.4);
            yline(ax, 0, 'w:', 'Alpha', 0.4);
            xlabel(ax, 'Re(c)', 'Color', colors.Text, 'FontSize', 9);
            ylabel(ax, 'Im(c)', 'Color', colors.Text, 'FontSize', 9);
        end

        function render(obj, ax, checkpoint, ~, ep)          % 4th input: the demo frame (unused here)
            ck = checkpoint;
            approxMap = OceanViz.Strategies.DecisionBoundaryRenderer.predictMap(ck, ck.weightHistory{ep});
            set(obj.ImageHandle_, 'CData', approxMap);

            title(ax, sprintf('NN Approximation  |  Epoch %d / %d  |  R^2 = %.4f', ...
                ep, ck.numEpochs, ck.accHistory(ep)), ...
                'Color', obj.Colors_.Text, 'FontSize', 10);
        end
    end

    methods (Static)
        function approxMap = predictMap(ck, v)
            % The network's output on the checkpoint's display grid for ONE flat weight
            % vector v (used by render and by the replay converter -- one code path).
            % Flat vector order: [b1; W1; b2; W2; ... bL; WL]; hidden layers tanh, output sigmoid.
            layers = ck.layerSizes;  % e.g. [128 256 256 1]
            numLayers = numel(layers) - 1;
            ptr = 1;
            W_cell = cell(numLayers, 1);
            b_cell = cell(numLayers, 1);
            for k = 1:numLayers
                dOut = layers(k+1);
                dIn  = layers(k);
                b_cell{k} = v(ptr : ptr + dOut - 1);
                ptr = ptr + dOut;
                W_cell{k} = reshape(v(ptr : ptr + dOut*dIn - 1), dOut, dIn);
                ptr = ptr + dOut * dIn;
            end

            % --- Generic forward pass on display grid ---
            G = ck.DISPLAY_RES;
            A = ck.gridNorm;   % [inputDim x G*G]
            for k = 1:numLayers - 1
                A = tanh(W_cell{k} * A + b_cell{k});           % hidden layers: tanh
            end
            Z_out = W_cell{end} * A + b_cell{end};
            A_out = 1 ./ (1 + exp(-Z_out));                     % output layer: sigmoid, [1 x G*G]
            approxMap = reshape(A_out, G, G);
        end
    end
end
