classdef NNGraphData
    % NNGRAPHDATA  Documentation class describing the NNGraphData struct contract.
    %
    %   NNGraphData is a plain MATLAB struct (not a class instance) produced
    %   by any ILayoutEngine.toData() call. It is the only coupling between
    %   the NNGraph model and the NetworkGraphStrategy renderer.
    %
    %   FIELDS:
    %
    %     Positions   (K×2) double
    %         XY coordinate in data units. Row k = position of the k-th node
    %         in flat (layer-major) ordering:
    %           k = sum(LayerSizes(1..L-1)) + nodeIndex
    %
    %     NodeColors  (K×3) double  [0,1]
    %         RGB color per node. Encodes activation value (brightness) or
    %         state (gray if Disabled).
    %
    %     NodeSizes   (K×1) double
    %         Scatter marker size. Base = 120 units. +20% if Highlighted.
    %
    %     NodeStates  (K×1) string
    %         "Active" | "Disabled" | "Highlighted" — for conditional styling.
    %
    %     EdgeFrom    (E×1) double
    %         Flat node index of the source (FromNode) of each edge.
    %
    %     EdgeTo      (E×1) double
    %         Flat node index of the destination (ToNode) of each edge.
    %
    %     EdgeWeights (E×1) double
    %         Raw scalar weight value. Sign encodes color; magnitude encodes width.
    %
    %     EdgeAlpha   (E×1) double
    %         Opacity per edge. 1.0 = both endpoints Active.
    %         0.15 = at least one endpoint is Disabled.
    %
    %     LayerNames  {1×L} cell of strings
    %         Human-readable label per layer, e.g. {"Input (linear)", "Hidden (relu)"}.
    %
    %     NumLayers   (1,1) double
    %         Total number of layers L.
    %
    %     NumNodes    (1,1) double
    %         Total flat node count K.
    %
    % See also: OceanDL.Viz.ILayoutEngine
    %           OceanDL.Viz.NNLayeredLayout
    %           OceanDL.Viz.MatlabNativeLayout

    methods (Static)
        function data = empty()
            % EMPTY  Return a zero-filled NNGraphData struct for initialisation.
            data.Positions   = zeros(0, 2);
            data.NodeColors  = zeros(0, 3);
            data.NodeSizes   = zeros(0, 1);
            data.NodeStates  = strings(0, 1);
            data.EdgeFrom    = zeros(0, 1);
            data.EdgeTo      = zeros(0, 1);
            data.EdgeWeights = zeros(0, 1);
            data.EdgeAlpha   = zeros(0, 1);
            data.LayerNames  = {};
            data.NumLayers   = 0;
            data.NumNodes    = 0;
        end
    end
end
