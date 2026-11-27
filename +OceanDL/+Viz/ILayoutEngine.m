classdef (Abstract) ILayoutEngine < handle
    % ILAYOUTENGINE  Abstract interface for NNGraph → NNGraphData converters.
    % BRIDGE PATTERN: "What are final coordinates, color, and states of the neurons? Pure data package."
    %
    %   Every layout backend must implement toData(), which converts a live
    %   NNGraph into a plain NNGraphData struct. The struct is the only
    %   coupling between the model layer (NNGraph) and the view layer
    %   (NetworkGraphStrategy).
    %
    %   This is the BRIDGE in the Bridge pattern:
    %     - Abstraction : NetworkGraphStrategy (renders from NNGraphData)
    %     - Implementor : ILayoutEngine (decides where nodes go)
    %
    %   Concrete implementations:
    %     NNLayeredLayout     — custom layered-grid positions (default)
    %     MatlabNativeLayout  — uses MATLAB digraph layout algorithms
    %
    % See also: OceanDL.Viz.NNLayeredLayout
    %           OceanDL.Viz.MatlabNativeLayout
    %           OceanDL.Viz.NNGraphData

    methods (Abstract)
        data = toData(obj, graph, options)
        % TODATA  Convert a live NNGraph to an NNGraphData struct.
        %
        % INPUTS:
        %   graph - OceanDL.Graph.NNGraph handle
        %
        % OUTPUTS:
        %   data - struct with fields:
        %     .Positions   (K×2) double  XY coordinate per node (flat index)
        %     .NodeColors  (K×3) double  RGB per node  [0,1]
        %     .NodeSizes   (K×1) double  scatter marker size per node
        %     .NodeStates  (K×1) string  "Active"|"Disabled"|"Highlighted"
        %     .EdgeFrom    (E×1) double  source flat node index
        %     .EdgeTo      (E×1) double  destination flat node index
        %     .EdgeWeights (E×1) double  scalar weight value
        %     .EdgeAlpha   (E×1) double  opacity: 1.0=active, 0.15=inactive
        %     .LayerNames  {1×L} cell    layer label strings
        %     .NumLayers   (1,1) double  number of layers L
        %     .NumNodes    (1,1) double  total K = sum of all layer sizes
    end
end
