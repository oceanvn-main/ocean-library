classdef (Abstract) IReplayLayout < handle
    % IREPLAYLAYOUT  Abstract interface for NNReplayEngine dashboard layout.
    %
    %   IMPLEMENTATIONS:
    %     OceanViz.Visualizers.NNGraphLayout  ← MNIST / classification
    %     OceanViz.Visualizers.BoundaryLayout ← Mandelbrot / shape boundary
    %
    % See also: OceanViz.Visualizers.NNReplayEngine

    methods (Abstract)
        % BUILDLAYOUT  Create figure axes and return handles.
        %   fig    - Figure handle (pre-created by the engine).
        %   ck     - NNTrainingCheckpoint struct.
        %   colors - OceanViz theme colors.
        %
        %   Returns axCost, axAcc, axSample, axPred.
        [axCost, axAcc, axSample, axPred] = buildLayout(obj, fig, ck, colors)
    end
end
