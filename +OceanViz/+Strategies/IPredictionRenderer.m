classdef (Abstract) IPredictionRenderer < handle
    % IPREDICTIONRENDERER  Abstract interface for rendering the model's
    %   globally learned output into a single axes panel.
    %
    %   IMPLEMENTATIONS:
    %     OceanViz.Strategies.NetworkGraphRenderer     ← NNGraph (MNIST)
    %     OceanViz.Strategies.DecisionBoundaryRenderer ← color grid (Mandelbrot)
    %
    % See also: OceanViz.Visualizers.NNReplayEngine

    methods (Abstract)
        % SETUP  Called once before the animation loop.
        %   ax         - Axes handle.
        %   checkpoint - NNTrainingCheckpoint struct.
        %   colors     - OceanViz theme colors struct.
        setup(obj, ax, checkpoint, colors)

        % RENDER  Called every animation frame.
        %   ax         - Axes handle.
        %   checkpoint - Full NNTrainingCheckpoint struct.
        %   frame      - struct: .X, .Y, .A1, .A2 for this epoch.
        %   ep         - current epoch index.
        render(obj, ax, checkpoint, frame, ep)
    end
end
