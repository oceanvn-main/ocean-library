classdef (Abstract) ISampleRenderer < handle
    % ISAMPLERENDERER  Abstract interface for rendering one training sample
    %   into a single axes panel inside NNReplayEngine.
    %
    %   IMPLEMENTATIONS:
    %     OceanViz.Adapters.ImageSampleRenderer   ← 2D pixel grid (MNIST, shapes)
    %     OceanViz.Adapters.ComplexPointRenderer  ← dot on complex plane (Mandelbrot)
    %     OceanViz.Adapters.SignalSampleRenderer  ← 1D waveform (audio, EEG)
    %
    % See also: OceanViz.Visualizers.NNReplayEngine

    methods (Abstract)
        % SETUP  Called once before the animation loop begins.
        %   ax     - Axes handle.
        %   config - struct with renderer-specific options (e.g. IMG_RES).
        %   colors - OceanViz theme colors struct.
        setup(obj, ax, config, colors)

        % RENDER  Called every animation frame.
        %   ax    - Axes handle (same as setup).
        %   frame - struct: .X [INPUT_DIM x 1], .Y (label), .A1, .A2
        %   ep    - current epoch index
        render(obj, ax, frame, ep)

        % GETTITLE  Returns a title string for the sample axes.
        t = getTitle(obj, frame, ep)
    end
end
