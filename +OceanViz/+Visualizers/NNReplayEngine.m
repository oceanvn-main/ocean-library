classdef NNReplayEngine < handle
    % NNREPLAYENGINE  Generic training-replay animation engine.
    %
    %   Loads an NNTrainingCheckpoint and animates the full training history
    %   using pluggable renderers and a swappable dashboard layout.
    %   Completely agnostic to dataset and model type.
    %
    %   Supports SPLIT checkpoints for large experiments:
    %     - Main  file (.mat)         : metadata, demo frames (loads instantly)
    %     - Sidecar file (_weights.mat): weightHistory only (loaded separately)
    %   Falls back gracefully to a single combined file (backward compatible).
    %
    %   USAGE:
    %     engine = OceanViz.Visualizers.NNReplayEngine();
    %     engine.SampleRenderer     = OceanViz.Adapters.ImageSampleRenderer();
    %     engine.PredictionRenderer = OceanViz.Strategies.NetworkGraphRenderer();
    %     engine.Layout             = OceanViz.Visualizers.NNGraphLayout();
    %     engine.run('path/to/checkpoint.mat');
    %
    % See also: OceanViz.Adapters.ISampleRenderer
    %           OceanViz.Strategies.IPredictionRenderer
    %           OceanViz.Visualizers.IReplayLayout

    properties
        SampleRenderer           % ISampleRenderer
        PredictionRenderer       % IPredictionRenderer
        Layout                   % IReplayLayout
        Theme                    = 'Dark'
        FigureTitle              = 'NN Training Replay'
        FPS                      = 10
        EpochStep                = 1    % Stride/step size for epoch loop (e.g. 2 or 5 for faster replay)
        Description              = ''   % Optional compact description string displayed at figure footer
        SampleRendererConfig     = struct()
    end

    properties (SetAccess = private)
        Axes  struct   % Populated after run(): .axCost .axAcc .axSample .axPred
    end

    methods
        function run(obj, checkpointPath)
            assert(~isempty(obj.SampleRenderer),     'OceanViz:NNReplayEngine:MissingPlugin', 'Set SampleRenderer first.');
            assert(~isempty(obj.PredictionRenderer), 'OceanViz:NNReplayEngine:MissingPlugin', 'Set PredictionRenderer first.');
            assert(~isempty(obj.Layout),             'OceanViz:NNReplayEngine:MissingPlugin', 'Set Layout first.');

            % --- Load metadata (fast: no weightHistory in this file) ---
            fprintf('[NNReplayEngine] Loading metadata: %s\n', checkpointPath);
            ck = load(checkpointPath);

            % --- Load weightHistory from sidecar if available ---
            [dir_, name_, ~] = fileparts(checkpointPath);
            weightsPath = fullfile(dir_, [name_, '_weights.mat']);
            if isfield(ck, 'weightHistory')
                fprintf('  weightHistory found in main file (%d epochs).\n', numel(ck.weightHistory));
            elseif isfile(weightsPath)
                fprintf('[NNReplayEngine] Loading weights sidecar: %s\n', weightsPath);
                wdata = load(weightsPath, 'weightHistory');
                ck.weightHistory = wdata.weightHistory;
                fprintf('  Weights loaded (%d epochs).\n', numel(ck.weightHistory));
            else
                error('OceanViz:NNReplayEngine:MissingWeights', ...
                    'weightHistory not found in checkpoint or sidecar:\n  %s', weightsPath);
            end

            fprintf('  Network: %s  |  Epochs: %d\n\n', ...
                mat2str(ck.layerSizes), ck.numEpochs);

            themeCore = OceanViz.Core('Theme', obj.Theme);
            colors    = themeCore.Colors;

            fig = figure('Name', obj.FigureTitle, 'NumberTitle', 'off', ...
                'Color', colors.Background, 'Position', [80, 80, 1380, 780], ...
                'MenuBar', 'none', 'ToolBar', 'none');

            [axCost, axAcc, axSample, axPred] = obj.Layout.buildLayout(fig, ck, colors);

            % Store axes for post-replay access (e.g. zoom controllers)
            obj.Axes = struct('axCost', axCost, 'axAcc', axAcc, ...
                              'axSample', axSample, 'axPred', axPred);

            lineCost = animatedline(axCost, 'Color', colors.Secondary, 'LineWidth', 2);
            lineAcc  = animatedline(axAcc,  'Color', colors.Primary,   'LineWidth', 2);

            obj.SampleRenderer.setup(axSample, obj.SampleRendererConfig, colors);
            obj.PredictionRenderer.setup(axPred, ck, colors);

            btn = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Stop', ...
                'Position', [20 20 70 32], 'FontSize', 10, ...
                'BackgroundColor', colors.Secondary);
            btn.Callback = @(~,~) setappdata(gcf, 'stopAnim', true);
            setappdata(fig, 'stopAnim', false);

            if ~isempty(obj.Description)
                uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
                    'Position', [0.05 0.002 0.90 0.045], ...
                    'String', obj.Description, ...
                    'FontSize', 8, 'HorizontalAlignment', 'center', ...
                    'BackgroundColor', colors.Background, 'ForegroundColor', colors.Text * 0.7);
            end

            fprintf('[NNReplayEngine] Animating %d frames at %d FPS (EpochStep=%d)...\n', ...
                ck.numEpochs, obj.FPS, obj.EpochStep);

            for ep = 1:obj.EpochStep:ck.numEpochs
                stopRequested = false;
                try, stopRequested = getappdata(fig, 'stopAnim'); catch, end
                if stopRequested || ~isvalid(fig), break; end

                frame.X  = ck.demoX(:, ep);
                frame.Y  = ck.demoY(ep);
                frame.A1 = ck.demoA1(:, ep);
                frame.A2 = ck.demoA2(:, ep);

                obj.SampleRenderer.render(axSample, frame, ep);
                obj.PredictionRenderer.render(axPred, ck, frame, ep);

                addpoints(lineCost, ep, ck.costHistory(ep));
                addpoints(lineAcc,  ep, ck.accHistory(ep));

                drawnow limitrate;
                pause(1 / obj.FPS);
            end

            fprintf('[NNReplayEngine] Replay complete.\n');
        end
    end
end
