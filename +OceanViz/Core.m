classdef Core < handle
    % OCEANVIZ.CORE The central hub.
    
    properties
        ThemeName (1,1) string
        Colors struct
        ConfigPath
    end
    
    methods
        function obj = Core(options)
            arguments
                options.Theme (1,1) string = "Dark"
                options.Config (1,1) string = ""
            end
            
            obj.ThemeName = options.Theme;
            
            % 1. Determine Path
            if options.Config == ""
                % Get the folder where THIS file (Core.m) lives
                [currentFolder, ~, ~] = fileparts(mfilename('fullpath'));
                obj.ConfigPath = fullfile(currentFolder, 'Themes.json');
            else
                obj.ConfigPath = options.Config;
            end
            
            % 2. Attempt Load (With Safety Net)
            try
                obj.loadTheme();
            catch ME
                warning('OceanViz:ConfigFail', ...
                    'Could not load JSON from: %s\nError: %s\nUSING HARDCODED FALLBACK.', ...
                    obj.ConfigPath, ME.message);
                obj.loadFallbackTheme();
            end
        end

        function palette = getColorSequence(obj, n)
            % GETCOLORSEQUENCE Returns an [N x 3] matrix of RGB colors.
            % scalable to any number of datasets.
            
            c1 = obj.ensureRow(obj.Colors.Primary);
            c2 = obj.ensureRow(obj.Colors.Secondary);
            c3 = obj.ensureRow(obj.Colors.Boundary);

            % 1. Start with the "Branded" Theme Colors
            % These are the high-priority colors defined in JSON
            branded = [c1; c2; c3];
            n_branded = size(branded, 1);
            
            if n <= n_branded
                % Scenario A: We have enough branded colors (e.g., n=2 or 3)
                palette = branded(1:n, :);
            else
                % Scenario B: User asked for more (e.g., n=5 or 10)
                % We keep the branded ones for consistency, and generate the rest.
                palette = zeros(n, 3);
                palette(1:n_branded, :) = branded;
                
                % Generate remaining colors using a high-contrast map
                n_needed = n - n_branded;
                
                % 'lines' is good for categorical, 'turbo' for high contrast on dark
                if obj.ThemeName == "Dark"
                    % Turbo is excellent for dark mode visibility
                    extra_colors = turbo(n_needed + 2); 
                    palette(n_branded+1:end, :) = extra_colors(2:end-1, :); % Skip extremes
                else
                    % Scientific/Light mode: 'lines' is cleaner
                    extra_colors = lines(n);
                    % We take from the end to avoid clashing with our custom blue/reds
                    palette(n_branded+1:end, :) = extra_colors(end-n_needed+1:end, :);
                end
            end
        end

        function vec = ensureRow(~, vec)
            % Helper: Forces 3x1 column vectors into 1x3 row vectors
            if size(vec, 1) > 1 && size(vec, 2) == 1
                vec = vec';
            end
        end
        
        function loadTheme(obj)
            % Check file existence
            if ~isfile(obj.ConfigPath)
                error("File not found");
            end
            
            % Read and Decode
            rawText = fileread(obj.ConfigPath);
            rawStruct = jsondecode(rawText);
            
            % Check for Theme
            if ~isfield(rawStruct, obj.ThemeName)
                error("Theme '%s' missing in JSON", obj.ThemeName);
            end
            
            obj.Colors = rawStruct.(obj.ThemeName);
            
            % Coerce all decoded colors to row vectors [r, g, b] 
            % Since jsondecode natively parses arrays as column vectors.
            fields = fieldnames(obj.Colors);
            for i = 1:numel(fields)
                val = obj.Colors.(fields{i});
                if isnumeric(val)
                    obj.Colors.(fields{i}) = obj.ensureRow(val);
                end
            end
            
            % DEBUG: Confirm it worked
            fprintf('[OceanViz] Successfully loaded theme "%s" from JSON.\n', obj.ThemeName);
        end
        
        function loadFallbackTheme(obj)
            % EMERGENCY BACKUP: If JSON fails, use these values so code runs.
            if obj.ThemeName == "Dark"
                obj.Colors.Background = [0.11, 0.11, 0.13];
                obj.Colors.Text       = [0.90, 0.90, 0.90];
                obj.Colors.Grid       = [0.25, 0.25, 0.25];
                obj.Colors.Primary    = [0.20, 0.60, 1.00];
                obj.Colors.Secondary  = [1.00, 0.40, 0.40];
                obj.Colors.Scatter    = [0.80, 0.80, 0.80];
                obj.Colors.Boundary   = [1.00, 0.80, 0.20];
            else
                % Scientific Fallback
                obj.Colors.Background = [1.00, 1.00, 1.00];
                obj.Colors.Text       = [0.00, 0.00, 0.00];
                obj.Colors.Grid       = [0.90, 0.90, 0.90];
                obj.Colors.Primary    = [0.00, 0.45, 0.74];
                obj.Colors.Secondary  = [0.85, 0.33, 0.10];
                obj.Colors.Scatter    = [0.20, 0.20, 0.20];
                obj.Colors.Boundary   = [0.10, 0.10, 0.10];
            end
        end
        
        function fig = createFigure(obj, name)
            fig = figure('Name', name, ...
                         'Color', obj.Colors.Background, ...
                         'NumberTitle', 'off', ...
                         'Renderer', 'painters', ...
                         'InvertHardcopy', 'off');   
        end
        
        function plot(obj, ax, strategyName, data, options)
            arguments
                obj, ax
                strategyName (1,1) string
                data struct
                options.Title = "Analysis"
                options.XLabel = "X"
                options.YLabel = "Y"
                options.ZLabel = "Z"
                options.ShowLegend (1,1) logical = true
                options.ShowComplexity (1,1) logical = true
                options.Projection = "2D"
                
                %Surface & 3D Specific Options
                options.FaceAlpha (1, 1) double = 0.85
                options.ViewAngle (1, 2) double = [-35, 25]
                %Surface-specific flags
                options.ShowScatter (1, 1) logical = true
                options.ShowContour (1, 1) logical = false
                options.ShowGradient (1, 1) logical = false

                % Legacy DualMetrics support
                options.YLabelRight (1, 1) string = "Metric 2"
                options.LegendLabels (1, 2) string = ["Metric 1", "Metric 2"]
                
                % Multi-Layer Interpretation
                options.ZMode (1, 1) string = "Label"

                % PartialDependence2D-specific options
                options.Colormap     (1,1) string  = "hot"
                options.AzimuthDeg   (1,1) double  = -50
                options.ElevationDeg (1,1) double  =  28
                options.EdgeAlpha    (1,1) double  = 0.2
                options.ShowColorbar (1,1) logical = true

                

                % GeographicScatter options
                options.DotSize        (1,1) double  = 12
                options.DotAlpha       (1,1) double  = 0.75
                options.ColorLabel     (1,1) string  = "Value"
                options.SymmetricColor (1,1) logical = true
                options.AspectRatio    (1,1) string  = "equal"
                options.OverlayX             double  = []
                options.OverlayY             double  = []
                options.BoundaryState  (1,1) string  = ""
                % LikelihoodOptimization-specific options
                options.Animate        (1,1) logical = false
                options.PauseTime      (1,1) double  = 0.05
                options.SaveGif        (1,1) string  = ""
            end

            %Instan
            
            switch strategyName
                case "Regression"
                    strategy = OceanViz.Strategies.Regression();
                case "Classification"
                    strategy = OceanViz.Strategies.Classification();
                case "DualMetrics"
                    strategy = OceanViz.Strategies.DualMetrics();
                case "MultiComparison"
                    strategy = OceanViz.Strategies.MultiComparison();
                case "LearningCurve"
                    strategy = OceanViz.Strategies.LearningCurve();
                case "BarGroup"
                    strategy = OceanViz.Strategies.BarGroup();
                case "Surface"
                    strategy = OceanViz.Strategies.Surface();
                case "Calibration"
                    strategy = OceanViz.Strategies.Calibration();
                case "FeatureAudit"
                    strategy = OceanViz.Strategies.FeatureAudit();
                case "Distribution"
                    strategy = OceanViz.Strategies.Distribution();
                case "PartialDependence2D"
                    strategy = OceanViz.Strategies.PartialDependence2D();
                case "GeographicScatter"
                    strategy = OceanViz.Strategies.GeographicScatter();
                case "PrototypeClassifier"
                    strategy = OceanViz.Strategies.PrototypeClassifierStrategy();
                case "FunctionalData"
                    strategy = OceanViz.Strategies.FunctionalDataStrategy();
                case "AssociationRule"
                    strategy = OceanViz.Strategies.AssociationRuleStrategy();
                case "SOMGrid"
                    strategy = OceanViz.Strategies.SOMGridStrategy();
                case "HuffmanTree"
                    strategy = OceanViz.Strategies.HuffmanTreeStrategy();
                case "CodeLengthBar"
                    strategy = OceanViz.Strategies.CodeLengthBarStrategy();
                case "TimeSeries"
                    strategy = OceanViz.Strategies.TimeSeriesVizStrategy();
                case "ACF"
                    strategy = OceanViz.Strategies.ACFStrategy();
                case "Candlestick"
                    strategy = OceanViz.Strategies.CandlestickStrategy();
                case "Image"
                    strategy = OceanViz.Strategies.ImageVizStrategy();
                case "ScatterInterp"
                    strategy = OceanViz.Strategies.ScatterInterpStrategy();
                case "TabularRL"
                    strategy = OceanViz.Strategies.TabularRLStrategy();
                case "LikelihoodOptimization"
                    strategy = OceanViz.Strategies.LikelihoodOptimizationStrategy();
                case "TriMesh"
                    strategy = OceanViz.Strategies.TriMeshStrategy();
                case "SpeechWaveform"
                    strategy = OceanViz.Strategies.Speech.WaveformStrategy();
                case "Spectrogram"
                    strategy = OceanViz.Strategies.Speech.SpectrogramStrategy();
                case "MelFilterBank"
                    strategy = OceanViz.Strategies.Speech.MelFilterBankStrategy();
                case "SpeechFeatures"
                    strategy = OceanViz.Strategies.Speech.FeatureMatrixStrategy();
                case "PitchContour"
                    strategy = OceanViz.Strategies.Speech.PitchContourStrategy();
                case "CTCPosterior"
                    strategy = OceanViz.Strategies.Speech.PosteriorStrategy();
                case "BeamCandidates"
                    strategy = OceanViz.Strategies.Speech.BeamCandidatesStrategy();
                case "ErrorAlignment"
                    strategy = OceanViz.Strategies.Speech.ErrorAlignmentStrategy();
                case "SpeechContour"
                    strategy = OceanViz.Strategies.Speech.SpeechContourStrategy();
                case "SpeechFrame"
                    strategy = OceanViz.Strategies.Speech.SpeechFrameStrategy();
                otherwise
                    error("Unknown strategy: " + strategyName);
            end
            
            %Dependency Injection
            strategy.Parent = obj;

            %Execute      
            obj.styleAxes(ax);

            strategy.plot(ax, data, options);
        end
        
        function styleAxes(obj, ax)
            set(ax, 'Color', obj.Colors.Background, ...
                    'XColor', obj.Colors.Text, ...
                    'YColor', obj.Colors.Text, ...
                    'GridColor', obj.Colors.Grid, ...
                    'GridAlpha', 0.5, ...
                    'LineWidth', 1.2, ...
                    'FontName', 'Helvetica', ...
                    'TickLabelInterpreter', 'latex');
            grid(ax, 'on');
            axis(ax, 'tight');
        end
        
        function export(obj, fig, filename)
            OceanViz.Utils.Exporter.save(fig, filename);
        end
    end
end