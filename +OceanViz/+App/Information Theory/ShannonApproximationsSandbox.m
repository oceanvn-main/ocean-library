classdef ShannonApproximationsSandbox < handle
    % SHANNONAPPROXIMATIONSSANDBOX Programmatic responsive dark-mode Shannon Text Approximations dashboard.
    %
    % This sandbox explores systematic entropy reduction in text as N-gram Markov order
    % context increases, and visualizes Shannon's original physical book-flip Monte Carlo search.
    %
    % See also: OceanMath.Information.Processes.ShannonTextGenerator,
    %           OceanMath.Information.Processes.MonteCarloSearchEngine,
    %           OceanViz.Strategies.CorpusSearchRenderer,
    %           OceanViz.Strategies.EntropySpectrumRenderer

    properties
        FigureHandle
        
        % Layout Containers
        MainGridLayout
        LeftPanel
        CenterPanel
        RightPanel
        
        % Controls (Left Panel)
        OrderButtonGroup
        ModeDropdown
        CorpusDropdown
        BtnLoadCustom
        BtnPlayPause
        BtnStep
        BtnReset
        SpeedSlider
        SpeedLabel
        
        % Canvas Components (Center Panel)
        BookAxes
        RetroTerminal
        GraphPanel
        AxProb
        AxEntropy
        
        % Metrics Components (Right Panel)
        LblActiveContext
        LblH0
        LblH1
        LblH2
        LblH3
        LblActiveEntropyValue
        EntropyDecayBadge
        
        % Engines and Renderers
        Generator
        SearchEngine
        BookRenderer
        SpectrumRenderer
        
        % Application State
        GeneratedText = char.empty(1, 0)
        ActiveContext = char.empty(1, 0)
        PlaybackTimer
        IsPlaying = false
        
        % Color Theme Palette (Modern Dark Slate)
        COLOR_BG = [0.13, 0.14, 0.17]
        COLOR_PANEL_BG = [0.17, 0.19, 0.23]
        COLOR_TEXT = [0.85, 0.88, 0.93]
        COLOR_SUBTEXT = [0.60, 0.65, 0.70]
        COLOR_CYAN = [0.30, 0.65, 0.95]
        COLOR_GREEN = [0.15, 0.65, 0.40]
        COLOR_MUTED_GREEN = [0.08, 0.25, 0.18]
        COLOR_ORANGE = [0.90, 0.55, 0.15]
    end
    
    methods
        function obj = ShannonApproximationsSandbox()
            % Construct the Shannon Sandbox app programmatically in invisible state
            obj.FigureHandle = uifigure('Name', 'Shannon Text Approximations Sandbox', ...
                'Color', obj.COLOR_BG, ...
                'Position', [100, 100, 1180, 780], ...
                'Visible', 'off');
            
            % Figure close request cleanup hook to prevent resource leaks
            obj.FigureHandle.CloseRequestFcn = @(src, event) obj.closeApp();
            
            % Main responsive grid layout
            obj.MainGridLayout = uigridlayout(obj.FigureHandle, [2, 3], ...
                'RowHeight', {50, '1x'}, ...
                'ColumnWidth', {290, '1x', 270}, ...
                'BackgroundColor', obj.COLOR_BG);
            
            % Premium Application Header
            titleLabel = uilabel(obj.MainGridLayout, ...
                'Text', 'SHANNON TEXT APPROXIMATIONS STUDIO', ...
                'FontName', 'Helvetica', ...
                'FontSize', 18, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_CYAN, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'center');
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = [1, 3];
            
            % -------------------------------------------------------------
            % 1. LEFT CONTROLS PANEL
            % -------------------------------------------------------------
            obj.LeftPanel = uipanel(obj.MainGridLayout, ...
                'Title', 'SIMULATION CONTROLS', ...
                'BackgroundColor', obj.COLOR_PANEL_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Helvetica', ...
                'FontWeight', 'bold', ...
                'FontSize', 11);
            obj.LeftPanel.Layout.Row = 2;
            obj.LeftPanel.Layout.Column = 1;
            
            leftGrid = uigridlayout(obj.LeftPanel, [8, 1], ...
                'RowHeight', {125, 55, 55, 35, 25, 40, 25, 45}, ...
                'Padding', [10, 10, 10, 10], ...
                'RowSpacing', 8, ...
                'BackgroundColor', obj.COLOR_PANEL_BG);
            
            % Approximation Order Selector
            obj.OrderButtonGroup = uibuttongroup(leftGrid, ...
                'Title', 'Markov Approximation Order', ...
                'BackgroundColor', obj.COLOR_PANEL_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Helvetica', ...
                'FontWeight', 'bold', ...
                'SelectionChangedFcn', @(src, event) obj.changeOrderCallback());
                
            btn0 = uiradiobutton(obj.OrderButtonGroup, 'Text', 'Zero-Order (Uniform Chaos)', 'UserData', 0, 'FontColor', obj.COLOR_TEXT, 'Position', [10, 80, 240, 22]);
            btn1 = uiradiobutton(obj.OrderButtonGroup, 'Text', 'First-Order (Unigram Freqs)', 'UserData', 1, 'FontColor', obj.COLOR_TEXT, 'Position', [10, 58, 240, 22]);
            btn2 = uiradiobutton(obj.OrderButtonGroup, 'Text', 'Second-Order (Bigram Markov)', 'UserData', 2, 'FontColor', obj.COLOR_TEXT, 'Position', [10, 36, 240, 22]);
            btn3 = uiradiobutton(obj.OrderButtonGroup, 'Text', 'Third-Order (Trigram Markov)', 'UserData', 3, 'FontColor', obj.COLOR_TEXT, 'Position', [10, 14, 240, 22]);
            
            % Simulation Mode Selector
            modeGrid = uigridlayout(leftGrid, [2, 1], 'Padding', [0, 0, 0, 0], 'RowSpacing', 2, 'BackgroundColor', obj.COLOR_PANEL_BG);
            uilabel(modeGrid, 'Text', 'Generation Engine Mode:', 'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold');
            obj.ModeDropdown = uidropdown(modeGrid, ...
                'Items', {'Monte Carlo Book Search (Animated Step)', 'Analytical Markov Table (Instant Draw)'}, ...
                'Value', 'Monte Carlo Book Search (Animated Step)', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT);
            
            % Corpus Selector
            corpGrid = uigridlayout(leftGrid, [2, 1], 'Padding', [0, 0, 0, 0], 'RowSpacing', 2, 'BackgroundColor', obj.COLOR_PANEL_BG);
            uilabel(corpGrid, 'Text', 'Source Text Database (Book):', 'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold');
            obj.CorpusDropdown = uidropdown(corpGrid, ...
                'Items', {'Shakespeare (Classic)', 'Scientific Journal (Technical)', 'Standard English (General)'}, ...
                'Value', 'Shakespeare (Classic)', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(src, event) obj.corpusDropdownCallback());
            
            % Load Custom Corpus Button
            obj.BtnLoadCustom = uibutton(leftGrid, ...
                'Text', 'Load Custom Corpus (.txt)', ...
                'ButtonPushedFcn', @(btn, event) obj.loadCustomCorpus(), ...
                'BackgroundColor', [0.25, 0.28, 0.35], ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontWeight', 'bold');
            
            % Playback Control Buttons
            uilabel(leftGrid, 'Text', 'Playback Controllers:', 'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold');
            
            btnRowGrid = uigridlayout(leftGrid, [1, 3], 'Padding', [0, 0, 0, 0], 'ColumnSpacing', 5, 'BackgroundColor', obj.COLOR_PANEL_BG);
            obj.BtnPlayPause = uibutton(btnRowGrid, 'Text', 'Play', ...
                'ButtonPushedFcn', @(btn, event) obj.togglePlayback(), ...
                'BackgroundColor', [0.15, 0.50, 0.80], 'FontColor', [1,1,1], 'FontWeight', 'bold');
            obj.BtnStep = uibutton(btnRowGrid, 'Text', 'Step', ...
                'ButtonPushedFcn', @(btn, event) obj.runStep(), ...
                'BackgroundColor', [0.25, 0.27, 0.32], 'FontColor', [1,1,1], 'FontWeight', 'bold');
            obj.BtnReset = uibutton(btnRowGrid, 'Text', 'Reset', ...
                'ButtonPushedFcn', @(btn, event) obj.resetSandbox(), ...
                'BackgroundColor', [0.60, 0.25, 0.25], 'FontColor', [1,1,1], 'FontWeight', 'bold');
                
            % Playback Speed Slider
            obj.SpeedLabel = uilabel(leftGrid, 'Text', 'Step Playback Interval: 200 ms', 'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold');
            
            obj.SpeedSlider = uislider(leftGrid, ...
                'Limits', [50, 2000], ...
                'Value', 200, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(src, event) obj.speedSliderCallback());
            
            % -------------------------------------------------------------
            % 2. CENTRAL CANVAS PANEL
            % -------------------------------------------------------------
            obj.CenterPanel = uipanel(obj.MainGridLayout, ...
                'Title', 'ANIMATED MONTE CARLO SCAN & TERMINAL SPECTRA', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Helvetica', ...
                'FontWeight', 'bold', ...
                'FontSize', 11);
            obj.CenterPanel.Layout.Row = 2;
            obj.CenterPanel.Layout.Column = 2;
            
            centerGrid = uigridlayout(obj.CenterPanel, [3, 1], ...
                'RowHeight', {180, 100, '1x'}, ...
                'RowSpacing', 10, ...
                'Padding', [10, 10, 10, 10], ...
                'BackgroundColor', obj.COLOR_BG);
            
            % Physical Book-Scan Visualizer Axes
            obj.BookAxes = uiaxes(centerGrid);
            obj.BookAxes.Layout.Row = 1;
            obj.BookAxes.Layout.Column = 1;
            title(obj.BookAxes, 'Shannon Physical Ancestral Search Grid (Corpus Sub-window)', 'Color', obj.COLOR_TEXT);
            
            % Glowing retro teleprinter text area
            obj.RetroTerminal = uitextarea(centerGrid, ...
                'Editable', 'off', ...
                'FontName', 'Courier New', ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'BackgroundColor', [0.07, 0.08, 0.10], ...
                'FontColor', [0.30, 0.95, 0.65]);
            obj.RetroTerminal.Layout.Row = 2;
            obj.RetroTerminal.Layout.Column = 1;
            
            % Double metrics visualization panel
            obj.GraphPanel = uipanel(centerGrid, ...
                'BorderType', 'none', ...
                'BackgroundColor', obj.COLOR_BG);
            obj.GraphPanel.Layout.Row = 3;
            obj.GraphPanel.Layout.Column = 1;
            
            graphGrid = uigridlayout(obj.GraphPanel, [1, 2], ...
                'ColumnWidth', {'1.1x', '1x'}, ...
                'Padding', [0, 0, 0, 0], ...
                'BackgroundColor', obj.COLOR_BG);
            
            obj.AxProb = uiaxes(graphGrid);
            obj.AxProb.Layout.Row = 1;
            obj.AxProb.Layout.Column = 1;
            
            obj.AxEntropy = uiaxes(graphGrid);
            obj.AxEntropy.Layout.Row = 1;
            obj.AxEntropy.Layout.Column = 2;
            
            % -------------------------------------------------------------
            % 3. RIGHT METRICS PANEL
            % -------------------------------------------------------------
            obj.RightPanel = uipanel(obj.MainGridLayout, ...
                'Title', 'INFORMATION METRICS', ...
                'BackgroundColor', obj.COLOR_PANEL_BG, ...
                'ForegroundColor', obj.COLOR_TEXT, ...
                'FontName', 'Helvetica', ...
                'FontWeight', 'bold', ...
                'FontSize', 11);
            obj.RightPanel.Layout.Row = 2;
            obj.RightPanel.Layout.Column = 3;
            
            rightGrid = uigridlayout(obj.RightPanel, [8, 1], ...
                'RowHeight', {30, 40, 30, 140, 30, 50, 40, '1x'}, ...
                'RowSpacing', 12, ...
                'Padding', [10, 10, 10, 10], ...
                'BackgroundColor', obj.COLOR_PANEL_BG);
            
            % Active Context Display
            uilabel(rightGrid, 'Text', 'ACTIVE SEARCH CONTEXT:', ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold', 'FontSize', 10);
                
            obj.LblActiveContext = uilabel(rightGrid, ...
                'Text', 'Context: '' ''', ...
                'FontName', 'Courier New', ...
                'FontSize', 14, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_CYAN, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'center', ...
                'BackgroundColor', [0.10, 0.12, 0.15]);
            
            % Analytical Entropies Grid
            uilabel(rightGrid, 'Text', 'ANALYTICAL CORPUS ENTROPIES:', ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold', 'FontSize', 10);
                
            entropyGrid = uigridlayout(rightGrid, [4, 2], ...
                'Padding', [5, 5, 5, 5], ...
                'RowSpacing', 3, ...
                'BackgroundColor', [0.14, 0.15, 0.19]);
                
            uilabel(entropyGrid, 'Text', 'H0 (Uniform):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblH0 = uilabel(entropyGrid, 'Text', '4.7549 bits', 'FontColor', obj.COLOR_CYAN, 'FontSize', 10, 'FontWeight', 'bold');
            
            uilabel(entropyGrid, 'Text', 'H1 (Unigram):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblH1 = uilabel(entropyGrid, 'Text', '0.0000 bits', 'FontColor', obj.COLOR_CYAN, 'FontSize', 10, 'FontWeight', 'bold');
            
            uilabel(entropyGrid, 'Text', 'H2 (Bigram):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblH2 = uilabel(entropyGrid, 'Text', '0.0000 bits', 'FontColor', obj.COLOR_CYAN, 'FontSize', 10, 'FontWeight', 'bold');
            
            uilabel(entropyGrid, 'Text', 'H3 (Trigram):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblH3 = uilabel(entropyGrid, 'Text', '0.0000 bits', 'FontColor', obj.COLOR_CYAN, 'FontSize', 10, 'FontWeight', 'bold');
            
            % Active Entropy Value
            uilabel(rightGrid, 'Text', 'ACTIVE DECAY ENTROPY (H_N):', ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontWeight', 'bold', 'FontSize', 10);
                
            obj.LblActiveEntropyValue = uilabel(rightGrid, ...
                'Text', '4.755 bits', ...
                'FontName', 'Helvetica', ...
                'FontSize', 22, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_CYAN, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'center');
            
            % Glowing Status Badge
            obj.EntropyDecayBadge = uilabel(rightGrid, ...
                'Text', 'MAXIMAL ENTROPY (CHAOS)', ...
                'FontName', 'Helvetica', ...
                'FontSize', 11, ...
                'FontWeight', 'bold', ...
                'FontColor', [1, 1, 1], ...
                'BackgroundColor', obj.COLOR_ORANGE, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'center');
                
            % -------------------------------------------------------------
            % TIMER INITIALIZATION & GRAPHICS RENDERING ATTACHMENT
            % -------------------------------------------------------------
            obj.PlaybackTimer = timer(...
                'ExecutionMode', 'fixedRate', ...
                'Period', 0.20, ...
                'TimerFcn', @(src, event) obj.runStep(), ...
                'BusyMode', 'drop');
            
            % Instantiate strategies
            obj.BookRenderer = OceanViz.Strategies.CorpusSearchRenderer(obj.BookAxes);
            obj.SpectrumRenderer = OceanViz.Strategies.EntropySpectrumRenderer(obj.AxProb, obj.AxEntropy);
            
            % Load built-in default Shakespeare corpus
            cleaned = obj.loadBuiltInCorpus('Shakespeare (Classic)');
            obj.loadCorpus(cleaned);
        end
        
        function show(obj)
            % Makes the figure visible and triggers initial rendering
            obj.FigureHandle.Visible = 'on';
            drawnow;
        end
        
        function text = loadBuiltInCorpus(obj, name)
            % LOADBUILTINCORPUS Robustly resolves base paths to load txt corpora files.
            classPath = which('ShannonApproximationsSandbox');
            if isempty(classPath)
                classPath = mfilename('fullpath');
            end
            [classDir, ~, ~] = fileparts(classPath);
            corporaDir = fullfile(classDir, 'Corpora');
            
            switch name
                case {'Shakespeare (Classic)', 'shakespeare'}
                    fileName = 'shakespeare.txt';
                case {'Scientific Journal (Technical)', 'scientific'}
                    fileName = 'scientific.txt';
                case {'Standard English (General)', 'standard_english'}
                    fileName = 'standard_english.txt';
                otherwise
                    fileName = '';
            end
            
            if ~isempty(fileName)
                filePath = fullfile(corporaDir, fileName);
                if exist(filePath, 'file')
                    text = OceanMath.Information.Processes.CorpusSanitizer.sanitizeFile(filePath);
                    return;
                end
            end
            
            % Safe fallback texts in case files are missing/not found
            switch name
                case {'Shakespeare (Classic)', 'shakespeare'}
                    text = 'TO BE OR NOT TO BE THAT IS THE QUESTION WHETHER TIS NOBLER IN THE MIND TO SUFFER THE SLINGS AND ARROWS OF OUTRAGEOUS FORTUNE OR TO TAKE ARMS AGAINST A SEA OF TROUBLES AND BY OPPOSING END THEM TO DIE TO SLEEP NO MORE';
                case {'Scientific Journal (Technical)', 'scientific'}
                    text = 'THE THERMODYNAMIC EFFICIENCY OF AN IDEAL CARNOT CYCLE IS DETERMINED SOLELY BY THE TEMPERATURE DIFFERENCE BETWEEN THE HOT AND COLD RESERVOIRS WE QUANTIFY THE ENTROPY PRODUCTION';
                otherwise
                    text = 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG THIS ANCIENT PANGRAM CONTAINS EVERY SINGLE LETTER OF THE ENGLISH ALPHABET MAKING IT PERFECT FOR STATISTICAL TESTING AND VISUAL REPRESENTATIONS';
            end
            text = OceanMath.Information.Processes.CorpusSanitizer.sanitize(text);
        end
        
        function loadCorpus(obj, corpusText)
            % LOADOCOPRUS Updates simulation engines, active entropies, and plots.
            arguments
                obj
                corpusText char
            end
            
            % Stop playback prior to updating corpus
            if obj.IsPlaying
                obj.togglePlayback();
            end
            
            % Instantiate the math models
            obj.Generator = OceanMath.Information.Processes.ShannonTextGenerator(corpusText);
            obj.SearchEngine = OceanMath.Information.Processes.MonteCarloSearchEngine(corpusText);
            
            % Reset cumulative generator histories
            obj.GeneratedText = char.empty(1, 0);
            obj.ActiveContext = char.empty(1, 0);
            obj.RetroTerminal.Value = '';
            obj.LblActiveContext.Text = "Context: ''";
            
            % Redraw metrics and spectra
            obj.updateEntropyLabels();
            order = obj.getActiveOrder();
            obj.SpectrumRenderer.updateEntropyDecayCurve(obj.Generator.Entropies, order);
            
            % Set base unigram probabilities
            obj.SpectrumRenderer.updateProbabilitySpectrum(...
                obj.Generator.Alphabet, obj.Generator.UnigramProbs, '', '');
                
            % Render standard text block inside simulated book window
            N_corp = length(obj.Generator.Corpus);
            if N_corp > 0
                startIdx = 1;
                displayIndices = mod((startIdx : startIdx + 249) - 1, N_corp) + 1;
                displayText = obj.Generator.Corpus(displayIndices);
                obj.BookRenderer.renderSearchStep(displayText, [], [], 0, []);
            end
        end
        
        function runStep(obj)
            % RUNSTEP Executes a single character generation step (analytical or physical).
            if isempty(obj.Generator)
                return;
            end
            
            order = obj.getActiveOrder();
            
            % 1. Determine or initialize conditioning context
            if order == 2
                if isempty(obj.GeneratedText)
                    % Sample starting unigram context from corpus
                    idx = randi(length(obj.Generator.Corpus));
                    obj.GeneratedText = obj.Generator.Corpus(idx);
                end
                obj.ActiveContext = obj.GeneratedText(end);
            elseif order == 3
                if length(obj.GeneratedText) < 2
                    % Sample starting bigram context from corpus
                    idx = randi(length(obj.Generator.Corpus) - 1);
                    obj.GeneratedText = obj.Generator.Corpus(idx : idx + 1);
                end
                obj.ActiveContext = obj.GeneratedText(end-1:end);
            else
                obj.ActiveContext = char.empty(1, 0);
            end
            
            context = obj.ActiveContext;
            nextChar = ' ';
            usedMonteCarlo = false;
            searchLog = struct();
            
            % Check mode selection
            if contains(obj.ModeDropdown.Value, 'Monte Carlo')
                if order >= 2
                    % Attempt search in the physical book (Monte Carlo ancestral sampling)
                    try
                        [nextChar, searchLog] = obj.SearchEngine.searchNext(context);
                        usedMonteCarlo = true;
                    catch ME
                        % Backoff Context Sampling if target N-gram context isn't printed
                        if strcmp(ME.identifier, 'OceanMath:Information:MonteCarloSearchEngine:ContextNotFound')
                            if order == 3
                                % Trigram missing -> backoff to Bigram
                                try
                                    backoffContext = context(2);
                                    [nextChar, searchLog] = obj.SearchEngine.searchNext(backoffContext);
                                    usedMonteCarlo = true;
                                catch ME2
                                    % Bigram missing -> analytical fallback to unigram
                                    [nextChar, ~] = obj.Generator.generateNextAnalytical('', 1);
                                end
                            else
                                % Bigram missing -> analytical fallback to unigram
                                [nextChar, ~] = obj.Generator.generateNextAnalytical('', 1);
                            end
                        else
                            rethrow(ME);
                        end
                    end
                else
                    % For Zero-Order and First-Order, draw analytically
                    [nextChar, ~] = obj.Generator.generateNextAnalytical('', order);
                end
            else
                % Analytical Draw mode
                [nextChar, ~] = obj.Generator.generateNextAnalytical(context, order);
            end
            
            % 2. Append chosen letter and update terminal
            obj.GeneratedText(end+1) = nextChar;
            
            % Limit terminal buffer to keep rendering fluid
            terminalText = obj.GeneratedText;
            if length(terminalText) > 600
                terminalText = terminalText(end-599:end);
            end
            obj.RetroTerminal.Value = terminalText;
            
            % 3. Update context text labels
            if order == 2
                obj.ActiveContext = obj.GeneratedText(end);
            elseif order == 3
                obj.ActiveContext = obj.GeneratedText(end-1:end);
            else
                obj.ActiveContext = char.empty(1, 0);
            end
            obj.LblActiveContext.Text = "Context: '" + string(obj.ActiveContext) + "'";
            
            % 4. Render spectra updates
            [~, analyticalProbs] = obj.Generator.generateNextAnalytical(context, order);
            obj.SpectrumRenderer.updateProbabilitySpectrum(...
                obj.Generator.Alphabet, analyticalProbs, context, nextChar);
            obj.SpectrumRenderer.updateEntropyDecayCurve(obj.Generator.Entropies, order);
            
            % 5. Render Scan Animation
            if usedMonteCarlo && ~isempty(fields(searchLog))
                % Adjust scan animation speed
                animSpeed = max(3, obj.SpeedSlider.Value / 15.0);
                obj.BookRenderer.animateScan(obj.Generator.Corpus, searchLog, animSpeed, length(context));
            else
                % Analytical step: static draw of a random section of the book page
                N_corp = length(obj.Generator.Corpus);
                if N_corp > 0
                    startIdx = randi(max(1, N_corp - 250));
                    displayIndices = mod((startIdx : startIdx + 249) - 1, N_corp) + 1;
                    displayText = obj.Generator.Corpus(displayIndices);
                    obj.BookRenderer.renderSearchStep(displayText, [], [], 0, []);
                end
            end
        end
        
        function togglePlayback(obj)
            % TOGGLEPLAYBACK Starts/stops the sequential loop.
            if obj.IsPlaying
                stop(obj.PlaybackTimer);
                obj.IsPlaying = false;
                obj.BtnPlayPause.Text = 'Play';
                obj.BtnPlayPause.BackgroundColor = [0.15, 0.50, 0.80];
            else
                % Load updated slider speed period safely
                newPeriod = max(0.02, obj.SpeedSlider.Value / 1000.0);
                obj.PlaybackTimer.Period = newPeriod;
                
                start(obj.PlaybackTimer);
                obj.IsPlaying = true;
                obj.BtnPlayPause.Text = 'Pause';
                obj.BtnPlayPause.BackgroundColor = [0.80, 0.50, 0.15];
            end
        end
        
        function speedSliderCallback(obj)
            % SPEEDSLIDERCALLBACK Dynamically adjusts speed interval of timers.
            obj.SpeedLabel.Text = sprintf('Step Playback Interval: %d ms', round(obj.SpeedSlider.Value));
            
            newPeriod = max(0.02, obj.SpeedSlider.Value / 1000.0);
            if obj.IsPlaying
                % Timer must be briefly stopped to update period
                stop(obj.PlaybackTimer);
                obj.PlaybackTimer.Period = newPeriod;
                start(obj.PlaybackTimer);
            else
                obj.PlaybackTimer.Period = newPeriod;
            end
        end
        
        function corpusDropdownCallback(obj)
            % CORPUSDROPDOWNCALLBACK Loads the selected pre-loaded text files.
            cleaned = obj.loadBuiltInCorpus(obj.CorpusDropdown.Value);
            obj.loadCorpus(cleaned);
        end
        
        function changeOrderCallback(obj)
            % CHANGEORDERCALLBACK Updates layouts when Markov order switches.
            if isempty(obj.Generator)
                return;
            end
            
            order = obj.getActiveOrder();
            
            % Sync active context memory sizes to the new order
            if order == 2
                if length(obj.GeneratedText) >= 1
                    obj.ActiveContext = obj.GeneratedText(end);
                else
                    obj.ActiveContext = ' ';
                end
            elseif order == 3
                if length(obj.GeneratedText) >= 2
                    obj.ActiveContext = obj.GeneratedText(end-1:end);
                elseif length(obj.GeneratedText) == 1
                    obj.ActiveContext = [' ', obj.GeneratedText];
                else
                    obj.ActiveContext = '  ';
                end
            else
                obj.ActiveContext = char.empty(1, 0);
            end
            
            obj.LblActiveContext.Text = "Context: '" + string(obj.ActiveContext) + "'";
            obj.updateEntropyLabels();
            
            % Update spectrum plots
            obj.SpectrumRenderer.updateEntropyDecayCurve(obj.Generator.Entropies, order);
            [~, probs] = obj.Generator.generateNextAnalytical(obj.ActiveContext, order);
            obj.SpectrumRenderer.updateProbabilitySpectrum(...
                obj.Generator.Alphabet, probs, obj.ActiveContext, '');
        end
        
        function updateEntropyLabels(obj)
            % UPDATEENTROPYLABELS Updates right sidebar metrics text.
            if isempty(obj.Generator)
                return;
            end
            
            obj.LblH0.Text = sprintf('%.4f bits', obj.Generator.Entropies(1));
            obj.LblH1.Text = sprintf('%.4f bits', obj.Generator.Entropies(2));
            obj.LblH2.Text = sprintf('%.4f bits', obj.Generator.Entropies(3));
            obj.LblH3.Text = sprintf('%.4f bits', obj.Generator.Entropies(4));
            
            order = obj.getActiveOrder();
            activeEnt = obj.Generator.Entropies(order + 1);
            obj.LblActiveEntropyValue.Text = sprintf('%.3f bits', activeEnt);
            
            % Update badge state
            if order >= 1
                obj.EntropyDecayBadge.BackgroundColor = obj.COLOR_GREEN;
                obj.EntropyDecayBadge.Text = 'ENTROPY DECAY ACTIVE';
            else
                obj.EntropyDecayBadge.BackgroundColor = obj.COLOR_ORANGE;
                obj.EntropyDecayBadge.Text = 'MAXIMAL ENTROPY (CHAOS)';
            end
        end
        
        function loadCustomCorpus(obj)
            % LOADCUSTOMCORPUS Imports custom user text files via uigetfile.
            [file, path] = uigetfile('*.txt', 'Select Custom Text Corpus');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            filePath = fullfile(path, file);
            try
                cleaned = OceanMath.Information.Processes.CorpusSanitizer.sanitizeFile(filePath);
                if length(cleaned) < 10
                    uialert(obj.FigureHandle, ...
                        'Text corpus too short. Please select a corpus with at least 10 valid characters.', ...
                        'Sanitization Failure');
                    return;
                end
                obj.loadCorpus(cleaned);
            catch ME
                uialert(obj.FigureHandle, ...
                    "Failed to process target text file: " + ME.message, ...
                    'Load Error');
            end
        end
        
        function resetSandbox(obj)
            % RESETSANDBOX Resets current simulation states.
            if obj.IsPlaying
                obj.togglePlayback();
            end
            
            obj.GeneratedText = char.empty(1, 0);
            obj.ActiveContext = char.empty(1, 0);
            obj.RetroTerminal.Value = '';
            obj.LblActiveContext.Text = "Context: ''";
            
            obj.updateEntropyLabels();
            order = obj.getActiveOrder();
            obj.SpectrumRenderer.updateEntropyDecayCurve(obj.Generator.Entropies, order);
            
            % Reset probability spectrum
            [~, probs] = obj.Generator.generateNextAnalytical('', order);
            obj.SpectrumRenderer.updateProbabilitySpectrum(...
                obj.Generator.Alphabet, probs, '', '');
                
            % Reset Book axes
            N_corp = length(obj.Generator.Corpus);
            if N_corp > 0
                startIdx = 1;
                displayIndices = mod((startIdx : startIdx + 249) - 1, N_corp) + 1;
                displayText = obj.Generator.Corpus(displayIndices);
                obj.BookRenderer.renderSearchStep(displayText, [], [], 0, []);
            end
        end
        
        function order = getActiveOrder(obj)
            % GETACTIVEORDER Resolves active radio button selection to N-gram order.
            selectedBtn = obj.OrderButtonGroup.SelectedObject;
            if isempty(selectedBtn)
                order = 0;
            else
                order = selectedBtn.UserData;
            end
        end
        
        function closeApp(obj)
            % Graceful App closing callback. Cleans timers to prevent MATLAB leaks.
            if ~isempty(obj.PlaybackTimer) && isvalid(obj.PlaybackTimer)
                stop(obj.PlaybackTimer);
                delete(obj.PlaybackTimer);
                obj.PlaybackTimer = [];
            end
            
            if ~isempty(obj.FigureHandle) && isvalid(obj.FigureHandle)
                delete(obj.FigureHandle);
                obj.FigureHandle = [];
            end
        end
    end
end
