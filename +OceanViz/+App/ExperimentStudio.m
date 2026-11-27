classdef ExperimentStudio < handle
    % EXPERIMENTSTUDIO  Master Generic Interactive UI & Drag-Drop Inspector.
    %
    %   Part of the Model-Driven Dynamic Visualization Architecture.
    %   Allows users to pass any ExperimentResult object or MAT file, inspect
    %   all outputs (images, curves, scalars, matrices), select channels,
    %   override OceanViz strategies, and view interactive multi-panel plots.
    %
    %   Usage:
    %       % Inspect an ExperimentResult object
    %       OceanViz.App.ExperimentStudio.inspect(res);
    %
    %       % Launch blank studio and pick file
    %       studio = OceanViz.App.ExperimentStudio();
    %
    %   Author: Ocean Library  |  2026-08-05

    properties
        UIFigure          matlab.ui.Figure
        MainLayout        matlab.ui.container.GridLayout
        HeaderPanel       matlab.ui.container.Panel
        SidebarPanel      matlab.ui.container.Panel
        ViewportPanel     matlab.ui.container.Panel
        
        TitleLabel        matlab.ui.control.Label
        SubTitleLabel     matlab.ui.control.Label
        
        ChannelList       matlab.ui.control.ListBox
        StrategyDropDown  matlab.ui.control.DropDown
        LayoutDropDown    matlab.ui.control.DropDown
        ThemeButton       matlab.ui.control.Button
        LoadButton        matlab.ui.control.Button
        ExportButton      matlab.ui.control.Button
        
        ParamTextArea     matlab.ui.control.TextArea
        
        VizEngine         OceanViz.Core
        CurrentResult     OceanData.ExperimentResult
        CurrentAxes       matlab.ui.control.UIAxes = matlab.ui.control.UIAxes.empty
    end

    methods (Static)
        function studio = inspect(expResult)
            % INSPECT Connects to active open Studio window in-memory or creates one. Zero disk save.
            persistent activeStudio
            if ischar(expResult) || isstring(expResult)
                expResult = OceanData.ExperimentResult.load(expResult);
            end
            
            if isempty(activeStudio) || ~isvalid(activeStudio) || ~isvalid(activeStudio.UIFigure)
                activeStudio = OceanViz.App.ExperimentStudio(expResult);
            else
                activeStudio.loadResult(expResult);
            end
            studio = activeStudio;
        end
    end

    methods
        function obj = ExperimentStudio(expResult)
            % EXPERIMENTSTUDIO Construct the studio UI figure and components.
            arguments
                expResult = []
            end

            % Initialize OceanViz engine (Dark theme default)
            obj.VizEngine = OceanViz.Core('Theme', 'Dark');
            
            % Build UI
            obj.buildUI();
            
            % Load result if provided
            if ~isempty(expResult) && isa(expResult, 'OceanData.ExperimentResult')
                obj.loadResult(expResult);
            end
        end

        function loadResult(obj, expResult)
            % LOADRESULT Load and render an ExperimentResult instance.
            obj.CurrentResult = expResult;
            
            % Update Header
            obj.TitleLabel.Text = expResult.Title;
            obj.SubTitleLabel.Text = sprintf('%s  |  %s', expResult.Timestamp, expResult.Description);
            
            % Populate Channel Listbox
            names = expResult.listChannelNames();
            items = cell(1, numel(names));
            for i = 1:numel(names)
                ch = expResult.getChannel(names(i));
                items{i} = sprintf('[%s] %s', upper(ch.Type), ch.Name);
            end
            
            if isempty(items)
                obj.ChannelList.Items = {'(No Data Channels)'};
            else
                obj.ChannelList.Items = items;
                obj.ChannelList.Value = items{1};
            end
            
            % Populate Parameters & Metrics text box
            paramStr = "=== PARAMETERS ===" + newline;
            if ~isempty(fieldnames(expResult.Parameters))
                pFields = fieldnames(expResult.Parameters);
                for i = 1:numel(pFields)
                    paramStr = paramStr + sprintf('%s: %s\n', pFields{i}, string(expResult.Parameters.(pFields{i})));
                end
            else
                paramStr = paramStr + "(None)" + newline;
            end
            
            paramStr = paramStr + newline + "=== METRICS ===" + newline;
            if ~isempty(fieldnames(expResult.Metrics))
                mFields = fieldnames(expResult.Metrics);
                for i = 1:numel(mFields)
                    paramStr = paramStr + sprintf('%s: %.4g\n', mFields{i}, expResult.Metrics.(mFields{i}));
                end
            else
                paramStr = paramStr + "(None)" + newline;
            end
            
            obj.ParamTextArea.Value = paramStr;
            
            % Trigger render
            obj.renderViewport();
        end
    end

    methods (Access = private)
        function buildUI(obj)
            % BUILDUI Construct uifigure, layout grid, sidebar, and viewport panels.
            colors = obj.VizEngine.Colors;
            
            obj.UIFigure = uifigure('Name', 'OceanViz Experiment Studio', ...
                'Color', colors.Background, ...
                'Position', [100 100 1280 760]);
            
            try
                obj.UIFigure.WindowDropFcn = @(~, event) obj.onDropFile(event);
            catch
                % Graceful fallback for MATLAB releases without WindowDropFcn
            end
            
            % Master 2-row x 2-col grid
            obj.MainLayout = uigridlayout(obj.UIFigure, [2, 2]);
            obj.MainLayout.RowHeight = {55, '1x'};
            obj.MainLayout.ColumnWidth = {310, '1x'};
            obj.MainLayout.BackgroundColor = colors.Background;
            
            % ── 1. Header Panel (Top, spanning 2 columns) ──────────────────
            obj.HeaderPanel = uipanel(obj.MainLayout, ...
                'BackgroundColor', colors.Background, ...
                'BorderType', 'none');
            obj.HeaderPanel.Layout.Row = 1;
            obj.HeaderPanel.Layout.Column = [1, 2];
            
            headerGrid = uigridlayout(obj.HeaderPanel, [1, 5]);
            headerGrid.ColumnWidth = {'1x', 110, 110, 110, 110};
            headerGrid.Padding = [10 5 10 5];
            
            labelsGroup = uigridlayout(headerGrid, [2, 1]);
            labelsGroup.RowHeight = {'1x', '1x'};
            labelsGroup.Padding = [0 0 0 0];
            
            obj.TitleLabel = uilabel(labelsGroup, ...
                'Text', 'OceanViz Experiment Studio', ...
                'FontWeight', 'bold', 'FontSize', 16, ...
                'FontColor', colors.Text);
            
            obj.SubTitleLabel = uilabel(labelsGroup, ...
                'Text', 'Model-Driven Dynamic Visualization Architecture', ...
                'FontSize', 11, ...
                'FontColor', colors.Text * 0.7);
            
            obj.LoadButton = uibutton(headerGrid, 'push', ...
                'Text', 'Open File...', ...
                'BackgroundColor', colors.Primary * 0.8, ...
                'FontColor', [1 1 1], ...
                'ButtonPushedFcn', @(~,~) obj.onLoadFile());
            
            obj.ExportButton = uibutton(headerGrid, 'push', ...
                'Text', 'Export Image', ...
                'BackgroundColor', colors.Secondary * 0.8, ...
                'FontColor', [1 1 1], ...
                'ButtonPushedFcn', @(~,~) obj.onExportImage());
            
            obj.ThemeButton = uibutton(headerGrid, 'push', ...
                'Text', 'Theme: Dark', ...
                'BackgroundColor', colors.Background * 1.5, ...
                'FontColor', colors.Text, ...
                'ButtonPushedFcn', @(~,~) obj.onToggleTheme());
            
            % ── 2. Sidebar Panel (Left, Row 2) ─────────────────────────────
            obj.SidebarPanel = uipanel(obj.MainLayout, ...
                'Title', ' Data Inspector', ...
                'BackgroundColor', colors.Background, ...
                'ForegroundColor', colors.Text, ...
                'FontWeight', 'bold');
            obj.SidebarPanel.Layout.Row = 2;
            obj.SidebarPanel.Layout.Column = 1;
            
            sideGrid = uigridlayout(obj.SidebarPanel, [8, 1]);
            sideGrid.RowHeight = {22, 160, 22, 30, 22, 30, 22, '1x'};
            sideGrid.Padding = [10 10 10 10];
            
            uilabel(sideGrid, 'Text', 'Channels / Variables:', 'FontColor', colors.Text, 'FontWeight', 'bold');
            
            obj.ChannelList = uilistbox(sideGrid, ...
                'Items', {'(Empty)'}, ...
                'Multiselect', 'on', ...
                'BackgroundColor', colors.Background * 1.2, ...
                'FontColor', colors.Text, ...
                'ValueChangedFcn', @(~,~) obj.renderViewport());
            
            uilabel(sideGrid, 'Text', 'Strategy Override:', 'FontColor', colors.Text, 'FontWeight', 'bold');
            
            obj.StrategyDropDown = uidropdown(sideGrid, ...
                'Items', {'Auto', 'Image', 'TimeSeries', 'Regression', 'Distribution', 'PartialDependence2D'}, ...
                'Value', 'Auto', ...
                'BackgroundColor', colors.Background * 1.3, ...
                'FontColor', colors.Text, ...
                'ValueChangedFcn', @(~,~) obj.renderViewport());
            
            uilabel(sideGrid, 'Text', 'Viewport Layout:', 'FontColor', colors.Text, 'FontWeight', 'bold');
            
            obj.LayoutDropDown = uidropdown(sideGrid, ...
                'Items', {'Auto', '1x1 (Full)', '1x2 (Split)', '2x2 (Grid)'}, ...
                'Value', 'Auto', ...
                'BackgroundColor', colors.Background * 1.3, ...
                'FontColor', colors.Text, ...
                'ValueChangedFcn', @(~,~) obj.renderViewport());
            
            uilabel(sideGrid, 'Text', 'Metadata & Metrics:', 'FontColor', colors.Text, 'FontWeight', 'bold');
            
            obj.ParamTextArea = uitextarea(sideGrid, ...
                'Value', {'No experiment loaded.'}, ...
                'BackgroundColor', colors.Background * 1.1, ...
                'FontColor', colors.Text * 0.95, ...
                'FontName', 'Courier');
            
            % ── 3. Viewport Panel (Right, Row 2) ────────────────────────────
            obj.ViewportPanel = uipanel(obj.MainLayout, ...
                'Title', ' Visualization Viewport', ...
                'BackgroundColor', colors.Background, ...
                'ForegroundColor', colors.Text, ...
                'FontWeight', 'bold');
            obj.ViewportPanel.Layout.Row = 2;
            obj.ViewportPanel.Layout.Column = 2;
        end

        function renderViewport(obj)
            % RENDERVIEWPORT Render selected channels into viewport panel using OceanViz.Core.
            if isempty(obj.CurrentResult)
                return;
            end
            
            selectedItems = obj.ChannelList.Value;
            if isempty(selectedItems) || (iscell(selectedItems) && strcmp(selectedItems{1}, '(No Data Channels)'))
                return;
            end
            
            if ischar(selectedItems)
                selectedItems = {selectedItems};
            end
            
            % Extract channel names from items like "[IMAGE_GRAY] NoisyInput"
            numSelected = numel(selectedItems);
            selNames = cell(1, numSelected);
            for i = 1:numSelected
                tokens = regexp(selectedItems{i}, '\]\s*(.*)$', 'tokens');
                if ~isempty(tokens)
                    selNames{i} = tokens{1}{1};
                else
                    selNames{i} = selectedItems{i};
                end
            end
            
            % Determine grid layout
            layoutMode = obj.LayoutDropDown.Value;
            if strcmp(layoutMode, 'Auto')
                if numSelected == 1
                    gridDim = [1, 1];
                elseif numSelected == 2
                    gridDim = [1, 2];
                else
                    gridDim = [2, 2];
                end
            elseif strcmp(layoutMode, '1x1 (Full)')
                gridDim = [1, 1];
            elseif strcmp(layoutMode, '1x2 (Split)')
                gridDim = [1, 2];
            else
                gridDim = [2, 2];
            end
            
            % Clear existing viewport contents
            delete(obj.ViewportPanel.Children);
            
            % Create new grid layout inside ViewportPanel
            vpGrid = uigridlayout(obj.ViewportPanel, gridDim);
            vpGrid.Padding = [10 10 10 10];
            vpGrid.RowSpacing = 10;
            vpGrid.ColumnSpacing = 10;
            vpGrid.BackgroundColor = obj.VizEngine.Colors.Background;
            
            obj.CurrentAxes = matlab.ui.control.UIAxes.empty;
            
            % Limit to max tiles
            maxTiles = gridDim(1) * gridDim(2);
            numRender = min(numSelected, maxTiles);
            
            for k = 1:numRender
                ax = uiaxes(vpGrid);
                obj.CurrentAxes(end+1) = ax;
                
                ch = obj.CurrentResult.getChannel(selNames{k});
                
                % Determine strategy to use
                strat = obj.StrategyDropDown.Value;
                if strcmp(strat, 'Auto')
                    strat = ch.Strategy;
                end
                
                % Format data struct for strategy plot
                if strcmp(strat, 'Image')
                    dataStruct = struct('Image', ch.Data, 'Title', ch.Name);
                elseif strcmp(strat, 'TimeSeries')
                    dataStruct = struct('Y', ch.Data(:), 'Lags', min(20, numel(ch.Data)-1), 'ModelName', ch.Name);
                else
                    dataStruct = struct('Data', ch.Data, 'Title', ch.Name);
                end
                
                % Call OceanViz.Core plot method
                obj.VizEngine.plot(ax, strat, dataStruct, 'Title', ch.Name, 'XLabel', ch.XLabel, 'YLabel', ch.YLabel);
            end
        end

        function onLoadFile(obj)
            % ONLOADFILE Trigger open file dialog for MAT files.
            [filename, path] = uigetfile('*.mat', 'Select ExperimentResult MAT file');
            if isequal(filename, 0)
                return;
            end
            fullPath = fullfile(path, filename);
            try
                res = OceanData.ExperimentResult.load(fullPath);
                obj.loadResult(res);
            catch ME
                uialert(obj.UIFigure, ME.message, 'File Load Error');
            end
        end

        function onDropFile(obj, event)
            % ONDROPFILE Drag and drop listener when a .mat file is dropped onto Studio.
            try
                if isprop(event, 'Files') && ~isempty(event.Files)
                    filePath = event.Files{1};
                    [~, ~, ext] = fileparts(filePath);
                    if strcmpi(ext, '.mat')
                        res = OceanData.ExperimentResult.load(filePath);
                        obj.loadResult(res);
                    end
                end
            catch ME
                uialert(obj.UIFigure, ME.message, 'Drag & Drop Load Error');
            end
        end

        function onExportImage(obj)
            % ONEXPORTIMAGE Save current figure to PNG.
            [filename, path] = uiputfile('*.png', 'Export Viewport Image', 'experiment_studio_view.png');
            if isequal(filename, 0)
                return;
            end
            exportgraphics(obj.ViewportPanel, fullfile(path, filename));
        end

        function onToggleTheme(obj)
            % ONTOGGLETHEME Switch between Dark and Scientific themes.
            if obj.VizEngine.ThemeName == "Dark"
                obj.VizEngine = OceanViz.Core('Theme', 'Scientific');
                obj.ThemeButton.Text = 'Theme: Light';
            else
                obj.VizEngine = OceanViz.Core('Theme', 'Dark');
                obj.ThemeButton.Text = 'Theme: Dark';
            end
            colors = obj.VizEngine.Colors;
            obj.UIFigure.Color = colors.Background;
            obj.MainLayout.BackgroundColor = colors.Background;
            obj.HeaderPanel.BackgroundColor = colors.Background;
            obj.SidebarPanel.BackgroundColor = colors.Background;
            obj.SidebarPanel.ForegroundColor = colors.Text;
            obj.ViewportPanel.BackgroundColor = colors.Background;
            obj.ViewportPanel.ForegroundColor = colors.Text;
            obj.TitleLabel.FontColor = colors.Text;
            
            if ~isempty(obj.CurrentResult)
                obj.renderViewport();
            end
        end
    end
end
