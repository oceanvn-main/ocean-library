classdef Shell < handle
    % SHELL  The Model Lab window: a top bar (model + input image sources),
    %   one tab per panel that SUPPORTS the loaded model (capability check),
    %   and a status line. Tabs are rebuilt when the model changes.
    %
    %   Launch with OceanViz.App.ModelLab.launch(...). Scriptable (tests, reports):
    %     lab.loadModel(h)            an IModelHandle, a built-in name, or a ModelCard path
    %     p = lab.panel("Forward process")   the panel object behind a tab
    %     lab.Context.useDatasetImage(3)

    properties (SetAccess = private)
        Figure
        Context
        Panels = {}             % panel objects of the current tabs
    end

    properties (Access = private)
        TabGroup
        StatusLabel
        ModelLabel
        BuiltinDropdown
        Listeners = {}
    end

    methods
        function obj = Shell(opts)
            arguments
                opts.Visible (1,1) logical = true
                opts.Theme (1,1) string = "Dark"
            end
            obj.Context = OceanViz.App.ModelLab.Context(opts.Theme);
            c = obj.Context.Theme;
            scr = get(0, 'ScreenSize'); W = 1320; H = 840;
            obj.Figure = uifigure('Name', 'Model Lab - Ocean Library', 'Color', c.Background, ...
                'Position', [max(0, (scr(3) - W) / 2), max(0, (scr(4) - H) / 2), W, H], ...
                'Visible', matlab.lang.OnOffSwitchState(opts.Visible));
            obj.Context.Figure = obj.Figure;
            root = uigridlayout(obj.Figure, [3 1], 'RowHeight', {44, '1x', 24}, 'Padding', [0 0 0 0], ...
                'RowSpacing', 0, 'BackgroundColor', c.Background);
            obj.buildTopBar(root);
            obj.TabGroup = uitabgroup(root);
            obj.TabGroup.Layout.Row = 2;
            obj.StatusLabel = uilabel(root, 'Text', ' Load a model to begin.', 'FontColor', c.Text, ...
                'BackgroundColor', c.Panel);
            obj.StatusLabel.Layout.Row = 3;
            obj.Context.StatusFcn = @(msg) obj.setStatus(msg);
            obj.Listeners = {addlistener(obj.Context, 'ModelChanged', @(~, ~) obj.rebuildTabs()), ...
                             addlistener(obj.Context, 'ImageChanged', @(~, ~) obj.broadcastImage())};
        end

        function loadModel(obj, source)
            % source: IModelHandle | built-in name | ModelCard .mat path
            try
                if isa(source, 'OceanViz.App.ModelLab.Models.IModelHandle')
                    h = source;
                elseif any(OceanViz.App.ModelLab.Models.ModelFactory.builtins() == string(source))
                    obj.setStatus("Preparing " + string(source) + " ...");
                    h = OceanViz.App.ModelLab.Models.ModelFactory.builtin(source);
                else
                    h = OceanViz.App.ModelLab.Models.ModelFactory.fromCard(char(source));
                end
                obj.Context.setModel(h);
                obj.ModelLabel.Text = "Model: " + h.Name;
                obj.setStatus(sprintf('Loaded %s  |  capabilities: %s', h.Name, strjoin(h.Capabilities, ', ')));
            catch err
                obj.setStatus(['Could not load model: ' err.message]);
                if ~strcmp(obj.Figure.Visible, 'on'), rethrow(err); end
            end
        end

        function p = panel(obj, titleText)
            p = [];
            for k = 1:numel(obj.Panels)
                if obj.Panels{k}.title() == string(titleText), p = obj.Panels{k}; return; end
            end
        end

        function t = tabTitles(obj)
            t = cellfun(@(p) p.title(), obj.Panels);
        end

        function setStatus(obj, msg)
            if isvalid(obj.StatusLabel), obj.StatusLabel.Text = [' ' char(msg)]; end
        end

        function close(obj)
            cellfun(@delete, obj.Listeners);
            if isvalid(obj.Figure), delete(obj.Figure); end
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Access = private)
        function buildTopBar(obj, root)
            c = obj.Context.Theme;
            bar = uigridlayout(root, [1 8], 'ColumnWidth', {150, 250, 130, 110, 120, 110, '1x', 10}, ...
                'Padding', [8 8 8 8], 'ColumnSpacing', 6, 'BackgroundColor', c.Panel);
            bar.Layout.Row = 1;
            uibutton(bar, 'Text', 'Load model card...', 'ButtonPushedFcn', @(~, ~) obj.onLoadCard());
            obj.BuiltinDropdown = uidropdown(bar, 'Items', ["Built-in models...", ...
                OceanViz.App.ModelLab.Models.ModelFactory.builtins()], 'ValueChangedFcn', @(dd, ~) obj.onBuiltin(dd));
            uibutton(bar, 'Text', 'Legacy CNN (.json)...', 'ButtonPushedFcn', @(~, ~) obj.onLoadLegacy());
            uibutton(bar, 'Text', 'Load image...', 'ButtonPushedFcn', @(~, ~) obj.onLoadImage());
            uibutton(bar, 'Text', 'Dataset image', 'ButtonPushedFcn', @(~, ~) obj.Context.useDatasetImage());
            uibutton(bar, 'Text', 'Random image', 'ButtonPushedFcn', @(~, ~) obj.onRandomImage());
            obj.ModelLabel = uilabel(bar, 'Text', 'Model: (none)', 'FontColor', c.Text, 'FontWeight', 'bold');
        end

        function rebuildTabs(obj)
            delete(obj.TabGroup.Children);
            obj.Panels = {};
            model = obj.Context.Model;
            c = obj.Context.Theme;
            for ctor = OceanViz.App.ModelLab.PanelRegistry.all()
                p = ctor{1}();
                if ~p.supports(model), continue; end
                tab = uitab(obj.TabGroup, 'Title', p.title(), 'BackgroundColor', c.Background, ...
                    'Tooltip', char(p.reference()));
                p.build(tab, obj.Context);
                obj.Panels{end+1} = p;
            end
        end

        function broadcastImage(obj)
            for k = 1:numel(obj.Panels), obj.Panels{k}.onImageChanged(obj.Context); end
        end

        function onBuiltin(obj, dd)
            if dd.Value ~= dd.Items(1)
                obj.loadModel(dd.Value);
                dd.Value = dd.Items{1};
            end
        end

        function onLoadCard(obj)
            [f, p] = uigetfile({'*.mat', 'Model card (*.mat)'}, 'Select a model card');
            if isequal(f, 0), return; end
            obj.loadModel(fullfile(p, f));
        end

        function onLoadLegacy(obj)
            [f, p] = uigetfile({'*.json', 'Legacy CNN checkpoint (*.json)'}, 'Select checkpoint .json');
            if isequal(f, 0), return; end
            [~, base] = fileparts(f);
            obj.loadModel(OceanViz.App.ModelLab.Models.ModelFactory.legacyCNN(fullfile(p, base)));
        end

        function onLoadImage(obj)
            [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Images'}, 'Select an image');
            if isequal(f, 0), return; end
            obj.Context.setImage(imread(fullfile(p, f)));
            obj.setStatus(sprintf('Input image: %s', f));
        end

        function onRandomImage(obj)
            if isempty(obj.Context.Model), return; end
            obj.Context.setImage(rand(obj.Context.Model.ImageSize));
        end
    end
end
