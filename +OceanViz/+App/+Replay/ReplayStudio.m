classdef ReplayStudio < handle
    % REPLAYSTUDIO  Explore a recorded computation without the model
    %   (OceanReplay; Planning/System/Replay). Any recording opens: tiles are
    %   drawn by kind (Renderers.Registry); a domain layout (Diffusion,
    %   Clustering) adds its equations and composite plots when it applies.
    %
    %   Toolbar   Open... (file picker) and Recent (recordings opened before, remembered
    %             across sessions) -- a new recording replaces the current one in this window;
    %             first / previous / PLAY-PAUSE / next / last, speed, layout,
    %             "Details" (opens the side drawer). Keys: Space = play/pause,
    %             Left / Right = previous / next stored point.
    %   Left      recording summary; one slider per axis (snaps to real axis values)
    %   Tiles     cards: plain-language title, the NOTATION (symbol) of what is
    %             shown, the plot with its legend / colour bar, and where the value
    %             was stored ("stored at step 500 - nearest stored to 507"; never
    %             blended). (i) opens that tile's full explanation (including whether a
    %             view is in the textbook, how it was made and its reference).
    %   Timeline  one scalar's full history (x in play order), a cursor labelled with the
    %             current step, and a separate strip marking where the tiles' data were
    %             stored; hover shows values; click to jump.
    %   Drawer    hidden by default (progressive disclosure): the layout's notes
    %             (e.g. Eq. 2.2.4 numbers), every description, HOW IT WAS MADE,
    %             references and provenance.
    %
    %   Example:
    %     s = OceanViz.App.Replay.ReplayStudio("ddpm.h5");
    %     s.setCoord("step", 250);  s.play();  s.pause();
    %     s = OceanViz.App.Replay.ReplayStudio();           % file picker
    %     s.loadFile("seed2_small.h5");                       % switch recording in the same window

    properties (SetAccess = private)
        Recording
        Figure
        Theme
        Coords = struct()
        Layout
        Layouts = {}
        Tiles = struct('Card', {}, 'Axes', {}, 'Title', {}, 'Notation', {}, 'Note', {}, 'InfoButton', {}, 'Caption', {}, 'Spec', {})
        AxisControls = struct('Name', {}, 'Slider', {}, 'Field', {})
        LayoutDropdown
        RecentDropdown
        SpeedDropdown
        PlayButton
        DetailsButton
        TimelineDropdown
        TimelineAxes
        TimelineHint                    % one-line key of the timeline (header, right)
        RugAxes                         % thin strip under the timeline: where the tiles' data were stored
        DrawerTitle
        DrawerText
        PlayAxis (1,1) string = ""
        Frames (1,:) double = []       % stored points the player visits (run order)
        Playing (1,1) logical = false
    end

    properties (Access = private)
        Body
        TileGrid
        Timer = []
        PlayFps (1,1) double = 5
        PlayStart (1,1) double = 0       % frame index when Play was pressed
        PlayClock = []                   % tic at Play
        Busy (1,1) logical = false
        DrawerTile (1,1) double = 0     % 0 = all details, k = tile k
        Remember (1,1) logical = true   % keep the Recent list in MATLAB preferences
    end

    properties (Constant, Access = private)
        DrawerWidth = 380
        Speeds = [2 5 10 20 30 60 120 240]   % stored points shown per second of wall-clock time
        PrefGroup = 'OceanReplay'       % preferences group for the Recent list
        MaxRecent = 10
    end

    methods
        function obj = ReplayStudio(file, opts)
            % ReplayStudio(file) opens a recording; ReplayStudio() asks for one (file picker).
            arguments
                file (1,1) string = ""
                opts.Visible (1,1) logical = true
                opts.Layout (1,1) string = "auto"
                opts.Theme (1,1) string = "Dark"
                opts.Remember (1,1) logical = true   % add opened files to the Recent list (preferences)
            end
            if strlength(file) == 0
                file = OceanViz.App.Replay.ReplayStudio.pickFile();
                if strlength(file) == 0
                    error('OceanViz:Replay:noFile', 'No recording chosen.');
                end
            end
            if ~isfile(file), error('OceanViz:Replay:noFile', 'Recording "%s" not found.', file); end
            obj.Remember = opts.Remember;
            c = OceanViz.Core('Theme', char(opts.Theme)).Colors;
            c.Panel = min(1, c.Background + 0.05); c.Card = min(1, c.Background + 0.035); c.Accent = c.Primary;
            obj.Theme = c;
            obj.Figure = uifigure('Name', 'Replay Studio', 'Position', [60 60 1500 880], 'Color', c.Background, ...
                'Visible', matlab.lang.OnOffSwitchState(opts.Visible), 'KeyPressFcn', @(~, e) obj.onKey(e), ...
                'CloseRequestFcn', @(~, ~) obj.close());
            obj.loadFile(file, 'Layout', opts.Layout);
        end

        function loadFile(obj, file, opts)
            % Show another recording in this window (the whole view is rebuilt for it).
            arguments
                obj
                file (1,1) string
                opts.Layout (1,1) string = "auto"
            end
            if ~isfile(file), error('OceanViz:Replay:noFile', 'Recording "%s" not found.', file); end
            rec = OceanReplay.Core.Recording(file);                   % read first: a bad file keeps the old view
            obj.pause();
            obj.Recording = rec;
            obj.Coords = struct();
            for a = rec.Axes
                if ~isempty(a.Values), obj.Coords.(a.Name) = a.Values(1); end
            end
            obj.Layouts = {OceanViz.App.Replay.Layouts.DiffusionLayout(), OceanViz.App.Replay.Layouts.ClusteringLayout(), OceanViz.App.Replay.Layouts.VAELayout(), OceanViz.App.Replay.Layouts.GridWorldLayout(), OceanViz.App.Replay.Layouts.ParityLearningLayout(), ...
                           OceanViz.App.Replay.Layouts.DefaultLayout()};
            obj.Layouts = obj.Layouts(cellfun(@(l) l.appliesTo(rec), obj.Layouts));
            obj.AxisControls = struct('Name', {}, 'Slider', {}, 'Field', {});
            obj.Tiles = struct('Card', {}, 'Axes', {}, 'Title', {}, 'Notation', {}, 'Note', {}, 'InfoButton', {}, 'Caption', {}, 'Spec', {});
            obj.DrawerTile = 0;
            delete(obj.Figure.Children);
            title = "Replay Studio";
            if isfield(rec.Meta, 'title'), title = title + " -- " + string(rec.Meta.title); end
            obj.Figure.Name = char(title);
            obj.remember(file);
            obj.buildContents();
            obj.setLayout(obj.pickLayout(opts.Layout));
        end

        function openDialog(obj)
            % Toolbar "Open...": pick a recording and show it in this window.
            f = OceanViz.App.Replay.ReplayStudio.pickFile(fileparts(obj.Recording.File));
            if strlength(f) > 0, obj.loadFile(f); end
        end

        function files = recentFiles(obj)
            % Recordings opened before (most recent first; only files that still exist).
            files = strings(1, 0);
            if ~obj.Remember || ~ispref(obj.PrefGroup, 'RecentFiles'), return; end
            files = string(getpref(obj.PrefGroup, 'RecentFiles'));
            files = files(arrayfun(@isfile, files));
        end

        % ------------------------------------------------------------ position
        function setCoord(obj, axisName, value)
            % Move an axis to the axis value nearest `value` (tiles show their nearest STORED point).
            arguments
                obj
                axisName (1,1) string
                value (1,1) double
            end
            vals = obj.Recording.axis(axisName).Values;
            [~, k] = min(abs(vals - value));
            obj.Coords.(axisName) = vals(k);
            j = find([obj.AxisControls.Name] == axisName, 1);
            if ~isempty(j)
                obj.AxisControls(j).Slider.Value = vals(k);
                obj.AxisControls(j).Field.Value = vals(k);
            end
            obj.refresh();
        end

        function moved = stepForward(obj)
            % Next stored point of the player axis (run order). false at the end.
            moved = obj.jump(+1);
        end

        function moved = stepBack(obj)
            moved = obj.jump(-1);
        end

        function first(obj)
            if ~isempty(obj.Frames), obj.setCoord(obj.PlayAxis, obj.Frames(1)); end
        end

        function last(obj)
            if ~isempty(obj.Frames), obj.setCoord(obj.PlayAxis, obj.Frames(end)); end
        end

        % ------------------------------------------------------------ playback
        function play(obj)
            % Step through the stored points in the background (timer); stops at the end.
            if obj.Playing || isempty(obj.Frames), return; end
            if obj.frameIndex() >= numel(obj.Frames), obj.first(); end    % at the end: start over
            k = find(string(obj.SpeedDropdown.Items) == string(obj.SpeedDropdown.Value), 1);
            fps = obj.Speeds(max(1, k));
            % Wall-clock pacing: fps = stored points per SECOND. The timer only polls (at most 20 Hz); each poll shows the
            % point the clock has reached, skipping points when drawing is slower than the rate, so the run always finishes
            % in numel(Frames)/fps seconds instead of falling behind.
            obj.PlayFps = fps; obj.PlayStart = obj.frameIndex(); obj.PlayClock = tic;
            obj.Timer = timer('ExecutionMode', 'fixedSpacing', 'Period', round(min(1 / fps, 0.05), 3), ...
                'BusyMode', 'drop', 'TimerFcn', @(~, ~) obj.tick(), 'Name', 'ReplayStudioPlayer');
            obj.Playing = true;
            obj.PlayButton.Text = char("⏸  Pause");
            start(obj.Timer);
        end

        function pause(obj)
            obj.stopTimer();
            obj.Playing = false;
            if ~isempty(obj.PlayButton) && isvalid(obj.PlayButton), obj.PlayButton.Text = char("▶  Play"); end
        end

        function togglePlay(obj)
            if obj.Playing, obj.pause(); else, obj.play(); end
        end

        % ------------------------------------------------------------ layout / details
        function setLayout(obj, name)
            arguments
                obj
                name (1,1) string
            end
            k = find(cellfun(@(l) l.name() == name, obj.Layouts), 1);
            if isempty(k)
                error('OceanViz:Replay:layout', 'Layout "%s" does not apply to this recording (available: %s).', ...
                    name, strjoin(cellfun(@(l) l.name(), obj.Layouts), ', '));
            end
            obj.Layout = obj.Layouts{k};
            obj.LayoutDropdown.Value = char(name);
            pref = obj.Layout.timeline(obj.Recording);
            if any(string(obj.TimelineDropdown.Items) == pref), obj.TimelineDropdown.Value = char(pref); end
            obj.buildTiles();
            obj.choosePlayer();
            obj.refresh();
        end

        function showDetails(obj, show)
            % Open (true) / close (false) the details drawer with ALL notes.
            if nargin < 2, show = ~obj.drawerOpen(); end
            obj.DrawerTile = 0;
            obj.setDrawer(show);
        end

        function showTileInfo(obj, k)
            % Open the drawer on tile k's explanation.
            obj.DrawerTile = k;
            obj.setDrawer(true);
        end

        function caption = drawChannel(obj, ax, name)
            % Draw the nearest stored value of a channel/view; return its short caption.
            rec = obj.Recording; info = rec.info(name);
            c = OceanViz.App.Replay.Renderers.PointsRenderer.coordsFor(info, obj.Coords);
            if numel(fieldnames(c)) ~= numel(info.Axes)
                cla(ax); caption = 'no current coordinate on its axes'; return;
            end
            [value, at] = rec.get(name, c);
            ctx = struct('Theme', obj.Theme, 'Recording', rec, 'Coords', obj.mergeCoords(at));
            OceanViz.App.Replay.Renderers.Registry.forInfo(info).draw(ax, value, info, ctx);
            caption = obj.whereText(at, c);
        end

        function refresh(obj)
            if obj.Busy || isempty(obj.Figure) || ~isvalid(obj.Figure), return; end
            obj.Busy = true; c = onCleanup(@() obj.clearBusy());
            for k = 1:numel(obj.Tiles)
                t = obj.Tiles(k);
                try
                    if isempty(t.Spec.Draw)
                        cap = obj.drawChannel(t.Axes, t.Spec.Names(1));
                    else
                        cap = t.Spec.Draw(t.Axes, obj);
                    end
                catch err
                    cla(t.Axes); cap = ['could not draw: ' err.message];
                end
                t.Caption.Text = cap;
            end
            obj.drawTimeline();
            if obj.drawerOpen(), obj.fillDrawer(); end
            drawnow limitrate;
        end

        function close(obj)
            obj.stopTimer();
            if ~isempty(obj.Figure) && isvalid(obj.Figure), delete(obj.Figure); end
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Static)
        function f = pickFile(startFolder)
            % File picker for Replay recordings ("" when cancelled).
            if nargin < 1 || strlength(string(startFolder)) == 0, startFolder = pwd; end
            [n, d] = uigetfile({'*.h5', 'Replay recordings (*.h5)'}, 'Open a recording', char(startFolder));
            if isequal(n, 0), f = ""; else, f = string(fullfile(d, n)); end
        end

        function lim = padded(y)
            % y limits with 8% headroom (flat or empty series stay readable)
            y = y(isfinite(y));
            if isempty(y), lim = [0 1]; return; end
            lo = min(y); hi = max(y); d = hi - lo; if d == 0, d = max(abs(hi), 1); end
            lim = [lo - 0.08 * d, hi + 0.08 * d];
        end

        function s = open(file)
            % Open a recording (file picker when no file is given; [] when cancelled).
            if nargin < 1
                file = OceanViz.App.Replay.ReplayStudio.pickFile();
                if strlength(file) == 0, s = []; return; end
            end
            s = OceanViz.App.Replay.ReplayStudio(string(file));
        end
    end

    methods (Access = private)
        function clearBusy(obj), obj.Busy = false; end

        % ------------------------------------------------------------ window
        function buildContents(obj)
            T = obj.Theme;
            root = uigridlayout(obj.Figure, [2 1], 'RowHeight', {46, '1x'}, 'Padding', [0 0 0 0], ...
                'RowSpacing', 0, 'BackgroundColor', T.Background);
            obj.buildToolbar(root);
            obj.Body = uigridlayout(root, [1 3], 'ColumnWidth', {240, '1x', 0}, 'Padding', [0 0 0 0], ...
                'ColumnSpacing', 0, 'BackgroundColor', T.Background);
            obj.Body.Layout.Row = 2;
            obj.buildSidebar(obj.Body);
            mid = uigridlayout(obj.Body, [2 1], 'RowHeight', {'1x', 235}, 'Padding', [8 8 8 8], ...
                'RowSpacing', 8, 'BackgroundColor', T.Background);
            mid.Layout.Column = 2;
            obj.TileGrid = uigridlayout(mid, [1 1], 'Padding', [0 0 0 0], 'RowSpacing', 8, 'ColumnSpacing', 8, ...
                'BackgroundColor', T.Background);
            obj.TileGrid.Layout.Row = 1;
            obj.buildTimeline(mid);
            obj.buildDrawer(obj.Body);
        end

        function buildToolbar(obj, root)
            T = obj.Theme;
            tb = uigridlayout(root, [1 12], 'ColumnWidth', {40, 40, 96, 40, 40, 90, 90, 240, '1x', 60, 150, 110}, ...
                'Padding', [10 7 10 7], 'ColumnSpacing', 6, 'BackgroundColor', T.Panel);
            tb.Layout.Row = 1;
            b = @(txt, tip, fn) uibutton(tb, 'Text', txt, 'Tooltip', tip, 'ButtonPushedFcn', @(~, ~) fn(), ...
                'BackgroundColor', T.Card, 'FontColor', T.Text, 'FontSize', 13);
            b(char("⏮"), 'First stored point', @() obj.first());
            b(char("⏴"), 'Previous stored point (Left arrow)', @() obj.stepBack());
            obj.PlayButton = b(char("▶  Play"), 'Play / pause (Space)', @() obj.togglePlay());
            obj.PlayButton.BackgroundColor = T.Accent; obj.PlayButton.FontColor = [1 1 1]; obj.PlayButton.FontWeight = 'bold';
            b(char("⏵"), 'Next stored point (Right arrow)', @() obj.stepForward());
            b(char("⏭"), 'Last stored point', @() obj.last());
            obj.SpeedDropdown = uidropdown(tb, 'Items', cellstr(compose("%d fps", obj.Speeds)), 'Value', '5 fps', ...
                'Tooltip', 'Playback speed (frames per second)', 'ValueChangedFcn', @(~, ~) obj.restartIfPlaying());
            obj.SpeedDropdown.Layout.Column = 6;
            ob = uibutton(tb, 'Text', char("📂  Open..."), 'Tooltip', 'Open another recording (.h5) in this window', ...
                'ButtonPushedFcn', @(~, ~) obj.openDialog(), 'BackgroundColor', T.Card, 'FontColor', T.Text);
            ob.Layout.Column = 7;
            rf = obj.recentFiles();
            [~, nm, ex] = fileparts(rf); labels = cellstr(nm + ex);
            obj.RecentDropdown = uidropdown(tb, 'Items', [{'Recent recordings...'}, labels], ...
                'ItemsData', [{''}, cellstr(rf)], 'Tooltip', 'Recordings opened before (click one to switch)', ...
                'ValueChangedFcn', @(d, ~) obj.onRecent(d));
            obj.RecentDropdown.Layout.Column = 8;
            lab = uilabel(tb, 'Text', 'Layout', 'FontColor', T.Text, 'HorizontalAlignment', 'right');
            lab.Layout.Column = 10;
            names = cellfun(@(l) char(l.name()), obj.Layouts, 'UniformOutput', false);
            obj.LayoutDropdown = uidropdown(tb, 'Items', names, 'ValueChangedFcn', @(d, ~) obj.setLayout(string(d.Value)));
            obj.LayoutDropdown.Layout.Column = 11;
            obj.DetailsButton = uibutton(tb, 'state', 'Text', char("ⓘ  Details"), 'Tooltip', ...
                'Show the equations, what every tile means, how views were made, references', ...
                'ValueChangedFcn', @(h, ~) obj.showDetails(h.Value), 'BackgroundColor', T.Card, 'FontColor', T.Text);
            obj.DetailsButton.Layout.Column = 12;
        end

        function buildSidebar(obj, body)
            T = obj.Theme; rec = obj.Recording;
            sb = OceanViz.App.Common.Sidebar(body, struct('Theme', T));
            sb.Grid.Layout.Column = 1;
            sb.heading('Recording');
            sb.label(char(obj.summary()));
            sb.heading('Position');
            for a = rec.Axes
                if isempty(a.Values), continue; end
                lim = [min(a.Values) max(a.Values)]; if lim(1) == lim(2), lim(2) = lim(1) + 1; end
                name = a.Name;
                s = sb.slider(sprintf('%s   (%s)', a.Name, a.Symbol), lim, a.Values(1), @(h, ~) obj.setCoord(name, h.Value));
                s.MajorTicks = unique(round(linspace(lim(1), lim(2), 5), 4, 'significant'));
                s.MinorTicks = [];
                f = sb.add(@(gg) uieditfield(gg, 'numeric', 'Value', a.Values(1), 'Limits', lim, ...
                    'ValueChangedFcn', @(h, ~) obj.setCoord(name, h.Value)));
                obj.AxisControls(end+1) = struct('Name', name, 'Slider', s, 'Field', f);
            end
            sb.label('Tiles show the nearest STORED point; values are never blended. Space = play, arrows = step.');
        end

        function buildTimeline(obj, mid)
            T = obj.Theme;
            card = uipanel(mid, 'BorderType', 'none', 'BackgroundColor', T.Card);
            card.Layout.Row = 2;
            g = uigridlayout(card, [3 3], 'RowHeight', {24, '1x', 40}, 'ColumnWidth', {70, 220, '1x'}, ...
                'Padding', [8 6 12 2], 'RowSpacing', 0, 'BackgroundColor', T.Card);
            uilabel(g, 'Text', 'Timeline', 'FontWeight', 'bold', 'FontColor', T.Text);
            items = cellstr(obj.timelineChannels()); if isempty(items), items = {'(no scalar channel)'}; end
            obj.TimelineDropdown = uidropdown(g, 'Items', items, 'Tooltip', 'Which scalar to plot over the axis', ...
                'ValueChangedFcn', @(~, ~) obj.drawTimeline());
            obj.TimelineHint = uilabel(g, 'Text', '', 'FontColor', T.Text * 0.7 + T.Card * 0.3, 'FontSize', 10, ...
                'HorizontalAlignment', 'right');
            obj.TimelineAxes = uiaxes(g);
            obj.TimelineAxes.Layout.Row = 2; obj.TimelineAxes.Layout.Column = [1 3];
            obj.RugAxes = uiaxes(g);
            obj.RugAxes.Layout.Row = 3; obj.RugAxes.Layout.Column = [1 3];
        end

        function buildDrawer(obj, body)
            T = obj.Theme;
            d = uigridlayout(body, [2 1], 'RowHeight', {30, '1x'}, 'Padding', [10 8 10 8], 'BackgroundColor', T.Panel);
            d.Layout.Column = 3;
            h = uigridlayout(d, [1 2], 'ColumnWidth', {'1x', 30}, 'Padding', [0 0 0 0], 'BackgroundColor', T.Panel);
            obj.DrawerTitle = uilabel(h, 'Text', 'Details', 'FontWeight', 'bold', 'FontColor', T.Accent, 'FontSize', 13);
            uibutton(h, 'Text', char("✕"), 'Tooltip', 'Close', 'ButtonPushedFcn', @(~, ~) obj.showDetails(false), ...
                'BackgroundColor', T.Panel, 'FontColor', T.Text);
            obj.DrawerText = uitextarea(d, 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 11, ...
                'BackgroundColor', T.Card, 'FontColor', T.Text);
        end

        function buildTiles(obj)
            delete(obj.TileGrid.Children);
            T = obj.Theme; rec = obj.Recording;
            specs = obj.Layout.tiles(rec);
            n = numel(specs); cols = max(1, ceil(sqrt(n))); rows = max(1, ceil(n / cols));
            obj.TileGrid.RowHeight = repmat({'1x'}, 1, rows);
            obj.TileGrid.ColumnWidth = repmat({'1x'}, 1, cols);
            obj.Tiles = struct('Card', {}, 'Axes', {}, 'Title', {}, 'Notation', {}, 'Note', {}, 'InfoButton', {}, 'Caption', {}, 'Spec', {});
            for k = 1:n
                sp = specs(k);
                card = uipanel(obj.TileGrid, 'BorderType', 'none', 'BackgroundColor', T.Card);
                card.Layout.Row = ceil(k / cols); card.Layout.Column = mod(k - 1, cols) + 1;
                g = uigridlayout(card, [5 1], 'RowHeight', {22, 20, '1x', 18, 0}, 'Padding', [8 6 8 6], ...
                    'RowSpacing', 2, 'BackgroundColor', T.Card);
                head = uigridlayout(g, [1 2], 'ColumnWidth', {'1x', 24}, 'Padding', [0 0 0 0], ...
                    'ColumnSpacing', 4, 'BackgroundColor', T.Card);
                ttl = uilabel(head, 'Text', char(sp.Title), 'FontWeight', 'bold', 'FontColor', T.Text, 'FontSize', 12);
                kk = k;
                infoBtn = uibutton(head, 'Text', char("ⓘ"), 'Tooltip', 'What is this? How was it made?', ...
                    'ButtonPushedFcn', @(~, ~) obj.showTileInfo(kk), 'BackgroundColor', T.Card, 'FontColor', T.Accent, ...
                    'FontSize', 13);
                notation = uilabel(g, 'Text', obj.notationOf(sp), 'FontColor', T.Accent, 'FontSize', 13, ...
                    'Interpreter', 'latex');
                ax = uiaxes(g); ax.Toolbar.Visible = 'off'; disableDefaultInteractivity(ax);
                cap = uilabel(g, 'Text', '', 'FontColor', T.Text * 0.8 + T.Background * 0.2, 'FontSize', 10);
                note = uilabel(g, 'Text', '', 'FontColor', T.Text * 0.65 + T.Background * 0.35, 'FontSize', 10, ...
                    'FontAngle', 'italic', 'WordWrap', 'on');
                % (whether a view is in the textbook is stated in its (i) details, not on the card)
                obj.Tiles(end+1) = struct('Card', card, 'Axes', ax, 'Title', ttl, 'Notation', notation, 'Note', note, ...
                    'InfoButton', infoBtn, 'Caption', cap, 'Spec', sp);
            end
        end

        % ------------------------------------------------------------ player
        function choosePlayer(obj)
            % Player axis: the first axis a tile channel depends on; frames: the points where
            % the tiles' channels were stored (so every frame shows new data), in run order.
            rec = obj.Recording; axisName = ""; stored = [];
            for t = obj.Tiles
                for nm = t.Spec.Names
                    info = rec.info(nm);
                    if isempty(info.Axes), continue; end
                    if axisName == "", axisName = info.Axes(1); end
                    if info.Axes(1) == axisName
                        s = rec.stored(nm); stored = [stored; s.Values(:, 1)]; %#ok<AGROW>
                    end
                end
            end
            if axisName == "" && ~isempty(rec.Axes), axisName = rec.Axes(1).Name; end
            obj.PlayAxis = axisName; obj.Frames = [];
            if axisName == "", return; end
            vals = rec.axis(axisName).Values;
            if numel(unique(stored)) >= 2, obj.Frames = vals(ismember(vals, stored)); else, obj.Frames = vals; end
        end

        function k = frameIndex(obj)
            % Position of the current coordinate among the frames (run order).
            vals = obj.Recording.axis(obj.PlayAxis).Values;
            cur = find(vals == obj.Coords.(obj.PlayAxis), 1);
            ord = arrayfun(@(v) find(vals == v, 1), obj.Frames);
            k = find(ord <= cur, 1, 'last'); if isempty(k), k = 0; end
        end

        function moved = jump(obj, direction)
            moved = false;
            if isempty(obj.Frames), return; end
            vals = obj.Recording.axis(obj.PlayAxis).Values;
            cur = find(vals == obj.Coords.(obj.PlayAxis), 1);
            ord = arrayfun(@(v) find(vals == v, 1), obj.Frames);
            if direction > 0, k = find(ord > cur, 1, 'first'); else, k = find(ord < cur, 1, 'last'); end
            if isempty(k), return; end
            obj.setCoord(obj.PlayAxis, obj.Frames(k));
            moved = true;
        end

        function tick(obj)
            if isempty(obj.Figure) || ~isvalid(obj.Figure), obj.stopTimer(); return; end
            try
                target = min(numel(obj.Frames), obj.PlayStart + floor(toc(obj.PlayClock) * obj.PlayFps));
                k = obj.frameIndex();
                if target > k, obj.setCoord(obj.PlayAxis, obj.Frames(target)); end
                if target >= numel(obj.Frames), obj.pause(); end
            catch
                obj.pause();
            end
        end

        function stopTimer(obj)
            if ~isempty(obj.Timer) && isvalid(obj.Timer)
                stop(obj.Timer); delete(obj.Timer);
            end
            obj.Timer = [];
        end

        function onRecent(obj, d)
            f = string(d.Value);
            if strlength(f) > 0 && f ~= string(obj.Recording.File), obj.loadFile(f); end
        end

        function remember(obj, file)
            if ~obj.Remember, return; end
            f = string(file);
            try, f = string(char(java.io.File(char(f)).getCanonicalPath())); catch, end   % one entry per file
            list = strings(1, 0);
            if ispref(obj.PrefGroup, 'RecentFiles'), list = string(getpref(obj.PrefGroup, 'RecentFiles')); end
            list = [f, list(list ~= f)];
            list = list(1:min(end, obj.MaxRecent));
            setpref(obj.PrefGroup, 'RecentFiles', cellstr(list));
        end

        function restartIfPlaying(obj)
            if obj.Playing, obj.pause(); obj.play(); end
        end

        function onKey(obj, e)
            switch e.Key
                case 'space', obj.togglePlay();
                case 'rightarrow', obj.stepForward();
                case 'leftarrow', obj.stepBack();
            end
        end

        % ------------------------------------------------------------ drawer
        function tf = drawerOpen(obj)
            w = obj.Body.ColumnWidth{3};
            tf = isnumeric(w) && w > 0;
        end

        function setDrawer(obj, show)
            if show, obj.Body.ColumnWidth{3} = obj.DrawerWidth; obj.fillDrawer();
            else, obj.Body.ColumnWidth{3} = 0; end
            obj.DetailsButton.Value = show && obj.DrawerTile == 0;
        end

        function fillDrawer(obj)
            if obj.DrawerTile > 0 && obj.DrawerTile <= numel(obj.Tiles)
                t = obj.Tiles(obj.DrawerTile);
                obj.DrawerTitle.Text = char("ⓘ  " + t.Spec.Title);
                lines = [obj.describe(t.Spec.Names), {'', ['Shown now: ' t.Caption.Text]}];
            else
                obj.DrawerTitle.Text = char("ⓘ  Details");
                lines = obj.Layout.notes(obj.Recording, obj.Coords);
                shown = strings(1, 0);
                for k = 1:numel(obj.Tiles), shown = [shown, obj.Tiles(k).Spec.Names]; end %#ok<AGROW>
                lines = [lines, {'', 'TILES'}, obj.describe(unique(shown, 'stable')), {'', 'RECORDING'}, ...
                    cellstr(splitlines(obj.provenance()))'];
            end
            obj.DrawerText.Value = lines;
        end

        function lines = describe(obj, names)
            rec = obj.Recording; lines = {};
            for n = names
                info = rec.info(n);
                lines{end+1} = sprintf('* %s  --  %s', info.Symbol, n); %#ok<AGROW>
                lines{end+1} = ['  ' char(info.Description)]; %#ok<AGROW>
                if info.Kind == "image" && numel(info.DisplayRange) == 2
                    lines{end+1} = sprintf('  Grey scale: black = %g, white = %g (colour bar).', info.DisplayRange); %#ok<AGROW>
                end
                if isa(info, 'OceanReplay.Core.ViewInfo')
                    if ~info.InTexts, lines{end+1} = '  WARNING -- NOT IN THE TEXTS'; end %#ok<AGROW>
                    lines{end+1} = ['  HOW IT WAS MADE: ' char(info.HowMade)]; %#ok<AGROW>
                    lines{end+1} = ['  Reference: ' char(info.Reference)]; %#ok<AGROW>
                end
            end
        end

        % ------------------------------------------------------------ timeline
        function names = timelineChannels(obj)
            names = strings(1, 0);
            for c = obj.Recording.Channels
                if c.Kind == "scalar" && isscalar(c.Axes), names(end+1) = c.Name; end %#ok<AGROW>
            end
        end

        function drawTimeline(obj)
            % One series (so no legend box; the dropdown and the y label name it), x in PLAY
            % order, a labelled cursor, and the stored points on a separate strip below.
            ax = obj.TimelineAxes; rug = obj.RugAxes; T = obj.Theme;
            R = 'OceanViz.App.Replay.Renderers.IChannelRenderer';
            feval([R '.prepare'], ax); feval([R '.prepare'], rug);
            name = string(obj.TimelineDropdown.Value);
            if ~any(obj.timelineChannels() == name), return; end
            info = obj.Recording.info(name); a = info.Axes(1);
            vals = obj.Recording.axis(a).Values;
            [x, y] = obj.Recording.series(name);
            % curve (2 px, recessive grid)
            hold(ax, 'on');
            h = plot(ax, x, y, '-', 'Color', T.Primary, 'LineWidth', 2);
            h.DataTipTemplate.DataTipRows(1).Label = char(a);
            h.DataTipTemplate.DataTipRows(2).Label = char(name);
            if isfield(obj.Coords, a)
                c = obj.Coords.(a);
                [~, k] = min(abs(x - c));
                xl = xline(ax, c, '-', sprintf('%s = %g', a, c), 'Color', T.Boundary, 'LineWidth', 1.5, ...
                    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'top', 'FontSize', 9);
                xl.HitTest = 'off';
                pt = plot(ax, x(k), y(k), 'o', 'MarkerSize', 7, 'MarkerFaceColor', T.Boundary, ...
                    'MarkerEdgeColor', T.Card, 'LineWidth', 1.5);
                pt.HitTest = 'off';
            end
            hold(ax, 'off');
            ylabel(ax, OceanViz.App.Replay.Notation.latex(info), 'Interpreter', 'latex', 'FontSize', 12);
            feval([R '.finish'], ax, T, true);
            grid(ax, 'on'); ax.GridAlpha = 0.15; ax.XTickLabel = {}; ax.TickLength = [0 0]; ax.Box = 'off';
            ax.YLim = OceanViz.App.Replay.ReplayStudio.padded(y);
            % strip: where the tiles' data were stored (not on top of the curve)
            stored = [];
            for kk = 1:numel(obj.Tiles)
                for n = obj.Tiles(kk).Spec.Names
                    ci = obj.Recording.info(n);
                    if ~isempty(ci.Axes) && ci.Axes(1) == a
                        st = obj.Recording.stored(n); stored = [stored; st.Values(:, 1)]; %#ok<AGROW>
                    end
                end
            end
            hold(rug, 'on');
            if ~isempty(stored)
                stored = unique(stored);
                r = plot(rug, stored, ones(size(stored)), 'o', 'MarkerSize', 4, 'LineStyle', 'none', ...
                    'MarkerFaceColor', T.Text * 0.6 + T.Card * 0.4, 'MarkerEdgeColor', 'none');
                r.HitTest = 'off';
            end
            if isfield(obj.Coords, a)
                xline(rug, obj.Coords.(a), '-', 'Color', T.Boundary, 'LineWidth', 1.5, 'HitTest', 'off');
            end
            hold(rug, 'off');
            feval([R '.finish'], rug, T, true);
            rug.YLim = [0 1.6]; rug.YTick = []; rug.YColor = 'none';          % dots float above the axis line
            rug.TickDir = 'out'; rug.Box = 'off';
            obj.TimelineHint.Text = sprintf('x: %s (play order)   ·   dots: where the tiles'' data are stored   ·   hover for values, click to jump', a);
            % shared x in play order (a descending run such as diffusion 1000 -> 1 reads left to right)
            lim = [min(vals) max(vals)]; if lim(1) == lim(2), lim = lim + [-0.5 0.5]; end
            dirn = 'normal'; if numel(vals) > 1 && vals(1) > vals(end), dirn = 'reverse'; end
            set([ax rug], 'XLim', lim, 'XDir', dirn);
            if all(vals == round(vals)) && numel(vals) <= 20, rug.XTick = sort(vals); end
            ax.XTick = rug.XTick;
            ax.Interactions = dataTipInteraction;                     % hover: the value at a step
            for hax = [ax rug]
                hax.UserData = a; hax.PickableParts = 'all';
                hax.ButtonDownFcn = @(~, e) obj.timelineClick(e);
            end
        end

        function timelineClick(obj, e)
            a = string(obj.TimelineAxes.UserData);
            if strlength(a) == 0, return; end
            obj.setCoord(a, e.IntersectionPoint(1));
        end

        % ------------------------------------------------------------ text helpers
        function name = pickLayout(obj, requested)
            names = cellfun(@(l) l.name(), obj.Layouts);
            if requested ~= "auto", name = requested; return; end
            name = names(1);                                          % domain layouts are listed first
            if isfield(obj.Recording.Meta, 'layout')
                k = find(lower(names) == lower(string(obj.Recording.Meta.layout)), 1);
                if ~isempty(k), name = names(k); end
            end
        end

        function s = notationOf(obj, spec)
            % The textbook notation of what a tile shows, as LaTeX -- always visible.
            parts = strings(1, 0);
            for n = spec.Names
                parts(end+1) = string(OceanViz.App.Replay.Notation.latex(obj.Recording.info(n))); %#ok<AGROW>
            end
            s = char(strjoin(unique(parts, 'stable'), ",\quad "));
        end

        function s = summary(obj)
            m = obj.Recording.Meta; rec = obj.Recording;
            parts = strings(0, 1);
            if isfield(m, 'title'), parts(end+1) = string(m.title); end
            if isfield(m, 'source') && isfield(m.source, 'language'), parts(end+1) = "Recorded in " + string(m.source.language); end
            parts(end+1) = "Status: " + rec.Status;
            if rec.isPartial(), parts(end) = parts(end) + " (PARTIAL)"; end
            s = strjoin(parts, newline);
        end

        function s = provenance(obj)
            m = obj.Recording.Meta; rec = obj.Recording;
            parts = strings(0, 1);
            if isfield(m, 'title'), parts(end+1) = "Title: " + string(m.title); end
            if isfield(m, 'source') && isfield(m.source, 'language')
                parts(end+1) = "Recorded in: " + string(m.source.language);
                if isfield(m.source, 'library'), parts(end) = parts(end) + " (" + string(m.source.library) + ")"; end
            end
            if isfield(m, 'runId'), parts(end+1) = "Run: " + string(m.runId); end
            if isfield(m, 'created'), parts(end+1) = "Created: " + string(m.created); end
            parts(end+1) = "Status: " + rec.Status;
            if isfield(m, 'extractedFrom'), parts(end+1) = "Extract of run: " + string(m.extractedFrom.archiveRunId); end
            if isfield(m, 'references')
                r = string(m.references);
                if ~isempty(r), parts = [parts(:); "References:"; "  " + r(:)]; end
            end
            s = strjoin(parts, newline);
        end

        function c = mergeCoords(obj, at)
            c = obj.Coords;
            f = fieldnames(at);
            for i = 1:numel(f), c.(f{i}) = at.(f{i}); end
        end

        function s = whereText(~, at, requested)
            f = fieldnames(at);
            if isempty(f), s = 'stored once (no axis)'; return; end
            parts = strings(1, numel(f)); moved = false;
            for i = 1:numel(f)
                parts(i) = sprintf('%s %g', f{i}, at.(f{i}));
                moved = moved || at.(f{i}) ~= requested.(f{i});
            end
            s = char("stored at " + strjoin(parts, ", "));
            if moved
                q = strings(1, numel(f));
                for i = 1:numel(f), q(i) = sprintf('%g', requested.(f{i})); end
                s = [s char("  ·  nearest stored to " + strjoin(q, ", "))];
            end
        end
    end
end
