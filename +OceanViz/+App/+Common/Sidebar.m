classdef Sidebar < handle
    % SIDEBAR  Vertical stack of labelled controls for a panel (grows row by
    %   row in a uigridlayout -- no hand-computed pixel positions).
    %
    %   sb = Sidebar(parentGrid, ctx);     sb.Grid is placed by the caller
    %   dd = sb.dropdown('Sampler:', items);   ed = sb.number('Seed:', 1);
    %   b  = sb.button('Run', @(~,~) ...);     s  = sb.slider('Step i:', [0 1000], 0, @cb);

    properties (SetAccess = private)
        Grid
        Theme
    end

    methods
        function obj = Sidebar(parent, ctx)
            obj.Theme = ctx.Theme;
            obj.Grid = uigridlayout(parent, [1 1], 'RowHeight', {}, 'ColumnWidth', {'1x'}, ...
                'RowSpacing', 4, 'Padding', [8 8 8 8], 'BackgroundColor', ctx.Theme.Panel, 'Scrollable', 'on');
        end

        function h = heading(obj, text)
            h = obj.add(@(g) uilabel(g, 'Text', text, 'FontWeight', 'bold', 'FontColor', obj.Theme.Accent));
        end

        function h = label(obj, text)
            h = obj.add(@(g) uilabel(g, 'Text', text, 'FontColor', obj.Theme.Text, 'WordWrap', 'on'), 'fit');
        end

        function h = dropdown(obj, labelText, items, value)
            obj.label(labelText);
            if nargin < 4, value = items{1}; end
            h = obj.add(@(g) uidropdown(g, 'Items', items, 'Value', value));
        end

        function h = number(obj, labelText, value, limits)
            obj.label(labelText);
            if nargin < 4, limits = [-Inf Inf]; end
            h = obj.add(@(g) uieditfield(g, 'numeric', 'Value', value, 'Limits', limits));
        end

        function h = checkbox(obj, text, value)
            h = obj.add(@(g) uicheckbox(g, 'Text', text, 'Value', value, 'FontColor', obj.Theme.Text));
        end

        function h = button(obj, text, callback, primary)
            if nargin < 4, primary = false; end
            h = obj.add(@(g) uibutton(g, 'Text', text, 'ButtonPushedFcn', callback), 30);
            if primary, h.BackgroundColor = obj.Theme.Accent; h.FontColor = [1 1 1]; h.FontWeight = 'bold'; end
        end

        function h = slider(obj, labelText, limits, value, callback)
            obj.label(labelText);
            h = obj.add(@(g) uislider(g, 'Limits', limits, 'Value', value, 'FontColor', obj.Theme.Text, ...
                'ValueChangedFcn', callback), 40);
        end

        function h = add(obj, makeFcn, height)
            if nargin < 3, height = 24; end
            obj.Grid.RowHeight{end+1} = height;
            h = makeFcn(obj.Grid);
            h.Layout.Row = numel(obj.Grid.RowHeight);
            h.Layout.Column = 1;
        end
    end
end
