classdef ChannelInfo
    % CHANNELINFO  Description of one recorded quantity (Format spec Sec. 1-3).
    %   Everything the replay needs to draw AND explain a channel: its kind,
    %   the axes it depends on, its logical shape, and its textbook symbol,
    %   units and description. ValueShape may be left empty in a domain's
    %   catalog when it is only known at run time; the writer then fixes it
    %   from the first recorded value.
    %
    %   Example:
    %     c = OceanReplay.Core.ChannelInfo("x", "image", 'Axes', "step", ...
    %         'ValueShape', [64 64 3], 'Symbol', "x_i", 'DisplayRange', [-1 1], ...
    %         'Description', "the sampler's current sample (input to the network)");

    properties
        Name (1,1) string
        Kind (1,1) string
        Axes (1,:) string = strings(1, 0)
        ValueShape (1,:) double = []
        Dtype (1,1) string = ""            % "" = the spec's precision ("string" for tokens)
        Symbol (1,1) string = ""
        SymbolLatex (1,1) string = ""      % optional: the symbol as LaTeX, e.g. '$\epsilon_\phi(x_i, i)$'
        Units (1,1) string = ""
        Description (1,1) string = ""
        DisplayRange (1,:) double = []
        DimFormat (1,1) string = ""
        Categories (1,:) string = strings(1, 0)
        RowLabels (1,:) string = strings(1, 0)
        ColLabels (1,:) string = strings(1, 0)
        LabelsChannel (1,1) string = ""
        Level (1,1) double = NaN
    end

    methods
        function obj = ChannelInfo(name, kind, opts)
            % name and kind are REQUIRED: were they optional, a name such as "label" would be
            % parsed as the start of the name-value option LabelsChannel.
            arguments
                name (1,1) string
                kind (1,1) string
                opts.?OceanReplay.Core.ChannelInfo
            end
            obj = obj.setup(name, kind, opts);
        end

        function obj = setup(obj, name, kind, opts)
            % Shared by subclasses: assign name, kind and name-value options, then check.
            if name == "" || contains(name, "/")
                error('OceanReplay:ChannelInfo:name', 'Channel name "%s" must be non-empty and contain no "/".', name);
            end
            OceanReplay.Core.Kinds.rank(kind);                                  % errors on unknown kind
            obj.Name = name; obj.Kind = kind;
            f = fieldnames(opts);
            for i = 1:numel(f)
                if ~ismember(f{i}, {'Name', 'Kind'}), obj.(f{i}) = opts.(f{i}); end
            end
            if kind == "image" && obj.DimFormat == "", obj.DimFormat = "HWC"; end
            if kind == "scalar" && isempty(obj.ValueShape), obj.ValueShape = 1; end
            if kind == "tokens", obj.Dtype = "string"; end
            if ~isempty(obj.ValueShape)
                msg = OceanReplay.Core.Kinds.checkShape(kind, obj.ValueShape);
                if msg ~= "", error('OceanReplay:ChannelInfo:shape', 'Channel "%s": %s.', name, msg); end
            end
            for a = obj.Axes
                if ~isvarname(a), error('OceanReplay:ChannelInfo:axis', 'Channel "%s": axis name "%s" is not a valid identifier.', name, a); end
            end
        end

        function msg = checkValue(obj, v)
            % "" when v matches this channel's kind and shape.
            if isempty(obj.ValueShape)
                msg = OceanReplay.Core.Kinds.checkValue(obj.Kind, OceanReplay.Core.ChannelInfo.shapeOf(obj.Kind, v), v);
            else
                msg = OceanReplay.Core.Kinds.checkValue(obj.Kind, obj.ValueShape, v);
            end
        end

        function s = attrs(obj)
            % Group attributes written to the file (Format spec Sec. 3.1). Optional ones only when set.
            s = struct('kind', obj.Kind, 'axes', obj.Axes, 'valueShape', int32(obj.ValueShape), ...
                'dtype', obj.Dtype, 'symbol', obj.Symbol, 'units', obj.Units, ...
                'description', obj.Description, 'level', int32(OceanReplay.Core.ChannelInfo.levelOut(obj.Level)));
            if ~isempty(obj.DisplayRange), s.displayRange = obj.DisplayRange; end
            if obj.DimFormat ~= "", s.dimFormat = obj.DimFormat; end
            if ~isempty(obj.Categories), s.categories = obj.Categories; end
            if ~isempty(obj.RowLabels), s.rowLabels = obj.RowLabels; end
            if ~isempty(obj.ColLabels), s.colLabels = obj.ColLabels; end
            if obj.LabelsChannel ~= "", s.labelsChannel = obj.LabelsChannel; end
            if obj.SymbolLatex ~= "", s.symbolLatex = obj.SymbolLatex; end
        end
    end

    methods (Static)
        function l = levelOut(l)
            % Level NaN (not from a spec group, e.g. a view) is stored as -1.
            if isnan(l), l = -1; end
        end

        function shape = shapeOf(kind, v)
            % Logical shape of a value of a kind (used when the catalog left ValueShape empty).
            switch kind
                case "scalar", shape = 1;
                case {"vector", "tokens"}, shape = numel(v);
                otherwise
                    shape = size(v, 1:OceanReplay.Core.Kinds.rank(kind));
            end
        end

        function obj = fromAttrs(name, a)
            % Rebuild from normalised file attributes (struct of the attrs() field names).
            opts = struct('Axes', a.axes, 'ValueShape', a.valueShape, 'Dtype', a.dtype, ...
                'Symbol', a.symbol, 'Units', a.units, 'Description', a.description, 'Level', a.level);
            opts = OceanReplay.Core.ChannelInfo.optionals(opts, a);
            c = namedargs2cell(opts);
            obj = OceanReplay.Core.ChannelInfo(name, a.kind, c{:});
        end

        function opts = optionals(opts, a)
            map = {'displayRange', 'DisplayRange'; 'dimFormat', 'DimFormat'; 'categories', 'Categories'; ...
                   'rowLabels', 'RowLabels'; 'colLabels', 'ColLabels'; 'labelsChannel', 'LabelsChannel'; ...
                   'symbolLatex', 'SymbolLatex'};
            for i = 1:size(map, 1)
                if isfield(a, map{i, 1}), opts.(map{i, 2}) = a.(map{i, 1}); end
            end
        end
    end
end
