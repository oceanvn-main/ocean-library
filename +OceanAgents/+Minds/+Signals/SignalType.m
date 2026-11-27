classdef (Sealed) SignalType
    % SIGNALTYPE  What a signal is, independent of who sends it: its shape in time, kind of value, width, allowed labels,
    % physical dimension, role and (for streams) rate. Parts declare the types they accept and emit; the router matches
    % them by these properties only, never by the names of parts (AB1_Signal_Body_Contract.md section 3.1).
    %
    %   T = @(varargin) OceanAgents.Minds.Signals.SignalType(varargin{:});
    %   move  = T('ValueKind', "label", 'Labels', ["up" "right" "down" "left"], 'Role', "command");
    %   level = T('Shape', "wave", 'ValueKind', "number", 'Role', "sense");
    %   move.accepts(T('ValueKind', "label", 'Labels', ["up" "down"], 'Role', "command"))     % true: subset of labels
    %
    %   Shape     "event" (one instant) | "stream" (samples at Rate) | "wave" (defined at every instant; value = a Wave)
    %   ValueKind "flag" | "number" | "vector" (1 x Width) | "label" (string in Labels) | "record" (scalar struct)
    %             | "array" (numeric array of exactly Size, e.g. [H W C] for an image; AB2)
    %             | "flags" (1 x Width true/false, one per entry of Labels when given, e.g. the moves the Laws permit)
    %   Role      "command" | "sense" | "query" | "reply" | "modulation" | "reward"

    properties (SetAccess = immutable)
        Shape string = "event"
        ValueKind string = "record"
        Width double = 1
        Labels string = strings(1, 0)
        Dimension = OceanAgents.World.Guard.Dimension()
        Role string = "command"
        Rate double = NaN
        Size double = zeros(1, 0)               % "array" only: the exact size (trailing singleton dimensions allowed)
    end

    properties (Constant)
        Shapes = ["event" "stream" "wave"]
        ValueKinds = ["flag" "number" "vector" "label" "record" "array" "flags"]
        Roles = ["command" "sense" "query" "reply" "modulation" "reward"]
    end

    methods
        function obj = SignalType(options)
            arguments
                options.Shape = "event"
                options.ValueKind = "record"
                options.Width = 1
                options.Labels = strings(1, 0)
                options.Dimension = OceanAgents.World.Guard.Dimension()
                options.Role = "command"
                options.Rate = NaN
                options.Size = zeros(1, 0)
            end
            bad = @(msg) error('OceanAgents:Signals:BadType', 'SignalType: %s', msg);
            obj.Shape = string(options.Shape);
            if ~isscalar(obj.Shape) || ~any(obj.Shape == obj.Shapes), bad('unknown Shape.'); end
            obj.ValueKind = string(options.ValueKind);
            if ~isscalar(obj.ValueKind) || ~any(obj.ValueKind == obj.ValueKinds), bad('unknown ValueKind.'); end
            w = options.Width;
            if ~(isnumeric(w) && isscalar(w) && w >= 1 && w == fix(w)), bad('Width must be a positive integer.'); end
            obj.Width = double(w);
            if ~any(obj.ValueKind == ["vector" "flags"]) && obj.Width ~= 1, bad('only a "vector" or "flags" has Width > 1.'); end
            obj.Labels = reshape(string(options.Labels), 1, []);
            if isempty(obj.Labels), obj.Labels = strings(1, 0); end
            if ~isempty(obj.Labels) && ~any(obj.ValueKind == ["label" "flags"])
                bad('Labels are allowed only for a "label" or "flags" value.');
            end
            if obj.ValueKind == "flags" && ~isempty(obj.Labels) && numel(obj.Labels) ~= obj.Width
                bad('"flags" Labels name each flag: one label per entry of Width.');
            end
            if numel(unique(obj.Labels)) ~= numel(obj.Labels), bad('Labels must be unique.'); end
            if ~isa(options.Dimension, 'OceanAgents.World.Guard.Dimension'), bad('Dimension must be a Guard.Dimension.'); end
            obj.Dimension = options.Dimension;
            if ~any(obj.ValueKind == ["number" "vector" "array"]) && ~obj.Dimension.isDimensionless()
                bad('only "number", "vector" and "array" values may carry a physical dimension.');
            end
            sz = options.Size;
            if obj.ValueKind == "array"
                if ~(isnumeric(sz) && isrow(sz) && numel(sz) >= 2 && all(sz >= 1) && all(sz == fix(sz)))
                    bad('an "array" needs Size: a row of at least two positive integers.');
                end
                obj.Size = double(sz);
            elseif ~isempty(sz)
                bad('Size is only for an "array".');
            end
            obj.Role = string(options.Role);
            if ~isscalar(obj.Role) || ~any(obj.Role == obj.Roles), bad('unknown Role.'); end
            r = options.Rate;
            if obj.Shape == "stream"
                if ~(isnumeric(r) && isscalar(r) && isfinite(r) && r > 0), bad('a "stream" needs a positive Rate.'); end
            elseif ~(isnumeric(r) && isscalar(r) && isnan(r))
                bad('Rate is only for a "stream".');
            end
            obj.Rate = double(r);
            if obj.Shape == "wave" && obj.ValueKind ~= "number", bad('a "wave" carries a "number" value.'); end
        end

        function tf = accepts(obj, offered)
            % ACCEPTS  True when a signal of type offered may be delivered to a receiver declaring obj.
            tf = strlength(obj.mismatch(offered)) == 0;
        end

        function why = mismatch(obj, offered)
            % MISMATCH  Name of the first property that prevents acceptance ("" when accepted).
            arguments
                obj
                offered (1,1) OceanAgents.Minds.Signals.SignalType
            end
            why = "";
            if obj.Shape ~= offered.Shape, why = "Shape";
            elseif obj.ValueKind ~= offered.ValueKind, why = "ValueKind";
            elseif obj.Width ~= offered.Width, why = "Width";
            elseif obj.Dimension ~= offered.Dimension, why = "Dimension";
            elseif obj.Role ~= offered.Role, why = "Role";
            elseif obj.Shape == "stream" && obj.Rate ~= offered.Rate, why = "Rate";
            elseif obj.ValueKind == "array" && ~OceanAgents.Minds.Signals.SignalType.sameSize_(obj.Size, offered.Size)
                why = "Size";
            elseif ~isempty(obj.Labels) && (isempty(offered.Labels) || ~all(ismember(offered.Labels, obj.Labels)))
                why = "Labels";
            end
        end

        function checkValue(obj, v)
            % CHECKVALUE  Throw OceanAgents:Signals:BadValue unless v is a valid value of this type.
            bad = @(msg) error('OceanAgents:Signals:BadValue', 'Signal value: %s', msg);
            if obj.Shape == "wave"
                if ~(isa(v, 'OceanAgents.Minds.Signals.Wave') && isscalar(v)), bad('a wave signal carries one Wave.'); end
                return
            end
            switch obj.ValueKind
                case "flag"
                    ok = isscalar(v) && (islogical(v) || (isnumeric(v) && (v == 0 || v == 1)));
                case "number"
                    ok = isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v);
                case "vector"
                    ok = isnumeric(v) && isreal(v) && isequal(size(v), [1 obj.Width]) && all(isfinite(v));
                case "flags"
                    ok = (islogical(v) || (isnumeric(v) && all(v(:) == 0 | v(:) == 1))) && isequal(size(v), [1 obj.Width]);
                case "array"
                    ok = isnumeric(v) && isreal(v) && all(isfinite(v(:))) && ...
                        OceanAgents.Minds.Signals.SignalType.sameSize_(size(v), obj.Size);
                case "label"
                    ok = (isstring(v) || ischar(v)) && isscalar(string(v)) && ...
                        (isempty(obj.Labels) || any(string(v) == obj.Labels));
                otherwise
                    ok = isstruct(v) && isscalar(v);
                    if ok, OceanAgents.Core.assertValueData(v, "Signal.Value"); end
            end
            if ~ok, bad(sprintf('not a valid "%s" value for this type.', obj.ValueKind)); end
        end

        function s = describe(obj)
            s = obj.Shape + " " + obj.ValueKind + " " + obj.Role;
            if any(obj.ValueKind == ["vector" "flags"]), s = s + " x" + obj.Width; end
            if obj.ValueKind == "array", s = s + " " + strjoin(string(obj.Size), "x"); end
            if ~isempty(obj.Labels), s = s + " {" + strjoin(obj.Labels, ",") + "}"; end
            if ~obj.Dimension.isDimensionless(), s = s + " [" + string(char(obj.Dimension)) + "]"; end
        end
    end

    methods (Static, Access = private)
        function tf = sameSize_(a, b)
            % SAMESIZE_  Equal sizes up to trailing singleton dimensions (MATLAB drops them: an H x W x 1 image is H x W).
            n = max(numel(a), numel(b));
            tf = isequal([a ones(1, n - numel(a))], [b ones(1, n - numel(b))]);
        end
    end

    methods (Static)
        function t = reply()
            % REPLY  The type of a receiver's answer (a record), e.g. a refusal with its reason. Cached.
            persistent cached
            if isempty(cached), cached = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "reply"); end
            t = cached;
        end

        function t = query()
            % QUERY  The type of a question to a memory (a record with Op). Cached.
            persistent cached
            if isempty(cached), cached = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "query"); end
            t = cached;
        end
    end
end
