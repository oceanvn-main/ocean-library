classdef Store < handle
    % STORE  All component data of a world: one row per entity, one plain array per component (no tables, no
    % per-entity objects), so every rule computes on whole columns.
    %
    %   D = OceanAgents.World.Guard.Dimension;  T = @OceanAgents.World.Guard.ComponentType;
    %   s = OceanAgents.World.Store(struct('Position', T('Width', 2, 'Dimension', D.L()), ...
    %                                      'Charge', T('Kind', "linear", 'Dimension', D.E()), ...
    %                                      'Born', T('Kind', "clock", 'Dimension', D.T())));
    %   rows = s.addRows(2);
    %   s.setOffer(rows(1), struct('Position', struct('Value', [0 0]), ...
    %       'Charge', struct('Value', struct('Value', 10, 'Rate', -1)), 'Born', struct('Value', 3)), 5);
    %   s.valueAt("Charge", rows(1), 8)     % 10 + (-1)*(8 - 5) = 7   (linear: Base + Rate.*(now - T0))
    %   s.valueAt("Born", rows(1), 8)       % 8 - 3 = 5               (clock: now - T0)
    %
    % Component kinds (W1_Kernel_Contract.md sections 3.2 and 5.1; World_Kernel_Design.md sections 4.2 and 5):
    %   value   Data (N x W, of the declared class: double | logical | string) and Has (N x 1)
    %   linear  Base, Rate (N x W double), T0 (N x 1): value at now = Base + Rate .* (now - T0); costs nothing while
    %           the rate is constant. A write REBASES: Base = new value at now, T0 = now, Rate = new rate (or the old).
    %   clock   T0 (N x 1): value at now = now - T0 (e.g. age); a write sets T0.
    %
    % Rows are appended by addRows, marked dead by killRows, never reordered during an episode, and compacted only by
    % clearAll (episode reset). Storage grows by doubling, so repeated single-row spawns stay amortised O(1).
    % The Store validates DATA (class, width, shape, finiteness, alive rows having the component); it does NOT check
    % which door may write what (the World does) nor whether an offered ComponentType matches (the Guard does).
    % Malformed data raises OceanAgents:World:BadValue; two writes to one (component, row) in one call raise
    % OceanAgents:World:WriteConflict.

    properties (SetAccess = private)
        Types struct = struct()         % name -> OceanAgents.World.Guard.ComponentType (declared, all world components)
        Names string = strings(1, 0)    % component names, in declaration order
        NumRows (1,1) double = 0        % rows in use (alive or dead) this episode
    end

    properties (Access = private)
        Cap_ (1,1) double = 0           % allocated rows (>= NumRows)
        Alive_ logical = false(0, 1)    % Cap_ x 1
        C_ struct = struct()            % name -> struct: Has, and Data | Base, Rate, T0 | T0  (Cap_ rows)
        Dirty_ struct = struct()        % name -> Cap_ x 1 logical (linear and clock components only)
        Frozen_ double = zeros(0, 1)    % Cap_ x 1: instant at which a row's time functions were frozen (NaN = not frozen)
    end

    properties (Constant, Hidden)
        InitialCapacity = 16
    end

    methods
        function obj = Store(types)
            arguments
                types (1,1) struct = struct()
            end
            names = string(fieldnames(types))';
            for n = names
                if ~isa(types.(n), 'OceanAgents.World.Guard.ComponentType') || ~isscalar(types.(n))
                    error('OceanAgents:World:BadValue', 'Store: type of "%s" must be a scalar Guard.ComponentType.', n);
                end
            end
            obj.Types = types;
            obj.Names = names;
            obj.clearAll();
        end

        % ------------------------------------------------------------ rows
        function rows = addRows(obj, k)
            % ADDROWS  Append k empty, alive rows (no component) and return their indices (k x 1).
            arguments
                obj
                k (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            first = obj.NumRows + 1;
            obj.ensureCapacity_(obj.NumRows + k);
            rows = (first:obj.NumRows + k)';
            obj.Alive_(rows) = true;
            obj.NumRows = obj.NumRows + k;
        end

        function tf = isAlive(obj, rows)
            % ISALIVE  n x 1 logical: the rows exist and have not been killed.
            arguments
                obj
                rows {mustBeIndexVector_}
            end
            rows = obj.checkRows_(rows, false);
            tf = obj.Alive_(rows);
        end

        function killRows(obj, rows)
            % KILLROWS  Entity removed: every component is cleared (Has = false, defaults) and the row is dead.
            arguments
                obj
                rows {mustBeIndexVector_}
            end
            rows = obj.checkRows_(rows, false);
            for n = obj.Names
                obj.resetRows_(n, rows);
                if isfield(obj.Dirty_, n), obj.Dirty_.(n)(rows) = false; end
            end
            obj.Alive_(rows) = false;
            obj.Frozen_(rows) = NaN;
        end

        function clearAll(obj)
            % CLEARALL  Episode reset: no rows at all (storage compacted).
            obj.NumRows = 0;
            obj.Cap_ = 0;
            obj.Alive_ = false(0, 1);
            obj.Frozen_ = zeros(0, 1);
            obj.C_ = struct();
            obj.Dirty_ = struct();
            for n = obj.Names
                t = obj.Types.(n);
                switch t.Kind
                    case "value"
                        obj.C_.(n) = struct('Has', false(0, 1), 'Data', defaults_(t, 0));
                    case "linear"
                        obj.C_.(n) = struct('Has', false(0, 1), 'Base', zeros(0, t.Width), ...
                            'Rate', zeros(0, t.Width), 'T0', zeros(0, 1));
                        obj.Dirty_.(n) = false(0, 1);
                    case "clock"
                        obj.C_.(n) = struct('Has', false(0, 1), 'T0', zeros(0, 1));
                        obj.Dirty_.(n) = false(0, 1);
                end
            end
        end

        % ------------------------------------------------------------ membership
        function setOffer(obj, rows, offer, now)
            % SETOFFER  Write the offered components of alive rows and mark them present (spawn / attach).
            %   offer.(name) = struct('Value', v[, 'Type', offeredType])  (the Command.Offer format; Type is ignored
            %   here - matching it is the Guard's job). v per kind, with 1 or numel(rows) rows:
            %     value  : n x W of the declared class        linear : struct('Value', n x W, 'Rate', n x W)
            %     clock  : T0 (n x 1)
            %   now: the current instant; required when a linear component is offered (its T0 becomes now).
            % Everything is validated before anything is written (OceanAgents:World:BadValue).
            arguments
                obj
                rows {mustBeIndexVector_}
                offer (1,1) struct
                now {mustBeTimeOrEmpty_} = []
            end
            rows = obj.checkRows_(rows, true);
            n = numel(rows);
            names = string(fieldnames(offer))';
            plan = struct('Name', {}, 'A', {}, 'B', {});
            for f = names
                if ~isfield(obj.Types, f)
                    bad_('offer names "%s", which is not a declared component.', f);
                end
                o = offer.(f);
                if ~(isstruct(o) && isscalar(o) && isfield(o, 'Value'))
                    bad_('offer.%s must be struct(''Value'', v[, ''Type'', t]) (the Command.Offer format).', f);
                end
                v = o.Value;
                t = obj.Types.(f);
                switch t.Kind
                    case "value"
                        a = checkBlock_(v, t, n, f, "value");  b = [];
                    case "linear"
                        if ~(isstruct(v) && isscalar(v) && isfield(v, 'Value') && isfield(v, 'Rate'))
                            bad_('linear offer "%s" must be struct(Value, Rate).', f);
                        end
                        checkNow_(now, f);
                        a = checkBlock_(v.Value, t, n, f, "value");
                        b = checkBlock_(v.Rate, t, n, f, "rate");
                    case "clock"
                        a = checkBlock_(v, t, n, f, "T0");  b = [];
                end
                plan(end + 1) = struct('Name', f, 'A', a, 'B', b); %#ok<AGROW>
            end
            for p = plan
                c = obj.C_.(p.Name);
                switch obj.Types.(p.Name).Kind
                    case "value"
                        c.Data(rows, :) = p.A;
                    case "linear"
                        c.Base(rows, :) = p.A;  c.Rate(rows, :) = p.B;  c.T0(rows) = now;
                        obj.Dirty_.(p.Name)(rows) = true;
                    case "clock"
                        c.T0(rows) = p.A;
                        obj.Dirty_.(p.Name)(rows) = true;
                end
                c.Has(rows) = true;
                obj.C_.(p.Name) = c;
            end
        end

        function clearComponent(obj, rows, name)
            % CLEARCOMPONENT  Detach: Has = false and the data back to defaults. Linear/clock rows are marked dirty so
            % a predictor learns that their parameters changed.
            arguments
                obj
                rows {mustBeIndexVector_}
                name {mustBeNameText_}
            end
            name = obj.checkName_(name);
            rows = obj.checkRows_(rows, false);
            obj.resetRows_(name, rows);
            if isfield(obj.Dirty_, name), obj.Dirty_.(name)(rows) = obj.Alive_(rows); end
        end

        function tf = has(obj, name, rows)
            % HAS  n x 1 logical: the rows have the component.
            arguments
                obj
                name {mustBeNameText_}
                rows {mustBeIndexVector_}
            end
            name = obj.checkName_(name);
            rows = obj.checkRows_(rows, false);
            tf = obj.C_.(name).Has(rows);
        end

        function rows = rowsWith(obj, names)
            % ROWSWITH  Alive rows having ALL of names, ascending (all alive rows for no names).
            arguments
                obj
                names string = strings(1, 0)
            end
            m = obj.Alive_(1:obj.NumRows);
            for n = reshape(names, 1, [])
                m = m & obj.C_.(obj.checkName_(n)).Has(1:obj.NumRows);
            end
            rows = find(m);
        end

        % ------------------------------------------------------------ reading
        function v = valueAt(obj, name, rows, now)
            % VALUEAT  Values at time now: value -> Data; linear -> Base + Rate .* (now - T0); clock -> now - T0.
            % Rows without the component return the kind's default evaluation (callers read Has alongside).
            arguments
                obj
                name {mustBeNameText_}
                rows {mustBeIndexVector_}
                now {mustBeTimeOrEmpty_} = []
            end
            name = obj.checkName_(name);
            rows = obj.checkRows_(rows, false);
            c = obj.C_.(name);
            kind = obj.Types.(name).Kind;
            if kind ~= "value"
                checkNow_(now, "valueAt of " + name);
            end
            at = now;                                  % frozen rows are evaluated at their freeze instant
            if kind ~= "value"
                f = obj.Frozen_(rows);
                if any(~isnan(f)), at = repmat(now, numel(rows), 1); m = ~isnan(f); at(m) = min(now, f(m)); end
            end
            switch kind
                case "value"
                    v = c.Data(rows, :);
                case "linear"
                    v = c.Base(rows, :) + c.Rate(rows, :) .* (at - c.T0(rows));
                case "clock"
                    v = at - c.T0(rows);
            end
        end

        function r = raw(obj, name)
            % RAW  Copy of the stored arrays of one component (rows 1..NumRows): Has, and Data | Base, Rate, T0 | T0.
            arguments
                obj
                name {mustBeNameText_}
            end
            name = obj.checkName_(name);
            r = structfun(@(a) a(1:obj.NumRows, :), obj.C_.(name), 'UniformOutput', false);
        end

        % ------------------------------------------------------------ writing
        function checkWrites(obj, writes, now)
            % CHECKWRITES  Validate a struct array of writes (Component, Rows, Values[, Rates]) without applying it:
            % known component, rows alive and having it, no (component, row) written twice, Values n x W of the
            % declared class (finite for double), Rates n x W finite (linear only, optional), clock Values n x 1.
            % Owner permission is NOT checked here (the World does that).
            arguments
                obj
                writes {mustBeWriteList_}
                now {mustBeTimeOrEmpty_}
            end
            obj.planWrites_(writes, now);
        end

        function applyWrites(obj, writes, now)
            % APPLYWRITES  Validate (as checkWrites) and apply, vectorised per write. Linear writes rebase
            % (Base = Values, T0 = now, Rate = Rates or the old rate); clock writes set T0 = Values.
            arguments
                obj
                writes {mustBeWriteList_}
                now {mustBeTimeOrEmpty_}
            end
            plan = obj.planWrites_(writes, now);
            for p = plan
                c = obj.C_.(p.Name);
                switch obj.Types.(p.Name).Kind
                    case "value"
                        c.Data(p.Rows, :) = p.Values;
                    case "linear"
                        c.Base(p.Rows, :) = p.Values;
                        c.T0(p.Rows) = now;
                        if ~isempty(p.Rates), c.Rate(p.Rows, :) = p.Rates; end
                        obj.Dirty_.(p.Name)(p.Rows) = true;
                    case "clock"
                        c.T0(p.Rows) = p.Values;
                        obj.Dirty_.(p.Name)(p.Rows) = true;
                end
                obj.C_.(p.Name) = c;
            end
        end

        function rows = takeDirty(obj, name)
            % TAKEDIRTY  Alive rows whose linear/clock parameters were written (by a write, an offer or a detach)
            % since the last take, ascending; then forgets them. Always empty for a value component.
            arguments
                obj
                name {mustBeNameText_}
            end
            name = obj.checkName_(name);
            if ~isfield(obj.Dirty_, name), rows = zeros(0, 1); return; end
            d = obj.Dirty_.(name);
            rows = find(d(1:obj.NumRows) & obj.Alive_(1:obj.NumRows));
            obj.Dirty_.(name)(:) = false;
        end

        % ------------------------------------------------------------ freezing (active registration)
        function tf = isFrozen(obj, rows)
            % ISFROZEN  n x 1 logical: the rows' time functions are frozen (their linear/clock values are held).
            arguments
                obj
                rows {mustBeIndexVector_}
            end
            rows = obj.checkRows_(rows, false);
            tf = ~isnan(obj.Frozen_(rows));
        end

        function t = frozenAt(obj, rows)
            % FROZENAT  n x 1 instants at which the rows were frozen (NaN: not frozen).
            rows = obj.checkRows_(rows, false);
            t = obj.Frozen_(rows);
        end

        function freeze(obj, rows, now)
            % FREEZE  Hold the linear and clock values of alive rows at instant now: valueAt returns the value they had
            % at now for every later time, until unfreeze. No stored component data is changed.
            arguments
                obj
                rows {mustBeIndexVector_}
                now (1,1) double
            end
            rows = obj.checkRows_(rows, true);
            if any(~isnan(obj.Frozen_(rows))), bad_('freeze: a row is already frozen.'); end
            obj.Frozen_(rows) = now;
        end

        function unfreeze(obj, rows, now)
            % UNFREEZE  Resume the time functions at instant now WITHOUT a jump: every linear/clock anchor of the rows
            % is shifted by the frozen duration (T0 += now - frozenAt), so the value at now equals the value at the
            % freeze instant and then evolves as before. Rows are marked dirty so predictors schedule from now.
            arguments
                obj
                rows {mustBeIndexVector_}
                now (1,1) double
            end
            rows = obj.checkRows_(rows, true);
            f = obj.Frozen_(rows);
            if any(isnan(f)), bad_('unfreeze: a row is not frozen.'); end
            if any(now < f), bad_('unfreeze at %g precedes the freeze instant.', now); end
            for n = obj.Names
                if obj.Types.(n).Kind == "value", continue; end
                c = obj.C_.(n);
                has = c.Has(rows);
                c.T0(rows(has)) = c.T0(rows(has)) + (now - f(has));
                obj.C_.(n) = c;
                obj.Dirty_.(n)(rows(has)) = true;
            end
            obj.Frozen_(rows) = NaN;
        end

        % ------------------------------------------------------------ whole state
        function snap = snapshot(obj)
            % SNAPSHOT  Value copy of the whole state (rows 1..NumRows), for tests and fail-stop verification.
            trim = @(a) a(1:obj.NumRows, :);
            C = struct();  D = struct();
            for n = obj.Names
                C.(n) = structfun(trim, obj.C_.(n), 'UniformOutput', false);
                if isfield(obj.Dirty_, n), D.(n) = trim(obj.Dirty_.(n)); end
            end
            snap = struct('Names', obj.Names, 'NumRows', obj.NumRows, 'Alive', trim(obj.Alive_), ...
                'Components', C, 'Dirty', D, 'Frozen', trim(obj.Frozen_));
        end

        function restore(obj, snap)
            % RESTORE  Return exactly to a snapshot taken from this Store (same component names).
            arguments
                obj
                snap {mustBeScalarStruct_}
            end
            if ~(isstruct(snap) && isscalar(snap) && all(isfield(snap, {'Names', 'NumRows', 'Alive', 'Components', 'Dirty'})) ...
                    && isequal(snap.Names, obj.Names) && numel(snap.Alive) == snap.NumRows)
                bad_('restore needs a snapshot taken from a Store with the same components.');
            end
            for n = obj.Names
                if ~isfield(snap.Components, n), bad_('snapshot lacks component "%s".', n); end
            end
            obj.NumRows = snap.NumRows;
            obj.Cap_ = snap.NumRows;
            obj.Alive_ = reshape(snap.Alive, [], 1);
            obj.C_ = snap.Components;
            obj.Dirty_ = snap.Dirty;
            if isfield(snap, 'Frozen'), obj.Frozen_ = reshape(snap.Frozen, [], 1); else, obj.Frozen_ = nan(snap.NumRows, 1); end
        end

        % ------------------------------------------------------------ pillars
        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('%d components stored as plain column arrays, one row per entity', numel(obj.Names)), ...
                'linear: Base + Rate (now - T0); clock: now - T0; rows never reordered within an episode'};
            meta.IsSymbolic = false;
            meta.Reference = "W1_Kernel_Contract.md section 5.1 (kernel engine)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "Store";
            spec.Inputs = struct('Name', "writes", 'Domain', "World-Writes", 'Dtype', "struct", 'Shape', "1 x K");
            spec.Outputs = struct('Name', "components", 'Domain', "World-Components", 'Dtype', "double|logical|string", ...
                'Shape', sprintf("%d rows x %d components", obj.NumRows, numel(obj.Names)));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "store");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function ensureCapacity_(obj, need)
            if need <= obj.Cap_, return; end
            cap = max([need, 2 * obj.Cap_, obj.InitialCapacity]);
            extra = cap - obj.Cap_;
            obj.Alive_(cap, 1) = false;
            obj.Frozen_ = [obj.Frozen_; nan(cap - numel(obj.Frozen_), 1)];
            for n = obj.Names
                t = obj.Types.(n);
                c = obj.C_.(n);
                c.Has(cap, 1) = false;
                switch t.Kind
                    case "value"
                        c.Data = [c.Data; defaults_(t, extra)];
                    case "linear"
                        c.Base(cap, t.Width) = 0;  c.Rate(cap, t.Width) = 0;  c.T0(cap, 1) = 0;
                        obj.Dirty_.(n)(cap, 1) = false;
                    case "clock"
                        c.T0(cap, 1) = 0;
                        obj.Dirty_.(n)(cap, 1) = false;
                end
                obj.C_.(n) = c;
            end
            obj.Cap_ = cap;
        end

        function resetRows_(obj, name, rows)
            t = obj.Types.(name);
            c = obj.C_.(name);
            c.Has(rows) = false;
            switch t.Kind
                case "value"
                    c.Data(rows, :) = defaults_(t, numel(rows));
                case "linear"
                    c.Base(rows, :) = 0;  c.Rate(rows, :) = 0;  c.T0(rows) = 0;
                case "clock"
                    c.T0(rows) = 0;
            end
            obj.C_.(name) = c;
        end

        function name = checkName_(obj, name)
            if ~((isstring(name) || ischar(name)) && isfield(obj.Types, string(name)))
                bad_('"%s" is not a declared component.', string(name));
            end
            name = string(name);
        end

        function rows = checkRows_(obj, rows, mustBeAlive)
            % Column of valid row indices (integers in 1..NumRows); optionally all alive.
            if ~(isnumeric(rows) && isreal(rows) && (isvector(rows) || isempty(rows)))
                bad_('rows must be a vector of row indices.');
            end
            rows = double(reshape(rows, [], 1));
            if any(rows < 1 | rows > obj.NumRows | rows ~= fix(rows))
                bad_('row indices must be integers in 1..%d.', obj.NumRows);
            end
            if mustBeAlive && ~all(obj.Alive_(rows))
                bad_('rows %s are dead.', mat2str(rows(~obj.Alive_(rows))'));
            end
        end

        function plan = planWrites_(obj, writes, now)
            % Validate every write; return the normalised plan (Name, Rows, Values, Rates). Nothing is changed.
            plan = struct('Name', {}, 'Rows', {}, 'Values', {}, 'Rates', {});
            if isempty(writes), return; end
            if ~isstruct(writes) || ~all(isfield(writes, {'Component', 'Rows', 'Values'}))
                bad_('writes must be a struct array with fields Component, Rows, Values[, Rates].');
            end
            hasRates = isfield(writes, 'Rates');
            linearUsed = false;
            seen = struct();
            for k = 1:numel(writes)
                w = writes(k);
                name = obj.checkName_(w.Component);
                t = obj.Types.(name);
                rows = obj.checkRows_(w.Rows, true);
                n = numel(rows);
                if numel(unique(rows)) ~= n
                    error('OceanAgents:World:WriteConflict', 'A write to "%s" names a row twice.', name);
                end
                if ~all(obj.C_.(name).Has(rows))
                    bad_('rows %s do not have component "%s".', mat2str(rows(~obj.C_.(name).Has(rows))'), name);
                end
                if isfield(seen, name)
                    if any(seen.(name)(rows))
                        error('OceanAgents:World:WriteConflict', '"%s" is written twice for the same row.', name);
                    end
                else
                    seen.(name) = false(obj.NumRows, 1);
                end
                seen.(name)(rows) = true;
                rates = [];
                if hasRates, rates = w.Rates; end
                switch t.Kind
                    case "value"
                        vals = checkBlock_(w.Values, t, n, name, "value", true);
                        if ~isempty(rates), bad_('"%s" is not linear and takes no Rates.', name); end
                    case "linear"
                        linearUsed = true;
                        vals = checkBlock_(w.Values, t, n, name, "value", true);
                        if ~isempty(rates), rates = checkBlock_(rates, t, n, name, "rate", true); end
                    case "clock"
                        vals = checkBlock_(w.Values, t, n, name, "T0", true);
                        if ~isempty(rates), bad_('"%s" is a clock and takes no Rates.', name); end
                end
                plan(end + 1) = struct('Name', name, 'Rows', rows, 'Values', vals, 'Rates', rates); %#ok<AGROW>
            end
            if linearUsed, checkNow_(now, "writes"); end
        end
    end
end

function d = defaults_(t, n)
switch t.Class
    case "double",  d = zeros(n, t.Width);
    case "logical", d = false(n, t.Width);
    case "string",  d = repmat("", n, t.Width);
end
end

function v = checkBlock_(v, t, n, name, what, exactRows)
% A block of n rows (or one row broadcast, unless exactRows) of the declared class and width.
if nargin < 6, exactRows = false; end
if what == "value", cls = t.Class; w = t.Width; elseif what == "rate", cls = "double"; w = t.Width;
else, cls = "double"; w = 1; end
okClass = (cls == "double" && isa(v, 'double') && isreal(v)) || (cls == "logical" && islogical(v)) || ...
    (cls == "string" && isstring(v));
if ~okClass
    bad_('%s of "%s" must be %s, got %s.', what, name, cls, class(v));
end
if ~ismatrix(v) || size(v, 2) ~= w || ~(size(v, 1) == n || (~exactRows && size(v, 1) == 1))
    bad_('%s of "%s" must be %d x %d, got %s.', what, name, n, w, mat2str(size(v)));
end
if cls == "double" && ~all(isfinite(v(:)))
    bad_('%s of "%s" contains NaN or Inf.', what, name);
end
if cls == "string" && any(ismissing(v(:)))
    bad_('%s of "%s" contains missing strings.', what, name);
end
if size(v, 1) ~= n, v = repmat(v, n, 1); end
end

function checkNow_(now, what)
if ~(isnumeric(now) && isreal(now) && isscalar(now) && isfinite(now) && now >= 0)
    bad_('a finite non-negative "now" is required for %s (linear components rebase at now).', what);
end
end

function bad_(fmt, varargin)
error('OceanAgents:World:BadValue', ['Store: ' fmt], varargin{:});
end

% ---- argument validators (raise OceanAgents:World:BadValue, the Store's documented ID) ----
function mustBeIndexVector_(x)
if ~(isnumeric(x) && isreal(x) && (isvector(x) || isempty(x)))
    bad_('rows must be a numeric vector of row indices.');
end
end

function mustBeNameText_(x)
if ~((isstring(x) && isscalar(x)) || (ischar(x) && (isrow(x) || isempty(x))))
    bad_('a component name must be text.');
end
end

function mustBeTimeOrEmpty_(x)
if ~(isempty(x) || (isnumeric(x) && isreal(x) && isscalar(x)))
    bad_('now must be a real numeric scalar.');
end
end

function mustBeWriteList_(x)
if ~(isempty(x) || isstruct(x))
    bad_('writes must be a struct array with fields Component, Rows, Values[, Rates].');
end
end

function mustBeScalarStruct_(x)
if ~(isstruct(x) && isscalar(x))
    bad_('restore needs a snapshot struct.');
end
end
