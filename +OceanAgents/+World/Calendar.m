classdef Calendar < handle
    % CALENDAR  The world's list of future discontinuities (design U4; S1.2b-2: keyed by OWNER, a registered
    % participant - the calendar knows no doors or rules): predicted crossings of linear/clock components
    % and structural commands scheduled for a later instant. The World asks it only two questions: "what is due now?"
    % (due) and "when is the next thing due?" (nextTime), so instants where nothing changes can be skipped exactly.
    %
    %   c = OceanAgents.World.Calendar('Phases', ["resolve" "act" "settle" "project"], 'Owners', ["mover" "battery"]);
    %   c.setNow(0, 0);                                                        % World: (instant, phase index) cursor
    %   c.schedulePrediction("battery/PredictEmpty", ids, times, "resolve", "battery", "Empty", 'Rank', 2);
    %   c.scheduleCommand(cmd, 370, "settle", "mover", 'Rank', 1);
    %   c.cancelEntity(id);                         % entity removed: its predictions and the commands targeting it go
    %   t = c.nextTime();                           % earliest item time, Inf if none
    %   items = c.due(t, "resolve");                % removes and returns the items at exactly (t, resolve), sorted
    %
    % ITEM FIELDS (struct array returned by due):
    %   Kind          "prediction" | "monitor" | "command"   ("monitor": a scheduled monitor check, see
    %                 scheduleMonitorCheck - marked by this flag, never by any key text)
    %   Time          integer tick at which the item is due
    %   Phase         phase name;  PhaseIndex  its position in the phase order
    %   Owner         owning participant's name (who scheduled it: a registered participant);  OwnerRank  its
    %                 registration position (1-based)
    %   Rank          position of the issuing part within its owner (0 when not given)
    %   IssuerEntity  prediction: the predicted entity; command: Command.IssuerEntity (0 = none)
    %   Seq           prediction: 0 (unique per (predictor, entity) anyway); command: Command.Seq (World-stamped)
    %   EntityID      the entity the item is ABOUT: prediction -> the predicted entity; command -> Command.Target
    %                 (0 for spawn). cancelEntity(id) drops every item with EntityID == id.
    %   PredictorKey  prediction: e.g. "battery/PredictEmpty" ("" for commands)
    %   DueEvent      prediction: event type to emit when due ("" for commands)
    %   Command       command: the OceanAgents.World.Command ([] for predictions)
    %
    % TIE-BREAK KEY (W1_Kernel_Contract.md section 5.3; the same everywhere):
    %   (Time, PhaseIndex, OwnerRank, Rank, IssuerEntity, Seq), then - only when all six are equal, which needs a
    %   caller to reuse a rank - predictions, then monitor checks, then commands, then key text. The order
    %   therefore never depends on insertion order.
    %
    % REPLACEMENT. schedulePrediction(key, ids, times, ...) replaces every earlier prediction for (key, id) for each
    % given id; a time of Inf removes it without replacement ("no crossing predicted").
    %
    % NO RETROACTIVE ITEMS. The calendar keeps a cursor (Now, NowPhase): NowPhase = k > 0 means phases 1..k of instant
    % Now have been (or are being) released; NowPhase = 0 means instant Now has not started. Scheduling at time < Now,
    % or at (Now, q) with q <= NowPhase, raises OceanAgents:World:Retroactive (the World moves same-phase items to the
    % synchronisation point before scheduling). due(t, p) moves the cursor to (t, p) and refuses to go backwards.
    %
    % STORAGE. One numeric column per key field (vectorised; 1e5 predictions cost a few arrays, not 1e5 structs).
    % Commands (few) are kept in a cell referenced by index, compacted as items are released or cancelled (the state
    % does not grow with history). Predictor keys and event types are interned string tables.

    properties (SetAccess = private)
        Phases string = strings(1, 0)     % optional phase order (lets callers pass phase names)
        Owners string = strings(1, 0)     % optional owner (participant) names in rank order (lets callers pass names)
        Now double = 0
        NowPhase double = 0
    end

    properties (Access = private)
        % one row per scheduled item (parallel columns)
        Time_ double = zeros(0, 1)
        PhaseIdx_ double = zeros(0, 1)
        OwnerRank_ double = zeros(0, 1)
        Rank_ double = zeros(0, 1)
        Issuer_ double = zeros(0, 1)
        Seq_ double = zeros(0, 1)
        IsCmd_ logical = false(0, 1)
        IsMon_ logical = false(0, 1)      % a monitor check (item flag; key text is never interpreted)
        Entity_ double = zeros(0, 1)
        KeyIdx_ double = zeros(0, 1)      % index into Keys_ (predictions), 0 for commands
        EventIdx_ double = zeros(0, 1)    % index into Events_ (predictions), 0 for commands
        CmdIdx_ double = zeros(0, 1)      % index into Cmds_ (commands), 0 for predictions
        Keys_ string = strings(0, 1)
        Events_ string = strings(0, 1)
        Cmds_ cell = cell(0, 1)
    end

    methods
        function obj = Calendar(options)
            arguments
                options.Phases string = strings(1, 0)
                options.Owners string = strings(1, 0)
            end
            obj.Phases = reshape(options.Phases, 1, []);
            obj.Owners = reshape(options.Owners, 1, []);
        end

        function setNow(obj, t, phaseIndex)
            % SETNOW  World: move the cursor to (t, phaseIndex); phaseIndex 0 = instant t not started. Never backwards.
            arguments
                obj
                t (1,1) double
                phaseIndex (1,1) double = 0
            end
            if ~isTick_(t) || ~isTick_(phaseIndex)
                bad_('BadValue', 'setNow needs a non-negative integer tick and phase index.');
            end
            if t < obj.Now || (t == obj.Now && phaseIndex < obj.NowPhase)
                bad_('Retroactive', 'Calendar cursor cannot move back from (%g, %g) to (%g, %g).', ...
                    obj.Now, obj.NowPhase, t, phaseIndex);
            end
            obj.Now = t;
            obj.NowPhase = phaseIndex;
        end

        function schedulePrediction(obj, predictorKey, entityIDs, times, phase, owner, dueEvent, options)
            % SCHEDULEPREDICTION  Replace the predictions of predictorKey for entityIDs by times (Inf = none).
            arguments
                obj
                predictorKey (1,1) string {mustBeNonzeroLengthText}
                entityIDs double
                times double
                phase
                owner
                dueEvent (1,1) string {mustBeNonzeroLengthText}
                options.Rank (1,1) double = 0
            end
            ids = reshape(entityIDs, [], 1);
            times = reshape(times, [], 1);
            obj.checkPredictionArgs_(ids, times);
            obj.schedule_(predictorKey, ids, times, phase, owner, obj.intern_("Events_", dueEvent), options.Rank, false);
        end

        function scheduleMonitorCheck(obj, checkKey, entityIDs, times, phase, owner)
            % SCHEDULEMONITORCHECK  Like schedulePrediction, for the World's skip-safe monitor checks: the items carry
            % the "monitor" flag (Kind "monitor", no event) and replace earlier checks of (checkKey, entity).
            arguments
                obj
                checkKey (1,1) string {mustBeNonzeroLengthText}
                entityIDs double
                times double
                phase
                owner
            end
            ids = reshape(entityIDs, [], 1);
            times = reshape(times, [], 1);
            obj.checkPredictionArgs_(ids, times);
            obj.schedule_(checkKey, ids, times, phase, owner, 0, 0, true);
        end

        function scheduleCommand(obj, cmd, time, phase, owner, options)
            % SCHEDULECOMMAND  A structural command to apply at (time, phase).
            arguments
                obj
                cmd (1,1) OceanAgents.World.Command
                time (1,1) double
                phase
                owner
                options.Rank (1,1) double = 0
            end
            if ~isTick_(time), bad_('BadValue', 'scheduleCommand: time must be a non-negative integer tick.'); end
            [p, d] = obj.resolve_(phase, owner, true);
            obj.checkNotRetroactive_(time, p);
            obj.Cmds_{end + 1, 1} = cmd;
            obj.append_(time, p, d, options.Rank, cmd.IssuerEntity, cmd.Seq, true, cmd.Target, 0, 0, numel(obj.Cmds_));
        end

        function cancelEntity(obj, entityIDs)
            % CANCELENTITY  Drop every item about these entities (their predictions, commands targeting them).
            arguments
                obj
                entityIDs double
            end
            drop = ismember(obj.Entity_, reshape(entityIDs, [], 1)) & obj.Entity_ > 0;
            obj.keep_(~drop);
        end

        function items = due(obj, t, phase)
            % DUE  Remove and return the items due exactly at (t, phase), sorted by the tie-break key.
            arguments
                obj
                t (1,1) double
                phase
            end
            p = obj.resolve_(phase, [], false);
            obj.setNow(t, p);
            stale = obj.Time_ < t | (obj.Time_ == t & obj.PhaseIdx_ < p);
            if any(stale)
                bad_('InvalidState', '%d calendar item(s) were due before (%g, phase %d) and never released.', ...
                    nnz(stale), t, p);
            end
            isDue = obj.Time_ == t & obj.PhaseIdx_ == p;
            items = obj.items_(find(isDue));
            obj.keep_(~isDue);
        end

        function t = nextTime(obj)
            % NEXTTIME  Earliest time over all items (Inf if the calendar is empty).
            if isempty(obj.Time_), t = Inf; else, t = min(obj.Time_); end
        end

        function n = count(obj)
            % COUNT  Number of scheduled items.
            n = numel(obj.Time_);
        end

        function items = peek(obj)
            % PEEK  All items, sorted by the tie-break key, without removing them (tests and traces).
            items = obj.items_((1:numel(obj.Time_))');
        end

        function clear(obj)
            % CLEAR  Remove every item and reset the cursor to (0, 0) (episode reset).
            obj.keep_(false(numel(obj.Time_), 1));
            obj.Cmds_ = cell(0, 1);
            obj.Keys_ = strings(0, 1);
            obj.Events_ = strings(0, 1);
            obj.Now = 0;
            obj.NowPhase = 0;
        end

        function snap = snapshot(obj)
            % SNAPSHOT  Complete value copy of the calendar state (fail-stop restore, K9 comparisons).
            snap = struct('Now', obj.Now, 'NowPhase', obj.NowPhase, 'Time', obj.Time_, 'PhaseIdx', obj.PhaseIdx_, ...
                'OwnerRank', obj.OwnerRank_, 'Rank', obj.Rank_, 'Issuer', obj.Issuer_, 'Seq', obj.Seq_, ...
                'IsCmd', obj.IsCmd_, 'IsMon', obj.IsMon_, 'Entity', obj.Entity_, 'KeyIdx', obj.KeyIdx_, 'EventIdx', obj.EventIdx_, ...
                'CmdIdx', obj.CmdIdx_, 'Keys', obj.Keys_, 'Events', obj.Events_, 'Cmds', {obj.Cmds_});
        end

        function restore(obj, snap)
            % RESTORE  Return to a state captured by snapshot().
            arguments
                obj
                snap (1,1) struct
            end
            obj.Now = snap.Now;  obj.NowPhase = snap.NowPhase;
            obj.Time_ = snap.Time;  obj.PhaseIdx_ = snap.PhaseIdx;  obj.OwnerRank_ = snap.OwnerRank;
            obj.Rank_ = snap.Rank;  obj.Issuer_ = snap.Issuer;  obj.Seq_ = snap.Seq;  obj.IsCmd_ = snap.IsCmd;  obj.IsMon_ = snap.IsMon;
            obj.Entity_ = snap.Entity;  obj.KeyIdx_ = snap.KeyIdx;  obj.EventIdx_ = snap.EventIdx;
            obj.CmdIdx_ = snap.CmdIdx;  obj.Keys_ = snap.Keys;  obj.Events_ = snap.Events;  obj.Cmds_ = snap.Cmds;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf(['%d scheduled items; tie-break (time, phase, owner rank, rank, issuer ' ...
                'entity, seq); predictions replace per (predictor, entity)'], obj.count())};
            meta.IsSymbolic = false;
            meta.Reference = "runtime/container component (W1_Kernel_Contract.md section 5.3)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "Calendar";
            spec.Inputs = [struct('Name', "prediction", 'Domain', "World-Prediction", 'Dtype', "double", 'Shape', "n x 1"), ...
                struct('Name', "command", 'Domain', "World-Command", 'Dtype', "OceanAgents.World.Command", 'Shape', "1 x 1")];
            spec.Outputs = struct('Name', "due", 'Domain', "World-CalendarItem", 'Dtype', "struct", 'Shape', "k x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "calendar");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function [p, d] = resolve_(obj, phase, owner, needOwner)
            % RESOLVE_  Phase index and owner rank from names (needs Phases/Owners) or positive integers.
            p = index_(phase, obj.Phases, 'phase');
            d = [];
            if needOwner, d = index_(owner, obj.Owners, 'owner'); end
        end

        function checkNotRetroactive_(obj, times, p)
            if any(times < obj.Now | (times == obj.Now & p <= obj.NowPhase))
                bad_('Retroactive', ['Cannot schedule at or before the cursor (%g, phase %d): earliest requested ' ...
                    'is (%g, phase %d).'], obj.Now, obj.NowPhase, min(times), p);
            end
        end

        function k = intern_(obj, table, text)
            k = find(obj.(table) == text, 1);
            if isempty(k)
                obj.(table)(end + 1, 1) = text;
                k = numel(obj.(table));
            end
        end

        function checkPredictionArgs_(~, ids, times)
            if numel(ids) ~= numel(times)
                bad_('BadValue', 'schedulePrediction: %d entity IDs but %d times.', numel(ids), numel(times));
            end
            if any(~(ids >= 1 & ids == fix(ids) & isfinite(ids))) || numel(unique(ids)) ~= numel(ids)
                bad_('BadValue', 'schedulePrediction: entity IDs must be unique positive integers.');
            end
            finite = isfinite(times);
            if any(isnan(times)) || any(times(finite) < 0 | times(finite) ~= fix(times(finite)))
                bad_('BadValue', 'schedulePrediction: times must be non-negative integer ticks or Inf.');
            end
        end

        function schedule_(obj, key, ids, times, phase, owner, eventIdx, rank, isMon)
            % Replace the (key, entity) items of this flag by the finite times (Inf removes).
            finite = isfinite(times);
            [p, d] = obj.resolve_(phase, owner, true);
            obj.checkNotRetroactive_(times(finite), p);
            k = obj.intern_("Keys_", key);
            old = ~obj.IsCmd_ & obj.IsMon_ == isMon & obj.KeyIdx_ == k & ismember(obj.Entity_, ids);
            obj.keep_(~old);
            ids = ids(finite);
            m = numel(ids);
            obj.append_(times(finite), repmat(p, m, 1), repmat(d, m, 1), repmat(rank, m, 1), ids, ...
                zeros(m, 1), false(m, 1), ids, repmat(k, m, 1), repmat(eventIdx, m, 1), zeros(m, 1), repmat(isMon, m, 1));
        end

        function append_(obj, t, p, d, r, issuer, seq, isCmd, entity, key, ev, cmd, isMon)
            if nargin < 13, isMon = false(numel(t), 1); end
            obj.IsMon_ = [obj.IsMon_; isMon];
            obj.Time_ = [obj.Time_; t];
            obj.PhaseIdx_ = [obj.PhaseIdx_; p];
            obj.OwnerRank_ = [obj.OwnerRank_; d];
            obj.Rank_ = [obj.Rank_; r];
            obj.Issuer_ = [obj.Issuer_; issuer];
            obj.Seq_ = [obj.Seq_; seq];
            obj.IsCmd_ = [obj.IsCmd_; isCmd];
            obj.Entity_ = [obj.Entity_; entity];
            obj.KeyIdx_ = [obj.KeyIdx_; key];
            obj.EventIdx_ = [obj.EventIdx_; ev];
            obj.CmdIdx_ = [obj.CmdIdx_; cmd];
        end

        function keep_(obj, mask)
            % KEEP_  Keep only the masked rows of every column; the command cell is compacted to the kept commands.
            obj.Time_ = obj.Time_(mask);
            obj.PhaseIdx_ = obj.PhaseIdx_(mask);
            obj.OwnerRank_ = obj.OwnerRank_(mask);
            obj.Rank_ = obj.Rank_(mask);
            obj.Issuer_ = obj.Issuer_(mask);
            obj.Seq_ = obj.Seq_(mask);
            obj.IsCmd_ = obj.IsCmd_(mask);
            obj.Entity_ = obj.Entity_(mask);
            obj.KeyIdx_ = obj.KeyIdx_(mask);
            obj.EventIdx_ = obj.EventIdx_(mask);
            obj.CmdIdx_ = obj.CmdIdx_(mask);
            obj.IsMon_ = obj.IsMon_(mask);
            if numel(obj.Cmds_) ~= nnz(obj.IsCmd_)                   % compact released/cancelled commands
                used = obj.CmdIdx_(obj.IsCmd_);
                obj.Cmds_ = obj.Cmds_(used);
                obj.CmdIdx_(obj.IsCmd_) = (1:numel(used))';
            end
        end

        function items = items_(obj, sel)
            % ITEMS_  Struct array of the selected rows, sorted by the tie-break key.
            sel = reshape(sel, [], 1);
            keyText = strings(numel(sel), 1);
            hasKey = obj.KeyIdx_(sel) > 0;
            keyText(hasKey) = obj.Keys_(obj.KeyIdx_(sel(hasKey)));
            [~, ~, keyRank] = unique(keyText);                      % rank of the key TEXT (not of insertion)
            K = [obj.Time_(sel) obj.PhaseIdx_(sel) obj.OwnerRank_(sel) obj.Rank_(sel) obj.Issuer_(sel) ...
                obj.Seq_(sel) 2 * double(obj.IsCmd_(sel)) + double(obj.IsMon_(sel)) keyRank(:)];
            [~, order] = sortrows(K);
            sel = sel(order);
            n = numel(sel);
            items = repmat(struct('Kind', "", 'Time', 0, 'Phase', "", 'PhaseIndex', 0, 'Owner', "", 'OwnerRank', 0, ...
                'Rank', 0, 'IssuerEntity', 0, 'Seq', 0, 'EntityID', 0, 'PredictorKey', "", 'DueEvent', "", ...
                'Command', []), n, 1);
            for j = 1:n
                r = sel(j);
                it = items(j);
                it.Time = obj.Time_(r);
                it.PhaseIndex = obj.PhaseIdx_(r);
                it.Phase = nameOf_(obj.Phases, it.PhaseIndex);
                it.OwnerRank = obj.OwnerRank_(r);
                it.Owner = nameOf_(obj.Owners, it.OwnerRank);
                it.Rank = obj.Rank_(r);
                it.IssuerEntity = obj.Issuer_(r);
                it.Seq = obj.Seq_(r);
                it.EntityID = obj.Entity_(r);
                if obj.IsCmd_(r)
                    it.Kind = "command";
                    it.Command = obj.Cmds_{obj.CmdIdx_(r)};
                elseif obj.IsMon_(r)
                    it.Kind = "monitor";
                    it.PredictorKey = obj.Keys_(obj.KeyIdx_(r));
                else
                    it.Kind = "prediction";
                    it.PredictorKey = obj.Keys_(obj.KeyIdx_(r));
                    it.DueEvent = obj.Events_(obj.EventIdx_(r));
                end
                items(j) = it;
            end
        end
    end
end

function k = index_(x, names, what)
if (isstring(x) || ischar(x)) && isscalar(string(x))
    k = find(names == string(x), 1);
    if isempty(k)
        bad_('BadValue', 'Unknown %s "%s" (declare it with the Calendar''s ''%ss'' option or pass its index).', ...
            what, string(x), [upper(what(1)) what(2:end)]);
    end
elseif isTick_(x) && x >= 1
    k = double(x);
else
    bad_('BadValue', 'A %s must be a name or a positive integer index.', what);
end
end

function s = nameOf_(names, k)
if k >= 1 && k <= numel(names), s = names(k); else, s = ""; end
end

function tf = isTick_(t)
tf = isnumeric(t) && isscalar(t) && isfinite(t) && t >= 0 && t == fix(t);
end

function bad_(reason, fmt, varargin)
error(sprintf('OceanAgents:World:%s', reason), fmt, varargin{:});
end
