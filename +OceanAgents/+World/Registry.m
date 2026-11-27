classdef Registry < handle
    % REGISTRY  Entity identity: allocates entity IDs (positive integers, increasing, NEVER reused - not even across
    % episodes) and maps each live ID to its Store row.
    %
    %   r = OceanAgents.World.Registry();
    %   ids = r.allocate(2);           % [1; 2]
    %   r.bind(ids, [1; 2]);           % id <-> row
    %   r.rowsOf(2)                    % 2        (unknown or removed ID -> OceanAgents:World:UnknownEntity)
    %   r.retire(1);  r.isAlive([1 2]) % [false; true]
    %   r.clearEpisode();              % rows released at reset; the next allocate still continues from 3
    %
    % ID states: allocated (not yet bound), alive (bound to a row), retired (removed, or released by clearEpisode).
    % Stored as plain arrays indexed by ID and by row (IDs are dense, so lookups are vectorised indexing).
    % Contract: W1_Kernel_Contract.md section 5.2.

    properties (SetAccess = private)
        LastID (1,1) double = 0         % the most recently allocated ID (the counter; never reset)
    end

    properties (Access = private)
        State_ uint8 = zeros(0, 1, 'uint8')   % per ID: 1 allocated, 2 alive, 3 retired
        Row_ double = zeros(0, 1)             % per ID: bound row (0 when none)
        IdOfRow_ double = zeros(0, 1)         % per row: alive ID bound to it (0 when none)
    end

    properties (Constant, Hidden)
        Allocated = uint8(1)
        Alive = uint8(2)
        Retired = uint8(3)
    end

    methods
        function obj = Registry()
        end

        function ids = allocate(obj, k)
            % ALLOCATE  k new IDs (k x 1), each larger than every ID ever allocated by this Registry.
            arguments
                obj
                k (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            ids = (obj.LastID + 1:obj.LastID + k)';
            obj.State_(ids, 1) = obj.Allocated;
            obj.Row_(ids, 1) = 0;
            obj.LastID = obj.LastID + k;
        end

        function bind(obj, ids, rows)
            % BIND  Bind allocated, not yet bound IDs to free rows (one row each; validated before any change).
            arguments
                obj
                ids {mustBeNumberVector_}
                rows {mustBeNumberVector_}
            end
            ids = obj.checkIds_(ids);
            rows = checkRows_(rows);
            if numel(ids) ~= numel(rows)
                bad_('bind needs as many rows (%d) as IDs (%d).', numel(rows), numel(ids));
            end
            if any(obj.State_(ids) ~= obj.Allocated)
                bad_('IDs %s are not freshly allocated (already bound or retired).', ...
                    mat2str(ids(obj.State_(ids) ~= obj.Allocated)'));
            end
            if numel(unique(ids)) ~= numel(ids) || numel(unique(rows)) ~= numel(rows)
                bad_('bind needs distinct IDs and distinct rows.');
            end
            inUse = rows <= numel(obj.IdOfRow_);
            inUse(inUse) = obj.IdOfRow_(rows(inUse)) ~= 0;
            if any(inUse)
                bad_('rows %s are already bound to live entities.', mat2str(rows(inUse)'));
            end
            obj.State_(ids) = obj.Alive;
            obj.Row_(ids) = rows;
            if ~isempty(rows) && max(rows) > numel(obj.IdOfRow_)
                obj.IdOfRow_(max(rows), 1) = 0;                       % grow (new entries are 0 = no entity)
            end
            obj.IdOfRow_(rows) = ids;
        end

        function rows = rowsOf(obj, ids)
            % ROWSOF  Rows of alive IDs (same order); any unknown, unbound or removed ID -> World:UnknownEntity.
            arguments
                obj
                ids {mustBeNumberVector_}
            end
            ids = obj.checkIds_(ids);
            if any(obj.State_(ids) ~= obj.Alive)
                error('OceanAgents:World:UnknownEntity', 'Entity ID(s) %s are not alive (unknown, unbound or removed).', ...
                    mat2str(ids(obj.State_(ids) ~= obj.Alive)'));
            end
            rows = obj.Row_(ids);
        end

        function ids = idsOf(obj, rows)
            % IDSOF  Alive entity ID bound to each row (0 for a row with no live entity).
            arguments
                obj
                rows {mustBeNumberVector_}
            end
            rows = checkRows_(rows);
            ids = zeros(numel(rows), 1);
            known = rows <= numel(obj.IdOfRow_);
            ids(known) = obj.IdOfRow_(rows(known));
        end

        function tf = isAlive(obj, ids)
            % ISALIVE  n x 1 logical; false (never an error) for unknown, unbound and removed IDs.
            arguments
                obj
                ids {mustBeNumberVector_}
            end
            if ~(isnumeric(ids) && isreal(ids) && (isvector(ids) || isempty(ids)))
                bad_('IDs must be a numeric vector.');
            end
            ids = double(reshape(ids, [], 1));
            tf = false(numel(ids), 1);
            ok = ids >= 1 & ids <= obj.LastID & ids == fix(ids);
            tf(ok) = obj.State_(ids(ok)) == obj.Alive;
        end

        function retire(obj, ids)
            % RETIRE  Removed entities: their rows are released and the IDs stay retired forever.
            arguments
                obj
                ids {mustBeNumberVector_}
            end
            ids = obj.checkIds_(ids);
            if any(obj.State_(ids) ~= obj.Alive)
                error('OceanAgents:World:UnknownEntity', 'Cannot retire entity ID(s) %s: not alive.', ...
                    mat2str(ids(obj.State_(ids) ~= obj.Alive)'));
            end
            obj.IdOfRow_(obj.Row_(ids)) = 0;
            obj.Row_(ids) = 0;
            obj.State_(ids) = obj.Retired;
        end

        function clearEpisode(obj)
            % CLEAREPISODE  Episode reset: every row is released and every allocated or alive ID retired; the ID
            % counter is NOT reset, so no ID is ever handed out twice.
            obj.State_(:) = obj.Retired;
            obj.Row_(:) = 0;
            obj.IdOfRow_ = zeros(0, 1);
        end

        function n = numAlive(obj)
            n = nnz(obj.State_ == obj.Alive);
        end

        function snap = snapshot(obj)
            % SNAPSHOT  Value copy of the whole state (for fail-stop verification).
            snap = struct('LastID', obj.LastID, 'State', obj.State_, 'Row', obj.Row_, 'IdOfRow', obj.IdOfRow_);
        end

        function restore(obj, snap)
            arguments
                obj
                snap {mustBeSnapshot_}
            end
            if ~(isstruct(snap) && isscalar(snap) && all(isfield(snap, {'LastID', 'State', 'Row', 'IdOfRow'})) && ...
                    numel(snap.State) == snap.LastID && numel(snap.Row) == snap.LastID)
                bad_('restore needs a Registry snapshot.');
            end
            obj.LastID = snap.LastID;
            obj.State_ = snap.State;
            obj.Row_ = snap.Row;
            obj.IdOfRow_ = snap.IdOfRow;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('%d IDs allocated, %d alive; IDs never reused', obj.LastID, obj.numAlive())};
            meta.IsSymbolic = false;
            meta.Reference = "W1_Kernel_Contract.md section 5.2 (kernel engine)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "Registry";
            spec.Inputs = struct('Name', "ids", 'Domain', "World-EntityID", 'Dtype', "double", 'Shape', "n x 1");
            spec.Outputs = struct('Name', "rows", 'Domain', "World-Row", 'Dtype', "double", 'Shape', "n x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "registry");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function ids = checkIds_(obj, ids)
            if ~(isnumeric(ids) && isreal(ids) && (isvector(ids) || isempty(ids)))
                bad_('IDs must be a numeric vector.');
            end
            ids = double(reshape(ids, [], 1));
            if any(ids < 1 | ids ~= fix(ids) | ~isfinite(ids))
                bad_('IDs must be positive integers.');
            end
            if any(ids > obj.LastID)
                error('OceanAgents:World:UnknownEntity', 'Entity ID(s) %s were never allocated.', mat2str(ids(ids > obj.LastID)'));
            end
        end
    end
end

function rows = checkRows_(rows)
if ~(isnumeric(rows) && isreal(rows) && (isvector(rows) || isempty(rows)))
    bad_('rows must be a numeric vector.');
end
rows = double(reshape(rows, [], 1));
if any(rows < 1 | rows ~= fix(rows) | ~isfinite(rows))
    bad_('rows must be positive integers.');
end
end

function bad_(fmt, varargin)
error('OceanAgents:World:BadValue', ['Registry: ' fmt], varargin{:});
end

% ---- argument validators (raise OceanAgents:World:BadValue, the Registry's documented ID) ----
function mustBeNumberVector_(x)
if ~(isnumeric(x) && isreal(x) && (isvector(x) || isempty(x)))
    bad_('IDs and rows must be numeric vectors.');
end
end

function mustBeSnapshot_(x)
if ~(isstruct(x) && isscalar(x))
    bad_('restore needs a Registry snapshot struct.');
end
end
