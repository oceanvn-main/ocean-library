classdef IsolationChecker < handle
    % ISOLATIONCHECKER  Deep check that two agents' object graphs share no mutable (handle) object.
    % AgentHost's SharedComponent check only compares the top-level mind/learner; this walks everything reachable,
    % so a Q table, a RandStream or a cache smuggled in through a cell three levels down is still found (C01/C02).
    %
    %   [pathsA, pathsB] = OceanAgents.Runtime.IsolationChecker.sharedHandles(hostA, hostB);
    %   paths = OceanAgents.Runtime.IsolationChecker.sharedHandles(a, b, 'Allow', ["OceanStats.RL.Env.Spaces.ISpace"]);
    %
    % pathsA(i) and pathsB(i) are where the i-th shared handle sits inside a and inside b, e.g.
    % "Mind.Policy.Cache{2}" (property names, {k} cell elements, (k) array elements, ('key') containers.Map values,
    % @workspace.var captured function-handle variables). Each shared handle is reported once, at the first path
    % found; objects below a shared handle are not reported again. The root itself is "<root>".
    %
    % What is walked: object properties including private/protected ones (struct(obj), with the
    % MATLAB:structOnObject warning suppressed and restored), struct fields, cell elements, containers.Map values
    % and function-handle workspaces. Constant properties are class-level, not per-instance state, and are skipped.
    % Identity is eq on same-class handles. Allowlisted classes (and their subclasses) are skipped entirely;
    % the default names the ISpace descriptions every agent legitimately shares (A0 section 2.7). Note that
    % OceanStats.RL.Env.Spaces.ISpace is currently a value class, so it can never be reported anyway; the
    % allowlist matters for handle classes. Empty handles (e.g. an empty containers.Map) are still handles.
    % MATLAB dictionary values are walked (path "(key)"). Java objects are compared by REFERENCE identity
    % (java.util.IdentityHashMap; MATLAB's == on Java objects calls equals() and would confuse two equal but distinct
    % maps) and are not walked inside. Python objects (py.*) are treated the same way: compared by reference
    % (Python `is`) and not walked inside (attributes are computed, e.g. ndarray.T, so the graph has no end);
    % immutable Python scalars (None, bool, int, float, str, bytes) are plain values. Cycle-safe (each handle is visited once); nesting deeper than MaxDepth (50)
    % -> OceanAgents:Runtime:IsolationDepth.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.7.

    properties (Constant)
        DefaultAllow = ["OceanStats.RL.Env.Spaces.ISpace" "OceanAgents.Observe.IPlayerObserver"]   % observers may follow several players; Player.attach refuses one holding agent parts
        PythonValues = ["py.NoneType" "py.bool" "py.int" "py.float" "py.str" "py.bytes"]   % immutable: plain values
        DefaultMaxDepth = 50
    end

    methods (Static)
        function [pathsA, pathsB] = sharedHandles(a, b, options)
            % SHAREDHANDLES  Paths (in a, and in b) of every handle object reachable from both a and b.
            arguments
                a
                b
                options.Allow string = OceanAgents.Runtime.IsolationChecker.DefaultAllow
                options.MaxDepth (1,1) double {mustBeInteger, mustBePositive} = OceanAgents.Runtime.IsolationChecker.DefaultMaxDepth
            end
            state = struct('Allow', reshape(options.Allow, 1, []), 'MaxDepth', options.MaxDepth);
            previous = warning('off', 'MATLAB:structOnObject');
            restoreWarning = onCleanup(@() warning(previous));

            % Pass 1: every handle reachable from a, with the first path it was reached by.
            seenA = newSeen_();
            visit_(a, "", 0, @(x, p) remember(seenA, x, p), state);

            % Pass 2: walk b; a handle already seen in a is a shared handle (report it, do not descend).
            seenB = newSeen_();
            pathsA = strings(0, 1);
            pathsB = strings(0, 1);
            visit_(b, "", 0, @checkAgainstA, state);
            pathsA = displayPath_(pathsA);
            pathsB = displayPath_(pathsB);

            function descend = checkAgainstA(x, p)
                descend = remember(seenB, x, p);
                if ~descend, return; end
                [found, pa] = lookup_(seenA, x);
                if found
                    pathsA(end + 1, 1) = pa;
                    pathsB(end + 1, 1) = p;
                    descend = false;
                end
            end

        end

        function paths = reachable(x, classes, options)
            % REACHABLE  Paths of every handle reachable from x that is an instance of one of classes (subclasses
            % included), walked exactly as sharedHandles walks. Used to refuse observers that hold agent parts (PG1).
            arguments
                x
                classes string
                options.Allow string = "OceanStats.RL.Env.Spaces.ISpace"
                options.MaxDepth (1,1) double {mustBeInteger, mustBePositive} = OceanAgents.Runtime.IsolationChecker.DefaultMaxDepth
            end
            state = struct('Allow', reshape(options.Allow, 1, []), 'MaxDepth', options.MaxDepth);
            previous = warning('off', 'MATLAB:structOnObject');
            restoreWarning = onCleanup(@() warning(previous));
            seen = newSeen_();
            paths = strings(0, 1);
            visit_(x, "", 0, @check, state);
            paths = displayPath_(paths);

            function descend = check(h, p)
                descend = remember(seen, h, p);
                if descend && any(arrayfun(@(c) isa(h, c), classes)), paths(end + 1, 1) = p; end
            end
        end
    end

    methods
        function meta = getMetadata(obj)
            meta = pillarMeta_(obj, {sprintf('default allowlist %s; max depth %d; identity = eq on same-class handles', ...
                strjoin(obj.DefaultAllow, ", "), obj.DefaultMaxDepth)}, "A0 contract section 4.7 (isolation check, C01/C02)");
        end

        function spec = getPipelineSpec(obj)
            spec = pillarSpec_(obj, strings(1, 0), strings(1, 0));
        end
    end
end

% ---------------------------------------------------------------------------------------------------------------
function visit_(x, path, depth, onHandle, state)
% VISIT_  Depth-first walk of x. onHandle(h, path) is called once per handle and returns whether
% to descend into it.
    if depth > state.MaxDepth
        fail_('IsolationDepth', 'Object graph deeper than %d levels at "%s".', state.MaxDepth, path);
    end
    if isjava(x)
        if ~any(arrayfun(@(c) isa(x, c), state.Allow))
            onHandle(x, path);
        end                      % reference identity only; Java internals not walked
        return
    end
    if isnumeric(x) || islogical(x) || ischar(x) || isstring(x)
        return                                  % plain values hold no handles
    end
    if isPython_(x)
        if ~any(class(x) == OceanAgents.Runtime.IsolationChecker.PythonValues) && ...
                ~any(arrayfun(@(c) isa(x, c), state.Allow))
            onHandle(x, path);
        end                      % reference identity only; Python internals not walked
        return
    end
    if isa(x, 'function_handle')
        info = functions(x);
        if isfield(info, 'workspace')
            for w = 1:numel(info.workspace)
                ws = info.workspace{w};
                for f = reshape(string(fieldnames(ws)), 1, [])
                    visit_(ws.(f), path + "@workspace." + f, depth + 1, onHandle, state);
                end
            end
        end
        return
    end
    if iscell(x)
        for k = 1:numel(x)
            visit_(x{k}, path + "{" + k + "}", depth + 1, onHandle, state);
        end
        return
    end
    if isstruct(x)
        names = reshape(string(fieldnames(x)), 1, []);
        for e = 1:numel(x)
            base = path;
            if ~isscalar(x), base = path + "(" + e + ")"; end
            for f = names
                visit_(x(e).(f), join_(base, f), depth + 1, onHandle, state);
            end
        end
        return
    end
    if ~isobject(x) && ~isa(x, 'handle')
        return                                  % other built-in value types hold no handles
    end
    if any(arrayfun(@(c) isa(x, c), state.Allow))
        return                                  % allowlisted (shared by design)
    end
    if builtin('numel', x) ~= 1                 % object array (or empty): walk each element
        for k = 1:builtin('numel', x)
            visit_(x(k), path + "(" + k + ")", depth + 1, onHandle, state);
        end
        return
    end
    if isa(x, 'handle')
        if ~isvalid(x) || ~onHandle(x, path)
            return
        end
    end
    if isa(x, 'dictionary')                     % value type: walk its values by key
        if ~isConfigured(x) || numEntries(x) == 0, return; end
        ks = keys(x);
        vs = values(x);
        for k = 1:numel(ks)
            if iscell(vs), v = vs{k}; else, v = vs(k); end
            visit_(v, path + "(" + keyText_(ks(k)) + ")", depth + 1, onHandle, state);
        end
        return
    end
    if isa(x, 'containers.Map')
        ks = keys(x);
        vs = values(x);
        for k = 1:numel(ks)
            visit_(vs{k}, path + "(" + keyText_(ks{k}) + ")", depth + 1, onHandle, state);
        end
        return
    end
    props = objectFields_(x);
    for f = reshape(string(fieldnames(props)), 1, [])
        visit_(props.(f), join_(path, f), depth + 1, onHandle, state);
    end
end

function seen = newSeen_()
% NEWSEEN_  Visited-set: MATLAB handles bucketed by class (identity via eq), Java objects in an IdentityHashMap.
seen = struct('Matlab', containers.Map('KeyType', 'char', 'ValueType', 'any'), 'Java', []);
if usejava('jvm'), seen.Java = java.util.IdentityHashMap(); end
end

function descend = remember(seen, x, path)
% REMEMBER  Record handle x at path unless it was already visited; returns true on first visit.
[found, ~] = lookup_(seen, x);
descend = ~found;
if found, return; end
if isjava(x)
    seen.Java.put(x, char(path));
    return
end
key = class(x);
if isKey(seen.Matlab, key)
    entry = seen.Matlab(key);
else
    entry = struct('Handles', {{}}, 'Paths', strings(0, 1));
end
entry.Handles{end + 1} = x;
entry.Paths(end + 1, 1) = path;
seen.Matlab(key) = entry;   % containers.Map is a handle: the assignment persists
end

function [found, path] = lookup_(seen, x)
% LOOKUP_  Has handle x (identity) been seen?
found = false;
path = "";
if isjava(x)
    found = seen.Java.containsKey(x);
    if found, path = string(seen.Java.get(x)); end
    return
end
key = class(x);
if ~isKey(seen.Matlab, key), return; end
entry = seen.Matlab(key);
for k = 1:numel(entry.Handles)
    if sameHandle_(entry.Handles{k}, x)
        found = true;
        path = entry.Paths(k);
        return
    end
end
end

function tf = isPython_(x)
tf = startsWith(class(x), "py.");
end

function tf = sameHandle_(a, b)
tf = false;
try
    if isPython_(a)                    % == on Python objects calls __eq__ (content); identity is `is`
        tf = logical(py.operator.is_(a, b));
        return
    end
    r = (a == b);                      % handle identity (a class overloading eq by content defeats this)
    tf = isscalar(r) && logical(r);
catch
end
end

function s = objectFields_(x)
% OBJECTFIELDS_  All per-instance properties of object x (private ones included), minus Constant properties.
try
    s = struct(x);                                  % MATLAB:structOnObject is suppressed by the caller
catch
    s = struct();
    for p = reshape(string(properties(x)), 1, [])
        try
            s.(p) = x.(p);
        catch
        end
    end
end
mc = metaclass(x);
if ~isempty(mc)
    constants = intersect(string({mc.PropertyList([mc.PropertyList.Constant]).Name}), string(fieldnames(s)));
    if ~isempty(constants), s = rmfield(s, cellstr(constants)); end
end
end

function p = join_(path, name)
if path == "", p = name; else, p = path + "." + name; end
end

function t = keyText_(key)
if ischar(key) || isstring(key), t = "'" + string(key) + "'"; else, t = string(mat2str(key)); end
end

function p = displayPath_(p)
p(p == "") = "<root>";
end
