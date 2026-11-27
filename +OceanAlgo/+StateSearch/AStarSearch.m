classdef AStarSearch < handle
    % ASTARSEARCH  A* and weighted A* best-first search on a deterministic planning model
    % (Hart, Nilsson & Raphael 1968; Russell & Norvig, "Artificial Intelligence: A Modern Approach", ch. 3;
    % weighted A*: Pohl 1970).
    %
    %   engine = OceanAlgo.StateSearch.AStarSearch('Heuristic', @(s) h, 'Weight', 1, ...
    %                'MaxExpansions', 1e5, 'RecordTrace', false)
    %   result = engine.search(model, s0)          model: OceanAlgo.StateSearch.IDistributionModel, one outcome
    %                                              per action (deterministic)
    %
    % Path cost: c(s,a) = -reward, which must be >= 0 (reward <= 0; no discounting). A goal is reached by a
    % terminated transition or at a state with model.isGoal(s). Heuristic h(s) is an estimate of the remaining
    % COST to a goal (default 0, which is uniform-cost search = Dijkstra).
    %
    %   f(n) = g(n) + w * h(n)      expand the open node with the smallest f; ties -> larger g, then the
    %                               node pushed first
    %
    % The goal test is applied when a node is selected for expansion, so the returned path is optimal when h is
    % admissible (h <= true cost-to-go) and w = 1. A node is re-opened whenever a cheaper path to it is found:
    % this keeps A* optimal for admissible but inconsistent h; with a consistent h no re-opening happens
    % (Work.Reopened = 0). With w > 1 (weighted A*) the path cost is at most w times optimal for admissible h,
    % usually with far fewer expansions.
    %
    % All terminated transitions lead to one goal sink node (their path cost is what matters); Path.States ends
    % with the actual successor state reported by the model.
    %
    % Status: "ok" (Path is the found plan), "no_path" (open list exhausted), "at_goal", "no_legal_action",
    % "budget_exhausted" (MaxExpansions reached: Path leads to the open node with the smallest f; Action is its
    % first action -- legal, but not a certified plan; Value = NaN, the partial path's return is sum(Path.Rewards)).
    %
    % Work: Expansions, Generated, Reopened, MaxFrontier, DistinctStates.
    % Trace (RecordTrace): table of expansions in order: Order, Key, G, F.
    properties (SetAccess = private)
        Heuristic (1,1) function_handle = @(s) 0
        Weight (1,1) double = 1
        MaxExpansions (1,1) double = 1e5
        RecordTrace (1,1) logical = false
    end
    properties (Access = private)
        States_ cell
        Parent_ double
        Action_ double
        Reward_ double
        G_ double
        Expanded_ logical
        N_ (1,1) double = 0
    end

    methods
        function obj = AStarSearch(opts)
            arguments
                opts.Heuristic (1,1) function_handle = @(s) 0
                opts.Weight (1,1) double {mustBeFinite, mustBeGreaterThanOrEqual(opts.Weight, 1)} = 1
                opts.MaxExpansions (1,1) double {mustBePositive} = 1e5
                opts.RecordTrace (1,1) logical = false
            end
            obj.Heuristic = opts.Heuristic; obj.Weight = opts.Weight;
            obj.MaxExpansions = opts.MaxExpansions; obj.RecordTrace = opts.RecordTrace;
        end

        function result = search(obj, model, s0)
            arguments
                obj
                model (1,1) OceanAlgo.StateSearch.IDistributionModel
                s0
            end
            name = "A*"; if obj.Weight > 1, name = "weighted A*"; end
            rootActions = model.actions(s0);
            if model.isGoal(s0)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "at_goal", 'Value', 0, ...
                    'Actions', rootActions); return
            end
            if isempty(rootActions)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_legal_action", ...
                    'Actions', rootActions); return
            end

            obj.reset_(256);
            index = OceanAlgo.StateSearch.StateTable();
            open = OceanAlgo.StateSearch.MinHeap(3);
            sink = 0; pushes = 0; generated = 1; expansions = 0; reopened = 0; maxFrontier = 1;
            trace = zeros(0, 3); traceKeys = strings(0, 1);
            root = obj.add_(s0, 0, NaN, 0, 0);
            index.set(model.key(s0), root);
            pushes = pushes + 1; open.push(root, [obj.Weight * obj.h_(s0), 0, pushes]);

            status = "no_path"; goalId = 0;
            while ~open.isEmpty()
                [id, k] = open.peek();
                if -k(2) > obj.G_(id)                       % stale entry (a cheaper path was found later)
                    open.pop(); continue
                end
                if id == sink || model.isGoal(obj.States_{id})
                    status = "ok"; goalId = id; break
                end
                if expansions >= obj.MaxExpansions
                    status = "budget_exhausted"; goalId = id; break
                end
                open.pop();
                expansions = expansions + 1;
                if obj.Expanded_(id), reopened = reopened + 1; end
                obj.Expanded_(id) = true;
                if obj.RecordTrace
                    trace(end+1, :) = [expansions, obj.G_(id), k(1)]; %#ok<AGROW>
                    traceKeys(end+1, 1) = string(model.key(obj.States_{id})); %#ok<AGROW>
                end
                s = obj.States_{id};
                for a = model.actions(s)
                    o = OceanAlgo.StateSearch.checkOutcomes(model.outcomes(s, a), true);
                    if o.Reward > 0
                        error('OceanAlgo:StateSearch:PositiveReward', ...
                            'A* needs reward <= 0 (cost = -reward >= 0); got reward %g.', o.Reward);
                    end
                    g2 = obj.G_(id) - o.Reward;
                    if o.Terminated
                        if sink == 0, sink = obj.add_(o.Next, Inf, NaN, 0, NaN); end
                        child = sink; hc = 0;
                    else
                        ck = model.key(o.Next);
                        child = index.get(ck);
                        if child == 0
                            child = obj.add_(o.Next, Inf, NaN, 0, NaN);
                            index.set(ck, child);
                        end
                        hc = obj.h_(o.Next);
                    end
                    if g2 < obj.G_(child)
                        obj.G_(child) = g2; obj.Parent_(child) = id; obj.Action_(child) = a;
                        obj.Reward_(child) = o.Reward;
                        if child == sink, obj.States_{sink} = o.Next; end
                        pushes = pushes + 1; generated = generated + 1;
                        open.push(child, [g2 + obj.Weight * hc, -g2, pushes]);
                    end
                end
                maxFrontier = max(maxFrontier, open.count());
            end

            work = struct('Expansions', expansions, 'Generated', generated, 'Reopened', reopened, ...
                'MaxFrontier', maxFrontier, 'DistinctStates', index.count() + (sink > 0));
            tr = [];
            if obj.RecordTrace
                tr = table(trace(:, 1), traceKeys, trace(:, 2), trace(:, 3), 'VariableNames', {'Order', 'Key', 'G', 'F'});
            end
            if goalId == 0
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_path", ...
                    'Actions', rootActions, 'Work', work, 'Trace', tr);
                return
            end
            path = obj.path_(goalId, goalId == sink);
            planValue = sum(path.Rewards); if status ~= "ok", planValue = NaN; end
            values = nan(1, numel(rootActions));
            values(rootActions == path.Actions(1)) = planValue;
            result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, ...
                'Action', path.Actions(1), 'Value', planValue, 'Actions', rootActions, ...
                'ActionValues', values, 'Path', path, 'Work', work, 'Trace', tr);
        end
    end

    methods (Access = private)
        function v = h_(obj, s)
            v = obj.Heuristic(s);
            if ~(isscalar(v) && isreal(v) && isfinite(v))
                error('OceanAlgo:StateSearch:BadHeuristic', 'The heuristic must return a finite real scalar.');
            end
        end
        function reset_(obj, cap)
            obj.States_ = cell(cap, 1); obj.Parent_ = zeros(cap, 1); obj.Action_ = nan(cap, 1);
            obj.Reward_ = zeros(cap, 1); obj.G_ = inf(cap, 1); obj.Expanded_ = false(cap, 1); obj.N_ = 0;
        end
        function id = add_(obj, s, g, a, r, parent)
            if obj.N_ == numel(obj.Parent_)                       % capacity doubling
                n = numel(obj.Parent_);
                obj.States_(end+1:2*n) = {[]}; obj.Parent_(end+1:2*n) = 0; obj.Action_(end+1:2*n) = NaN;
                obj.Reward_(end+1:2*n) = 0; obj.G_(end+1:2*n) = Inf; obj.Expanded_(end+1:2*n) = false;
                obj.States_ = obj.States_(:); obj.Parent_ = obj.Parent_(:); obj.Action_ = obj.Action_(:);
                obj.Reward_ = obj.Reward_(:); obj.G_ = obj.G_(:); obj.Expanded_ = obj.Expanded_(:);
            end
            obj.N_ = obj.N_ + 1; id = obj.N_;
            obj.States_{id} = s; obj.G_(id) = g; obj.Action_(id) = a; obj.Reward_(id) = r;
            if isnan(parent), parent = 0; end
            obj.Parent_(id) = parent;
        end
        function p = path_(obj, id, terminated)
            ids = id;
            while obj.Parent_(ids(1)) > 0, ids = [obj.Parent_(ids(1)), ids]; end %#ok<AGROW>
            p = struct('States', {obj.States_(ids)'}, 'Actions', obj.Action_(ids(2:end))', ...
                'Rewards', obj.Reward_(ids(2:end))', 'Terminated', terminated);
        end
    end
end
