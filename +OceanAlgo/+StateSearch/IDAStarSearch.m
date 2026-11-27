classdef IDAStarSearch < handle
    % IDASTARSEARCH  Iterative-deepening A* (Korf 1985, "Depth-first iterative-deepening: an optimal
    % admissible tree search"; Russell & Norvig ch. 3) on a deterministic planning model.
    %
    %   engine = OceanAlgo.StateSearch.IDAStarSearch('Heuristic', @(s) h, 'MaxExpansions', 1e5)
    %   result = engine.search(model, s0)          model: OceanAlgo.StateSearch.IDistributionModel, deterministic
    %
    % Same cost convention as AStarSearch: c(s,a) = -reward >= 0, h(s) = estimated remaining cost, goal = a
    % terminated transition or model.isGoal(s).
    %
    %   threshold_0 = h(s0)
    %   iteration i: depth-first search that prunes every node with f = g + h > threshold_i
    %   threshold_{i+1} = the smallest f that was pruned in iteration i
    %
    % The first goal reached with f <= threshold is returned. With admissible h its cost is optimal. Memory is
    % proportional to the path length; nodes are re-expanded in every iteration (no closed set). States already
    % on the current path are skipped (cycle check by model.key). Children are tried in model.actions order.
    % The depth-first recursion is bounded by MATLAB's RecursionLimit (default 500 path steps).
    %
    % Status: "ok", "no_path" (no unpruned node left: the goal is unreachable), "at_goal", "no_legal_action",
    % "budget_exhausted" (MaxExpansions reached; Path = the depth-first path being explored, Action = its first
    % action, or the first legal action when the path is empty -- legal, but not a certified plan; Value = NaN).
    %
    % Work: Expansions, Iterations, Thresholds (1 x Iterations).
    properties (SetAccess = private)
        Heuristic (1,1) function_handle = @(s) 0
        MaxExpansions (1,1) double = 1e5
    end
    properties (Access = private)
        Model_
        OnPath_
        PathStates_ cell
        PathActions_ double
        PathRewards_ double
        PathTerminated_ (1,1) logical = false
        Expansions_ (1,1) double = 0
        OutOfBudget_ (1,1) logical = false
    end

    methods
        function obj = IDAStarSearch(opts)
            arguments
                opts.Heuristic (1,1) function_handle = @(s) 0
                opts.MaxExpansions (1,1) double {mustBePositive} = 1e5
            end
            obj.Heuristic = opts.Heuristic; obj.MaxExpansions = opts.MaxExpansions;
        end

        function result = search(obj, model, s0)
            arguments
                obj
                model (1,1) OceanAlgo.StateSearch.IDistributionModel
                s0
            end
            name = "IDA*";
            rootActions = model.actions(s0);
            if model.isGoal(s0)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "at_goal", 'Value', 0, ...
                    'Actions', rootActions); return
            end
            if isempty(rootActions)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_legal_action", ...
                    'Actions', rootActions); return
            end
            obj.Model_ = model; obj.Expansions_ = 0; obj.OutOfBudget_ = false;
            threshold = obj.h_(s0); thresholds = zeros(1, 0); status = "no_path";
            while true
                thresholds(end+1) = threshold; %#ok<AGROW>
                obj.OnPath_ = OceanAlgo.StateSearch.StateTable(); obj.OnPath_.set(model.key(s0), 1);
                obj.PathStates_ = {s0}; obj.PathActions_ = zeros(1, 0); obj.PathRewards_ = zeros(1, 0);
                obj.PathTerminated_ = false;
                [found, next] = obj.dfs_(s0, 0, threshold);
                if found, status = "ok"; break; end
                if obj.OutOfBudget_, status = "budget_exhausted"; break; end
                if isinf(next), break; end
                threshold = next;
            end

            work = struct('Expansions', obj.Expansions_, 'Iterations', numel(thresholds), 'Thresholds', thresholds);
            obj.Model_ = []; obj.OnPath_ = [];
            if status == "no_path"
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, ...
                    'Actions', rootActions, 'Work', work);
                return
            end
            path = struct('States', {obj.PathStates_}, 'Actions', obj.PathActions_, ...
                'Rewards', obj.PathRewards_, 'Terminated', obj.PathTerminated_);
            if isempty(path.Actions), action = rootActions(1); else, action = path.Actions(1); end
            planValue = sum(path.Rewards); if status ~= "ok", planValue = NaN; end
            values = nan(1, numel(rootActions));
            if status == "ok", values(rootActions == action) = sum(path.Rewards); end
            result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, 'Action', action, ...
                'Value', planValue, 'Actions', rootActions, 'ActionValues', values, 'Path', path, 'Work', work);
        end
    end

    methods (Access = private)
        function [found, next] = dfs_(obj, s, g, threshold)
            % Returns found = true with the plan on the path stack, else the smallest pruned f below s.
            found = false;
            f = g + obj.h_(s);
            if f > threshold + obj.tol_(threshold)
                next = f; return
            end
            if obj.Model_.isGoal(s)
                found = true; next = f; return
            end
            if obj.Expansions_ >= obj.MaxExpansions
                obj.OutOfBudget_ = true; next = Inf; return
            end
            obj.Expansions_ = obj.Expansions_ + 1;
            next = Inf;
            for a = obj.Model_.actions(s)
                o = OceanAlgo.StateSearch.checkOutcomes(obj.Model_.outcomes(s, a), true);
                if o.Reward > 0
                    error('OceanAlgo:StateSearch:PositiveReward', ...
                        'IDA* needs reward <= 0 (cost = -reward >= 0); got reward %g.', o.Reward);
                end
                g2 = g - o.Reward;
                if o.Terminated
                    if g2 <= threshold + obj.tol_(threshold)
                        obj.push_(o.Next, a, o.Reward); obj.PathTerminated_ = true;
                        found = true; next = g2; return
                    end
                    next = min(next, g2); continue
                end
                k = obj.Model_.key(o.Next);
                if obj.OnPath_.has(k), continue; end              % cycle along the current path
                obj.push_(o.Next, a, o.Reward); obj.OnPath_.set(k, 1);
                [found, t] = obj.dfs_(o.Next, g2, threshold);
                if found, next = t; return; end
                if obj.OutOfBudget_, return; end
                obj.OnPath_.set(k, 0); obj.pop_();
                next = min(next, t);
            end
        end
        function push_(obj, s, a, r)
            obj.PathStates_{end+1} = s; obj.PathActions_(end+1) = a; obj.PathRewards_(end+1) = r;
        end
        function pop_(obj)
            obj.PathStates_(end) = []; obj.PathActions_(end) = []; obj.PathRewards_(end) = [];
        end
        function v = h_(obj, s)
            v = obj.Heuristic(s);
            if ~(isscalar(v) && isreal(v) && isfinite(v))
                error('OceanAlgo:StateSearch:BadHeuristic', 'The heuristic must return a finite real scalar.');
            end
        end
    end
    methods (Static, Access = private)
        function t = tol_(threshold), t = 1e-9 * max(1, abs(threshold)); end
    end
end
