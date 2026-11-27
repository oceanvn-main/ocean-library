classdef DepthLimitedSearch < handle
    % DEPTHLIMITEDSEARCH  Heuristic search as decision-time planning (Sutton & Barto 2nd ed., Section 8.9):
    % a full-width tree of all continuations up to Depth actions, leaf values from an approximate value
    % function, backed up with expected updates with maxes toward the root.
    %
    %   engine = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 3, 'LeafValue', @(s) v, ...
    %                'DiscountFactor', 1, 'Transpositions', true, 'MaxExpansions', Inf, 'RecordBackups', false)
    %   result = engine.search(model, s0)          model: OceanAlgo.StateSearch.IDistributionModel
    %
    %   V_0(s)   = h(s)                                              h = LeafValue (never changed by the search)
    %   q_d(s,a) = Σ_{s',r} p(s',r | s,a) [ r + γ V_{d-1}(s') ]      V(s') := 0 after a terminated transition or
    %                                                                at a goal state (episode over)
    %   V_d(s)   = max_a q_d(s,a)                                    V_d(s) := h(s) if s has no legal action
    %   A_t      = argmax_a q_Depth(S_t, a)                          ties -> first in model.actions order
    %
    % Backing up stops at the root's state-action nodes; all backed-up values are discarded after the call.
    % Depth = 1 is the greedy action with respect to h (one-step lookahead). With an exact model, q_Depth(S_t,.)
    % equals the root row after Depth synchronous value-iteration sweeps started from h, but only states
    % reachable within Depth steps of S_t are touched. Searching to the end of the episode removes the effect
    % of an imperfect h; a depth k with γ^k small gives near-optimal actions (S&B 8.9).
    %
    % Not in the text (announced): 'Transpositions' = true stores V_d(s) per (state key, d) during one call, so
    % a state reached along several paths is expanded once per remaining depth. V_d depends only on (s, d), so
    % the result is identical; only the work changes (exponential -> at most Depth x reachable states).
    % Budget: with a finite MaxExpansions the search deepens iteratively, d = 1, 2, ..., Depth (reusing the
    % transposition table), and returns the deepest completed depth (Work.CompletedDepth) with status
    % "budget_exhausted" if Depth was not completed. MaxExpansions must allow the root expansion (>= 1).
    %
    % Work: Expansions (state nodes backed up), LeafEvaluations, TranspositionHits, CompletedDepth.
    % Trace (RecordBackups): table of one-step backups in the order a depth-first search performs them
    % (S&B Figure 8.9): Key, Remaining (= d), Action, Value (= V_d(s)).
    properties (SetAccess = private)
        Depth (1,1) double = 1
        LeafValue (1,1) function_handle = @(s) 0
        DiscountFactor (1,1) double = 1
        Transpositions (1,1) logical = true
        MaxExpansions (1,1) double = Inf
        RecordBackups (1,1) logical = false
    end
    properties (Access = private)
        Model_
        Memo_ cell                  % Memo_{d}: StateTable key -> slot; MemoV_{d}(slot) = V_d
        MemoV_ cell
        Expanded_ (1,1) double = 0
        Leaves_ (1,1) double = 0
        Hits_ (1,1) double = 0
        Out_ (1,1) logical = false
        Log_ cell = cell(0, 4)
    end

    methods
        function obj = DepthLimitedSearch(opts)
            arguments
                opts.Depth (1,1) double {mustBeInteger, mustBePositive} = 1
                opts.LeafValue (1,1) function_handle = @(s) 0
                opts.DiscountFactor (1,1) double {mustBeInRange(opts.DiscountFactor, 0, 1)} = 1
                opts.Transpositions (1,1) logical = true
                opts.MaxExpansions (1,1) double {mustBeGreaterThanOrEqual(opts.MaxExpansions, 1)} = Inf
                opts.RecordBackups (1,1) logical = false
            end
            names = fieldnames(opts);
            for k = 1:numel(names), obj.(names{k}) = opts.(names{k}); end
        end

        function result = search(obj, model, s0)
            arguments
                obj
                model (1,1) OceanAlgo.StateSearch.IDistributionModel
                s0
            end
            name = "depth-limited search";
            rootActions = model.actions(s0);
            if model.isGoal(s0)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "at_goal", 'Value', 0, ...
                    'Actions', rootActions); return
            end
            if isempty(rootActions)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_legal_action", ...
                    'Value', obj.leaf_(s0), 'Actions', rootActions); return
            end
            obj.Model_ = model; obj.Expanded_ = 0; obj.Leaves_ = 0; obj.Hits_ = 0; obj.Out_ = false;
            obj.Memo_ = cell(obj.Depth, 1); obj.MemoV_ = cell(obj.Depth, 1);
            for d = 1:obj.Depth, obj.Memo_{d} = OceanAlgo.StateSearch.StateTable(); obj.MemoV_{d} = zeros(0, 1); end
            obj.Log_ = cell(0, 4);

            if isinf(obj.MaxExpansions), depths = obj.Depth; else, depths = 1:obj.Depth; end
            q = nan(1, numel(rootActions)); completed = 0;
            for D = depths
                qD = nan(1, numel(rootActions));
                for k = 1:numel(rootActions)
                    qD(k) = obj.actionValue_(s0, rootActions(k), D);
                    if obj.Out_, break; end
                end
                if obj.Out_, break; end
                obj.Expanded_ = obj.Expanded_ + 1;                     % the root backup itself
                q = qD; completed = D;
                [~, best] = obj.greedy_(q);
                obj.log_(model.key(s0), D, rootActions(best), q(best));
            end
            [value, best] = obj.greedy_(q);
            status = "ok"; if completed < obj.Depth, status = "budget_exhausted"; end
            work = struct('Expansions', obj.Expanded_, 'LeafEvaluations', obj.Leaves_, ...
                'TranspositionHits', obj.Hits_, 'CompletedDepth', completed);
            tr = [];
            if obj.RecordBackups
                tr = cell2table(obj.Log_, 'VariableNames', {'Key', 'Remaining', 'Action', 'Value'});
            end
            result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, ...
                'Action', rootActions(best), 'Value', value, 'Actions', rootActions, 'ActionValues', q, ...
                'Work', work, 'Trace', tr);
            obj.Model_ = []; obj.Memo_ = {}; obj.MemoV_ = {};       % backed-up values are discarded
        end
    end

    methods (Access = private)
        function v = value_(obj, s, d)
            if d == 0
                obj.Leaves_ = obj.Leaves_ + 1; v = obj.leaf_(s); return
            end
            acts = obj.Model_.actions(s);
            if isempty(acts)
                obj.Leaves_ = obj.Leaves_ + 1; v = obj.leaf_(s); return
            end
            k = obj.Model_.key(s);
            if obj.Transpositions
                slot = obj.Memo_{d}.get(k);
                if slot > 0
                    obj.Hits_ = obj.Hits_ + 1; v = obj.MemoV_{d}(slot); return
                end
            end
            if obj.Expanded_ >= obj.MaxExpansions
                obj.Out_ = true; v = NaN; return
            end
            obj.Expanded_ = obj.Expanded_ + 1;
            q = nan(1, numel(acts));
            for i = 1:numel(acts)
                q(i) = obj.actionValue_(s, acts(i), d);
                if obj.Out_, v = NaN; return; end
            end
            [v, best] = obj.greedy_(q);
            if obj.Transpositions
                obj.MemoV_{d}(end+1, 1) = v; obj.Memo_{d}.set(k, numel(obj.MemoV_{d}));
            end
            obj.log_(k, d, acts(best), v);
        end

        function q = actionValue_(obj, s, a, d)
            o = OceanAlgo.StateSearch.checkOutcomes(obj.Model_.outcomes(s, a));
            q = 0;
            for k = 1:numel(o)
                c = 0;
                if ~o(k).Terminated && ~obj.Model_.isGoal(o(k).Next)
                    c = obj.value_(o(k).Next, d - 1);
                    if obj.Out_, q = NaN; return; end
                end
                q = q + o(k).Probability * (o(k).Reward + obj.DiscountFactor * c);
            end
        end

        function [v, i] = greedy_(~, q)
            v = max(q); i = find(q >= v - 1e-12 * max(1, abs(v)), 1, 'first');
        end
        function v = leaf_(obj, s)
            v = obj.LeafValue(s);
            if ~(isscalar(v) && isreal(v) && isfinite(v))
                error('OceanAlgo:StateSearch:BadLeafValue', 'LeafValue must return a finite real scalar.');
            end
        end
        function log_(obj, k, d, a, v)
            if obj.RecordBackups, obj.Log_(end+1, :) = {string(k), d, a, v}; end
        end
    end
end
