classdef MonteCarloTreeSearch < handle
    % MONTECARLOTREESEARCH  Monte Carlo Tree Search (Sutton & Barto 2nd ed., Section 8.11; UCB tree policy:
    % Kocsis & Szepesvári 2006, "Bandit based Monte-Carlo planning").
    %
    %   engine = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 1000, 'ExplorationConstant', sqrt(2), ...
    %                'MaxDepth', 100, 'RolloutPolicy', OceanAlgo.StateSearch.UniformRandomRolloutPolicy(), ...
    %                'LeafValue', @(s) 0, 'DiscountFactor', 1, 'Seed', 1, 'FinalSelection', "most_visited", ...
    %                'RecordIterations', false)
    %   result = engine.search(model, s0)      model: ISampleModel (an IDistributionModel is sampled)
    %   kept   = engine.advance(action, s1)    keep the subtree below (root, action, s1) for the next search
    %   tree   = engine.getTree()              inspection snapshot of the current tree
    %
    % Each iteration (S&B Figure 8.10):
    %   1. Selection  from the root, the tree policy picks edges by UCB until it leaves the tree:
    %                 untried edges first (random among them), else argmax Q(s,a) + c sqrt( ln N(s) / N(s,a) )
    %   2. Expansion  the first state reached outside the tree is added as one new node
    %   3. Simulation from the new node the rollout policy plays on (rolloutReturn) until the episode ends or
    %                 the total depth reaches MaxDepth, where the return is corrected by γ^k h(S_k), h = LeafValue
    %   4. Backup     along the in-tree path, from the last edge to the root:
    %                 G <- R + γ G;  N(s,a) <- N(s,a) + 1;  Q(s,a) <- Q(s,a) + (G - Q(s,a)) / N(s,a)
    %                 (no values are stored for states visited by the rollout policy)
    % After NumIterations iterations the root action is the most visited edge (ties -> larger Q, then first)
    % or, with FinalSelection "max_value", the edge with the largest Q (both offered by the text).
    %
    % Not in the text (announced): stochastic models -- an edge (node, a) keeps one child node per sampled next
    % state (identified by model.key), i.e. the edge acts as a chance node. A terminated transition or a goal
    % state ends the iteration with no continuation value and creates no node. ExplorationConstant is scale
    % dependent: c = sqrt(2) suits returns in [0, 1]; scale c to the spread of returns of the problem.
    %
    % Tree reuse (S&B 8.11: "often starting with a tree containing any descendants of this node"): after the
    % real environment moves to s1 by `action`, advance(action, s1) re-roots the tree at that child (the rest is
    % discarded) and the next search(model, s1) continues from it. Without advance, every search starts fresh.
    % Reused nodes keep the legal-action sets and edge statistics computed when they were created, so reuse
    % assumes the planning model did not change between the two searches; with a model that is still learning,
    % skip advance (DecisionTimeAgent 'ReuseTree', false) to start every search from a fresh tree.
    %
    % Randomness: the engine owns RandStream('mrg32k3a', Seed); successive calls continue the stream.
    % Status: "ok", "no_legal_action", "at_goal". (The iteration count is the budget; it always completes.)
    % ActionValues = root Q (NaN for unvisited edges), ActionVisits = root N(s,a), Value = visit-weighted root Q.
    % Work: Iterations, NodeCount, MaxTreeDepth, SimulatedSteps, ReusedVisits.
    % Trace (RecordIterations): struct array per iteration: Nodes, Actions, Rewards, ExpandedNode, Tail, Return.
    properties (SetAccess = private)
        NumIterations (1,1) double = 1000
        ExplorationConstant (1,1) double = sqrt(2)
        MaxDepth (1,1) double = 100
        RolloutPolicy
        LeafValue (1,1) function_handle = @(s) 0
        DiscountFactor (1,1) double = 1
        FinalSelection (1,1) string = "most_visited"
        RecordIterations (1,1) logical = false
        Stream
    end
    properties (Access = private)
        Rule_
        Model_
        State_ cell
        Key_ cell              % char key of each node's state
        Parent_ double
        ParentEdge_ double
        Depth_ double
        Visits_ double
        Actions_ cell
        EdgeN_ cell
        EdgeQ_ cell
        Children_ cell
        N_ (1,1) double = 0
        ChildIndex_            % StateTable: 'parent|edge|childKey' -> node id
        Reuse_ (1,1) logical = false
        Steps_ (1,1) double = 0
    end

    methods
        function obj = MonteCarloTreeSearch(opts)
            arguments
                opts.NumIterations (1,1) double {mustBeInteger, mustBePositive} = 1000
                opts.ExplorationConstant (1,1) double {mustBeNonnegative, mustBeFinite} = sqrt(2)
                opts.MaxDepth (1,1) double {mustBeInteger, mustBePositive} = 100
                opts.RolloutPolicy (1,1) OceanAlgo.StateSearch.IRolloutPolicy = OceanAlgo.StateSearch.UniformRandomRolloutPolicy()
                opts.LeafValue (1,1) function_handle = @(s) 0
                opts.DiscountFactor (1,1) double {mustBeInRange(opts.DiscountFactor, 0, 1)} = 1
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 1
                opts.FinalSelection (1,1) string {mustBeMember(opts.FinalSelection, ["most_visited", "max_value"])} = "most_visited"
                opts.RecordIterations (1,1) logical = false
            end
            obj.NumIterations = opts.NumIterations; obj.ExplorationConstant = opts.ExplorationConstant;
            obj.MaxDepth = opts.MaxDepth; obj.RolloutPolicy = opts.RolloutPolicy; obj.LeafValue = opts.LeafValue;
            obj.DiscountFactor = opts.DiscountFactor; obj.FinalSelection = opts.FinalSelection;
            obj.RecordIterations = opts.RecordIterations;
            obj.Stream = RandStream('mrg32k3a', 'Seed', opts.Seed);
            obj.Rule_ = OceanAlgo.StateSearch.UCB1Selection(opts.ExplorationConstant);
            obj.clearTree_();
        end

        function result = search(obj, model, s0)
            arguments
                obj
                model (1,1) OceanAlgo.StateSearch.ISearchModel
                s0
            end
            name = "MCTS";
            model = OceanAlgo.StateSearch.RolloutSearch.asSampleModel(model);
            acts = model.actions(s0);
            if model.isGoal(s0)
                obj.clearTree_();
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "at_goal", 'Value', 0, ...
                    'Actions', acts); return
            end
            if isempty(acts)
                obj.clearTree_();
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_legal_action", ...
                    'Actions', acts); return
            end
            rootKey = obj.keyText_(model.key(s0));
            if ~(obj.Reuse_ && obj.N_ >= 1 && strcmp(obj.Key_{1}, rootKey))
                obj.clearTree_();
                obj.add_(s0, rootKey, acts, 0, 0, 0);
            end
            obj.Reuse_ = false;
            obj.Model_ = model;
            obj.grow_(obj.NumIterations);
            reused = obj.Visits_(1); obj.Steps_ = 0;
            if obj.RecordIterations
                trace = repmat(struct('Nodes', [], 'Actions', [], 'Rewards', [], 'ExpandedNode', 0, ...
                    'Tail', "", 'Return', 0), obj.NumIterations, 1);
            end
            for it = 1:obj.NumIterations
                rec = obj.iterate_();
                if obj.RecordIterations, trace(it) = rec; end
            end

            N = obj.EdgeN_{1}; Q = obj.EdgeQ_{1};
            visited = find(N > 0);
            if obj.FinalSelection == "most_visited"
                [~, order] = sortrows([-N(visited)', -Q(visited)', visited']);   % most visits, then larger Q, then first
                pick = visited(order(1));
            else
                best = max(Q(visited));
                pick = visited(find(Q(visited) >= best - 1e-12 * max(1, abs(best)), 1, 'first'));
            end
            values = Q; values(N == 0) = NaN;
            work = struct('Iterations', obj.NumIterations, 'NodeCount', obj.N_, ...
                'MaxTreeDepth', max(obj.Depth_(1:obj.N_)), 'SimulatedSteps', obj.Steps_, 'ReusedVisits', reused);
            tr = []; if obj.RecordIterations, tr = trace; end
            result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "ok", ...
                'Action', obj.Actions_{1}(pick), 'Value', sum(N .* Q) / sum(N), 'Actions', obj.Actions_{1}, ...
                'ActionValues', values, 'ActionVisits', N, 'Work', work, 'Trace', tr);
        end

        function kept = advance(obj, action, s1)
            % ADVANCE  Re-root the tree at the child reached by (action, s1); returns false (and clears the tree)
            % when that child is not in the tree.
            kept = false;
            if obj.N_ == 0 || isempty(obj.Model_), obj.clearTree_(); return; end
            e = find(obj.Actions_{1} == action, 1);
            child = 0;
            if ~isempty(e)
                child = obj.ChildIndex_.get(obj.childKey_(1, e, obj.keyText_(obj.Model_.key(s1))));
            end
            if child == 0, obj.clearTree_(); return; end
            obj.reroot_(child);
            obj.Reuse_ = true; kept = true;
        end

        function t = getTree(obj)
            % GETTREE  Snapshot: one row per node (Key, Parent, ParentAction, Depth, Visits) plus per-node edges.
            n = obj.N_;
            pa = nan(n, 1);
            for i = 2:n, pa(i) = obj.Actions_{obj.Parent_(i)}(obj.ParentEdge_(i)); end
            t = struct('Key', {obj.Key_(1:n)}, 'Parent', obj.Parent_(1:n), 'ParentAction', pa, ...
                'Depth', obj.Depth_(1:n), 'Visits', obj.Visits_(1:n), 'Actions', {obj.Actions_(1:n)}, ...
                'EdgeVisits', {obj.EdgeN_(1:n)}, 'EdgeValues', {obj.EdgeQ_(1:n)});
        end
    end

    methods (Access = private)
        function rec = iterate_(obj)
            model = obj.Model_; gamma = obj.DiscountFactor;
            node = 1; depth = 0;
            nodes = zeros(1, 0); edges = zeros(1, 0); rewards = zeros(1, 0); expanded = 0;
            while true
                s = obj.State_{node};
                if isempty(obj.Actions_{node})
                    tail = obj.leaf_(s); why = "no_action"; break
                end
                if depth >= obj.MaxDepth
                    tail = obj.leaf_(s); why = "horizon"; break
                end
                e = obj.Rule_.select(obj.EdgeQ_{node}, obj.EdgeN_{node}, obj.Stream);       % 1. selection
                o = model.sample(s, obj.Actions_{node}(e), obj.Stream);
                obj.Steps_ = obj.Steps_ + 1; depth = depth + 1;
                nodes(end+1) = node; edges(end+1) = e; rewards(end+1) = o.Reward; %#ok<AGROW>
                if o.Terminated || model.isGoal(o.Next)
                    tail = 0; why = "terminated"; break
                end
                ck = obj.childKey_(node, e, obj.keyText_(model.key(o.Next)));
                child = obj.ChildIndex_.get(ck);
                if child == 0                                                                % 2. expansion
                    child = obj.add_(o.Next, obj.keyText_(model.key(o.Next)), model.actions(o.Next), node, e, depth);
                    obj.ChildIndex_.set(ck, child); expanded = child;
                    [tail, st, why] = OceanAlgo.StateSearch.rolloutReturn(model, o.Next, NaN, ...    % 3. simulation
                        obj.RolloutPolicy, obj.Stream, gamma, obj.MaxDepth - depth, obj.LeafValue);
                    obj.Steps_ = obj.Steps_ + st;
                    break
                end
                node = child;
            end
            G = tail;                                                                        % 4. backup
            for i = numel(nodes):-1:1
                G = rewards(i) + gamma * G;
                n = nodes(i); e = edges(i);
                obj.EdgeN_{n}(e) = obj.EdgeN_{n}(e) + 1;
                obj.EdgeQ_{n}(e) = obj.EdgeQ_{n}(e) + (G - obj.EdgeQ_{n}(e)) / obj.EdgeN_{n}(e);
                obj.Visits_(n) = obj.Visits_(n) + 1;
            end
            rec = struct('Nodes', nodes, 'Actions', arrayfun(@(i) obj.Actions_{nodes(i)}(edges(i)), 1:numel(nodes)), ...
                'Rewards', rewards, 'ExpandedNode', expanded, 'Tail', why, 'Return', G);
        end

        function v = leaf_(obj, s)
            v = obj.LeafValue(s);
            if ~(isscalar(v) && isreal(v) && isfinite(v))
                error('OceanAlgo:StateSearch:BadLeafValue', 'LeafValue must return a finite real scalar.');
            end
        end

        function id = add_(obj, s, key, acts, parent, edge, depth)
            obj.N_ = obj.N_ + 1; id = obj.N_;
            if id > numel(obj.Parent_), obj.grow_(max(16, numel(obj.Parent_))); end
            obj.State_{id} = s; obj.Key_{id} = key; obj.Parent_(id) = parent; obj.ParentEdge_(id) = edge;
            obj.Depth_(id) = depth; obj.Visits_(id) = 0; obj.Actions_{id} = acts;
            obj.EdgeN_{id} = zeros(1, numel(acts)); obj.EdgeQ_{id} = zeros(1, numel(acts));
            obj.Children_{id} = zeros(1, 0);
            if parent > 0, obj.Children_{parent}(end+1) = id; end
        end

        function grow_(obj, extra)
            % Preallocate room for `extra` more nodes (one node per iteration at most).
            need = obj.N_ + extra; have = numel(obj.Parent_);
            if need <= have, return; end
            obj.State_(have+1:need, 1) = {[]}; obj.Key_(have+1:need, 1) = {''};
            obj.Parent_(have+1:need, 1) = 0; obj.ParentEdge_(have+1:need, 1) = 0; obj.Depth_(have+1:need, 1) = 0;
            obj.Visits_(have+1:need, 1) = 0; obj.Actions_(have+1:need, 1) = {[]};
            obj.EdgeN_(have+1:need, 1) = {[]}; obj.EdgeQ_(have+1:need, 1) = {[]}; obj.Children_(have+1:need, 1) = {[]};
        end

        function clearTree_(obj)
            obj.State_ = cell(0, 1); obj.Key_ = cell(0, 1); obj.Parent_ = zeros(0, 1); obj.ParentEdge_ = zeros(0, 1);
            obj.Depth_ = zeros(0, 1); obj.Visits_ = zeros(0, 1); obj.Actions_ = cell(0, 1);
            obj.EdgeN_ = cell(0, 1); obj.EdgeQ_ = cell(0, 1); obj.Children_ = cell(0, 1);
            obj.N_ = 0; obj.ChildIndex_ = OceanAlgo.StateSearch.StateTable(); obj.Reuse_ = false;
        end

        function reroot_(obj, root)
            % Keep only the subtree below `root`, renumbered breadth-first (root = 1, depths shifted).
            order = root; head = 1;
            while head <= numel(order)
                order = [order, obj.Children_{order(head)}]; head = head + 1; %#ok<AGROW>
            end
            newId = zeros(obj.N_, 1); newId(order) = 1:numel(order);
            base = obj.Depth_(root);
            St = obj.State_(order); K = obj.Key_(order); P = obj.Parent_(order); PE = obj.ParentEdge_(order);
            D = obj.Depth_(order) - base; V = obj.Visits_(order); A = obj.Actions_(order);
            EN = obj.EdgeN_(order); EQ = obj.EdgeQ_(order); C = obj.Children_(order);
            obj.clearTree_();
            n = numel(order);
            obj.State_ = St; obj.Key_ = K; obj.Depth_ = D; obj.Visits_ = V; obj.Actions_ = A;
            obj.EdgeN_ = EN; obj.EdgeQ_ = EQ; obj.N_ = n;
            obj.Parent_ = zeros(n, 1); obj.ParentEdge_ = zeros(n, 1); obj.Children_ = cell(n, 1);
            for i = 1:n
                obj.Children_{i} = newId(C{i})';
                if i > 1
                    obj.Parent_(i) = newId(P(i)); obj.ParentEdge_(i) = PE(i);
                    obj.ChildIndex_.set(obj.childKey_(obj.Parent_(i), PE(i), K{i}), i);
                end
            end
        end
    end
    methods (Static, Access = private)
        function t = keyText_(k)
            if isnumeric(k), t = sprintf('%.17g', k); else, t = char(k); end
        end
        function k = childKey_(parent, edge, childKey)
            k = sprintf('%d|%d|%s', parent, edge, childKey);
        end
    end
end
