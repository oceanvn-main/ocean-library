classdef BeamSearch < handle
    % BEAMSEARCH  Beam search over action sequences on a deterministic planning model (local beam search:
    % Lowerre 1976; Russell & Norvig ch. 4), with optional diverse beam search groups (Vijayakumar et al. 2018,
    % "Diverse Beam Search" -- not in the RL text, announced).
    %
    %   engine = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 4, 'MaxDepth', 10, 'LeafValue', @(s) 0, ...
    %                'DiscountFactor', 1, 'NBest', 1, 'MergeDuplicates', true, ...
    %                'Groups', 1, 'DiversityStrength', 0, 'MaxExpansions', Inf)
    %   result = engine.search(model, s0)          model: OceanAlgo.StateSearch.IDistributionModel, deterministic
    %
    % A hypothesis is an action sequence a_1..a_d from s0 with discounted return G_d = Σ_{i<=d} γ^(i-1) r_i.
    % Its score (rewards are maximised, LeafValue h = estimated value of the remaining episode):
    %
    %   live      score = G_d + γ^d h(s_d)
    %   finished  score = G_d                  terminated transition or goal state reached (no continuation)
    %             score = G_d + γ^d h(s_d)     s_d has no legal action (valued h, as everywhere in this package)
    %   truncated score = G_D + γ^D h(s_D)     still live at MaxDepth
    %
    % Each depth: every live hypothesis is expanded by every legal action; finished children leave the beam;
    % with MergeDuplicates, candidates reaching the same state (model.key) at the same depth keep only the best
    % (a state's future does not depend on how it was reached, so this never loses the best plan); the BeamWidth
    % best candidates by score stay live (ties -> earlier beam row, then model.actions order). The search stops
    % when no hypothesis is live or MaxDepth is reached. The answer is the best finished/truncated hypothesis;
    % Candidates holds the NBest best. With BeamWidth = 1 the search follows the one-step lookahead
    % argmax_a [ r + γ h(s') ]; with BeamWidth >= the number of hypotheses it is exhaustive (exact optimum over
    % all action sequences of length <= MaxDepth, equal to depth-limited search).
    %
    % Diverse beam (Groups G > 1): BeamWidth must be divisible by G; each group keeps BeamWidth / G live
    % hypotheses. Groups are filled in order; at each depth a candidate of group g is ranked by
    %   score - DiversityStrength * (number of earlier groups' selected hypotheses whose last action equals its last action)
    % (Hamming diversity on the current step). DiversityStrength = 0 gives G independent beams. The diversity
    % term only affects selection; reported scores are the undiscounted-by-diversity scores above.
    %
    % Status: "ok", "no_path" (nothing finished, e.g. MaxDepth = 0 is not allowed so this means no legal
    % continuation at all), "at_goal", "no_legal_action", "budget_exhausted" (MaxExpansions hypothesis expansions
    % used; live hypotheses are scored as truncated; if no hypothesis has an action yet, Action is the first
    % legal root action, unplanned, with NaN ActionValues).
    %
    % Work: Expansions (hypotheses expanded), Generated (children scored), Depth (deepest level reached), Merged.
    properties (SetAccess = private)
        BeamWidth (1,1) double
        MaxDepth (1,1) double
        LeafValue (1,1) function_handle = @(s) 0
        DiscountFactor (1,1) double = 1
        NBest (1,1) double = 1
        MergeDuplicates (1,1) logical = true
        Groups (1,1) double = 1
        DiversityStrength (1,1) double = 0
        MaxExpansions (1,1) double = Inf
    end

    methods
        function obj = BeamSearch(opts)
            arguments
                opts.BeamWidth (1,1) double {mustBeInteger, mustBePositive} = 4
                opts.MaxDepth (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.LeafValue (1,1) function_handle = @(s) 0
                opts.DiscountFactor (1,1) double {mustBeInRange(opts.DiscountFactor, 0, 1)} = 1
                opts.NBest (1,1) double {mustBeInteger, mustBePositive} = 1
                opts.MergeDuplicates (1,1) logical = true
                opts.Groups (1,1) double {mustBeInteger, mustBePositive} = 1
                opts.DiversityStrength (1,1) double {mustBeNonnegative, mustBeFinite} = 0
                opts.MaxExpansions (1,1) double {mustBePositive} = Inf
            end
            if mod(opts.BeamWidth, opts.Groups) ~= 0
                error('OceanAlgo:StateSearch:BadGroups', 'BeamWidth (%d) must be divisible by Groups (%d).', ...
                    opts.BeamWidth, opts.Groups);
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
            name = "beam search"; if obj.Groups > 1, name = "diverse beam search"; end
            rootActions = model.actions(s0);
            if model.isGoal(s0)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "at_goal", 'Value', 0, ...
                    'Actions', rootActions); return
            end
            if isempty(rootActions)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_legal_action", ...
                    'Actions', rootActions); return
            end

            gamma = obj.DiscountFactor; width = obj.BeamWidth / obj.Groups;
            root = obj.hyp_({s0}, zeros(1, 0), zeros(1, 0), 0, obj.leaf_(s0), "live");
            live = repmat({root}, 1, obj.Groups);                % every group starts from the root
            finished = repmat(root, 0, 1);
            expansions = 0; generated = 0; merged = 0; depth = 0; status = "ok";
            while depth < obj.MaxDepth && any(cellfun(@numel, live) > 0)
                lastChosen = zeros(1, 0);                        % last actions selected by earlier groups
                for g = 1:obj.Groups
                    cands = repmat(root, 0, 1);
                    for i = 1:numel(live{g})
                        if expansions >= obj.MaxExpansions, status = "budget_exhausted"; break; end
                        h = live{g}(i); s = h.States{end};
                        expansions = expansions + 1;
                        for a = model.actions(s)
                            o = OceanAlgo.StateSearch.checkOutcomes(model.outcomes(s, a), true);
                            G2 = h.G + gamma^depth * o.Reward;
                            states = [h.States, {o.Next}]; acts = [h.Actions, a]; rews = [h.Rewards, o.Reward];
                            generated = generated + 1;
                            if o.Terminated || model.isGoal(o.Next)
                                finished(end+1, 1) = obj.hyp_(states, acts, rews, G2, G2, "finished"); %#ok<AGROW>
                            elseif isempty(model.actions(o.Next))
                                finished(end+1, 1) = obj.hyp_(states, acts, rews, G2, ...
                                    G2 + gamma^(depth+1) * obj.leaf_(o.Next), "no_action"); %#ok<AGROW>
                            else
                                cands(end+1, 1) = obj.hyp_(states, acts, rews, G2, ...
                                    G2 + gamma^(depth+1) * obj.leaf_(o.Next), "live"); %#ok<AGROW>
                            end
                        end
                    end
                    if status == "budget_exhausted"
                        % Unexpanded hypotheses of this and later groups stay live and are scored as truncated.
                        for gg = g:obj.Groups
                            if gg == g, rest = live{g}(i:end); else, rest = live{gg}; end
                            finished = [finished; obj.truncate_(rest)]; %#ok<AGROW>
                            if gg > g, live{gg} = repmat(root, 0, 1); end
                        end
                        finished = [finished; obj.truncate_(cands)]; %#ok<AGROW>
                        live{g} = repmat(root, 0, 1);
                        break
                    end
                    if obj.MergeDuplicates && ~isempty(cands)
                        [cands, m] = obj.merge_(model, cands); merged = merged + m;
                    end
                    if isempty(cands), live{g} = cands; continue; end
                    last = arrayfun(@(c) c.Actions(end), cands);
                    penalty = obj.DiversityStrength * arrayfun(@(x) sum(lastChosen == x), last);
                    [~, order] = sort([cands.Score]' - penalty(:), 'descend');   % stable: ties keep order
                    keep = order(1:min(width, numel(order)));
                    live{g} = cands(keep);
                    lastChosen = [lastChosen, last(keep)']; %#ok<AGROW>
                end
                if status == "budget_exhausted", break; end
                depth = depth + 1;
            end
            for g = 1:obj.Groups
                finished = [finished; obj.truncate_(live{g})]; %#ok<AGROW>
            end

            work = struct('Expansions', expansions, 'Generated', generated, 'Depth', depth, 'Merged', merged);
            if isempty(finished) && status == "budget_exhausted"
                % Budget ran out before any hypothesis got an action: still return a legal (unplanned) action.
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, ...
                    'Action', rootActions(1), 'Actions', rootActions, 'ActionValues', nan(1, numel(rootActions)), ...
                    'Work', work);
                return
            end
            if isempty(finished)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_path", ...
                    'Actions', rootActions, 'Work', work);
                return
            end
            [~, order] = sort([finished.Score]', 'descend');
            finished = finished(order);
            best = finished(1);
            values = nan(1, numel(rootActions));
            for k = 1:numel(rootActions)
                hit = arrayfun(@(f) f.Actions(1) == rootActions(k), finished);
                if any(hit), values(k) = finished(find(hit, 1)).Score; end
            end
            nb = finished(1:min(obj.NBest, numel(finished)));
            cand = arrayfun(@(f) struct('States', {f.States}, 'Actions', f.Actions, 'Rewards', f.Rewards, ...
                'Terminated', f.Kind == "finished", 'Score', f.Score, 'Kind', f.Kind), nb, 'UniformOutput', false);
            cand = [cand{:}];
            path = struct('States', {best.States}, 'Actions', best.Actions, 'Rewards', best.Rewards, ...
                'Terminated', best.Kind == "finished");
            result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, ...
                'Action', best.Actions(1), 'Value', best.Score, 'Actions', rootActions, 'ActionValues', values, ...
                'Path', path, 'Candidates', cand, 'Work', work);
        end
    end

    methods (Access = private)
        function v = leaf_(obj, s)
            v = obj.LeafValue(s);
            if ~(isscalar(v) && isreal(v) && isfinite(v))
                error('OceanAlgo:StateSearch:BadLeafValue', 'LeafValue must return a finite real scalar.');
            end
        end
        function t = truncate_(~, hs)
            % Live hypotheses become "truncated" answers; an unexpanded root (no action yet) is not an answer.
            t = hs(:);
            t = t(arrayfun(@(x) ~isempty(x.Actions), t));
            for k = 1:numel(t)
                if t(k).Kind == "live", t(k).Kind = "truncated"; end
            end
        end
        function [out, m] = merge_(~, model, cands)
            % Keep the best-scoring candidate per state key (first one on ties), preserving order.
            idx = OceanAlgo.StateSearch.StateTable(); keep = false(numel(cands), 1);
            for k = 1:numel(cands)
                key = model.key(cands(k).States{end});
                j = idx.get(key);
                if j == 0
                    idx.set(key, k); keep(k) = true;
                elseif cands(k).Score > cands(j).Score
                    keep(j) = false; keep(k) = true; idx.set(key, k);
                end
            end
            out = cands(keep); m = nnz(~keep);
        end
    end
    methods (Static, Access = private)
        function h = hyp_(states, acts, rews, G, score, kind)
            h = struct('States', {states}, 'Actions', acts, 'Rewards', rews, 'G', G, 'Score', score, ...
                'Kind', string(kind));
        end
    end
end
