classdef SearchResult
    % SEARCHRESULT  What one search call decided and how it got there. Immutable value object.
    %
    %   Algorithm       engine name
    %   Status          "ok"               finished normally
    %                   "budget_exhausted" stopped by its work budget; Action is still a legal action, chosen from
    %                                      the incomplete search (see the engine's help)
    %                   "no_path"          (deterministic engines) no goal / terminated transition is reachable
    %                   "no_legal_action"  the root admits no action; Action = NaN
    %                   "at_goal"          the root is already a goal; Action = NaN
    %   Action          chosen action label at the root (NaN when there is none)
    %   Value           engine's estimate of the root value (path return for path engines)
    %   Actions         1 x K legal action labels at the root
    %   ActionValues    1 x K value estimate per root action (NaN: not evaluated)
    %   ActionVisits    1 x K simulations / samples per root action (MCTS, rollout; else [])
    %   Path            struct: States (cell), Actions, Rewards, Terminated -- the chosen plan (path engines);
    %                   empty fields for decision-time engines
    %   Candidates      N-best alternatives (beam search) as Path-like structs with a Score field
    %   Work            struct of work counters (Expansions, Generated, Iterations, SimulatedSteps, ...)
    %   Trace           optional inspection record (engine specific; empty unless requested)
    properties (SetAccess = immutable)
        Algorithm (1,1) string = ""
        Status (1,1) string = "ok"
        Action (1,1) double = NaN
        Value (1,1) double = NaN
        Actions (1,:) double = zeros(1, 0)
        ActionValues (1,:) double = zeros(1, 0)
        ActionVisits (1,:) double = zeros(1, 0)
        Path struct = OceanAlgo.StateSearch.SearchResult.emptyPath()
        Candidates struct = struct([])
        Work struct = struct()
        Trace = []
    end
    properties (Constant)
        Statuses = ["ok", "budget_exhausted", "no_path", "no_legal_action", "at_goal"]
    end
    methods
        function obj = SearchResult(opts)
            arguments
                opts.Algorithm (1,1) string = ""
                opts.Status (1,1) string = "ok"
                opts.Action (1,1) double = NaN
                opts.Value (1,1) double = NaN
                opts.Actions (1,:) double = zeros(1, 0)
                opts.ActionValues (1,:) double = zeros(1, 0)
                opts.ActionVisits (1,:) double = zeros(1, 0)
                opts.Path struct = OceanAlgo.StateSearch.SearchResult.emptyPath()
                opts.Candidates struct = struct([])
                opts.Work struct = struct()
                opts.Trace = []
            end
            if ~ismember(opts.Status, OceanAlgo.StateSearch.SearchResult.Statuses)
                error('OceanAlgo:StateSearch:BadStatus', 'Unknown search status "%s".', opts.Status);
            end
            if ~isempty(opts.ActionValues) && numel(opts.ActionValues) ~= numel(opts.Actions)
                error('OceanAlgo:StateSearch:BadResult', 'ActionValues must align with Actions.');
            end
            names = fieldnames(opts);
            for k = 1:numel(names), obj.(names{k}) = opts.(names{k}); end
        end
    end
    methods (Static)
        function p = emptyPath()
            p = struct('States', {{}}, 'Actions', zeros(1, 0), 'Rewards', zeros(1, 0), 'Terminated', false);
        end
    end
end
