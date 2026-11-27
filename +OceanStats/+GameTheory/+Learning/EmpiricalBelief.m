classdef EmpiricalBelief < handle
    % EMPIRICALBELIEF  What one player believes about how each opponent plays: the opponent's empirical mixed
    % strategy, smoothed by a Dirichlet prior (Brown 1951; Fudenberg & Levine 1998, ch. 2):
    %
    %   p_j(s) = (alpha_j(s) + n_j(s)) / (sum_s' alpha_j(s') + t_j)
    %
    % n_j(s) = how often opponent j was seen playing s, t_j = how often j was observed. One count vector per opponent
    % (beliefs are independent across opponents, the classical assumption), so the cost of an update is O(1) and the
    % memory O(sum of strategy counts), for any number of players.
    %
    %   b = OceanStats.GameTheory.Learning.EmpiricalBelief([2 3], 'Prior', 1);   % two opponents, 2 and 3 strategies
    %   b.observe([1 3]);                    % opponent 1 played its strategy 1, opponent 2 its strategy 3
    %   p = b.probabilities(1);             % 1 x 2 belief about opponent 1

    properties (SetAccess = private)
        Counts cell                          % Counts{j}: 1 x m_j observed counts of opponent j
        Prior cell                           % Prior{j}: 1 x m_j Dirichlet pseudo-counts (alpha_j)
        Observations double                  % 1 x J how often each opponent was observed
    end

    methods
        function obj = EmpiricalBelief(numStrategies, options)
            arguments
                numStrategies (1,:) double {mustBeInteger, mustBePositive}
                options.Prior = 1                % scalar for every strategy, or a cell of 1 x m_j vectors
            end
            J = numel(numStrategies);
            obj.Counts = arrayfun(@(m) zeros(1, m), numStrategies, 'UniformOutput', false);
            prior = options.Prior;
            if ~iscell(prior), prior = arrayfun(@(m) repmat(prior, 1, m), numStrategies, 'UniformOutput', false); end
            if numel(prior) ~= J || any(cellfun(@numel, prior) ~= numStrategies) || ...
                    any(cellfun(@(a) any(a < 0) || ~all(isfinite(a)), prior))
                error('OceanStats:GameTheory:BadBelief', ...
                    'Prior must be a non-negative scalar or one 1 x m_j vector per opponent.');
            end
            obj.Prior = cellfun(@(a) reshape(double(a), 1, []), prior, 'UniformOutput', false);
            obj.Observations = zeros(1, J);
        end

        function observe(obj, actions)
            % OBSERVE  One observation of every opponent (NaN = that opponent was not seen this time).
            arguments
                obj
                actions (1,:) double
            end
            if numel(actions) ~= numel(obj.Counts)
                error('OceanStats:GameTheory:BadBelief', 'Expected %d opponent actions, got %d.', numel(obj.Counts), numel(actions));
            end
            for j = find(~isnan(actions))
                s = actions(j);
                if ~(s >= 1 && s <= numel(obj.Counts{j}) && s == fix(s))
                    error('OceanStats:GameTheory:BadBelief', 'Opponent %d has no strategy %g.', j, s);
                end
                obj.Counts{j}(s) = obj.Counts{j}(s) + 1;
                obj.Observations(j) = obj.Observations(j) + 1;
            end
        end

        function p = probabilities(obj, j)
            % PROBABILITIES  The belief about opponent j (uniform when nothing was seen and the prior is zero).
            a = obj.Prior{j} + obj.Counts{j};
            total = sum(a);
            if total == 0, p = ones(size(a)) / numel(a); else, p = a / total; end
        end

        function P = all(obj)
            % ALL  Beliefs about every opponent, one cell per opponent.
            P = arrayfun(@(j) obj.probabilities(j), 1:numel(obj.Counts), 'UniformOutput', false);
        end

        function reset(obj)
            obj.Counts = cellfun(@(c) zeros(size(c)), obj.Counts, 'UniformOutput', false);
            obj.Observations(:) = 0;
        end

        function s = exportState(obj)
            s = struct('Counts', {obj.Counts}, 'Observations', obj.Observations);
        end

        function importState(obj, s)
            % a saved state must fit this belief and be a valid record: whole, non-negative counts per opponent,
            % summing to how often that opponent was observed
            ok = isstruct(s) && isscalar(s) && all(isfield(s, {'Counts', 'Observations'})) && iscell(s.Counts) && ...
                numel(s.Counts) == numel(obj.Counts) && isnumeric(s.Observations) && ...
                numel(s.Observations) == numel(obj.Observations);
            if ok
                ok = all(cellfun(@(c, m) isnumeric(c) && numel(c) == numel(m) && all(isfinite(c)) && all(c >= 0) && ...
                    all(c == fix(c)), s.Counts, obj.Counts)) && ...
                    isequal(reshape(double(s.Observations), 1, []), cellfun(@(c) sum(double(c)), s.Counts));
            end
            if ~ok
                error('OceanStats:GameTheory:BadBelief', ['The state does not fit this belief (Counts: whole, ' ...
                    'non-negative, one row per opponent; Observations: their sums).']);
            end
            obj.Counts = cellfun(@(c) reshape(double(c), 1, []), s.Counts, 'UniformOutput', false);
            obj.Observations = reshape(double(s.Observations), 1, []);
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "p_j(s) = \frac{\alpha_j(s) + n_j(s)}{\sum_{s'} \alpha_j(s') + t_j}";
            meta.Symbols = [struct('Symbol', 'n_j(s)', 'Description', 'times opponent j was seen playing s'), ...
                struct('Symbol', '\alpha_j(s)', 'Description', 'Dirichlet prior pseudo-count'), ...
                struct('Symbol', 't_j', 'Description', 'times opponent j was observed')];
            meta.Assumptions = {'beliefs are independent across opponents (product of marginals)', ...
                'stationary opponents: every observation counts equally'};
            meta.IsSymbolic = true;
            meta.Reference = "Brown (1951); Fudenberg & Levine, The Theory of Learning in Games (1998), ch. 2";
        end

        function spec = getPipelineSpec(~)
            spec = struct('ComponentName', "EmpiricalBelief", 'Kind', "NONVIRTUAL", ...
                'Inputs', struct('Name', "actions", 'Domain', "GameTheory-Profile", 'Dtype', "double", 'Shape', "1 x J"), ...
                'Outputs', struct('Name', "beliefs", 'Domain', "GameTheory-MixedStrategy", 'Dtype', "cell", 'Shape', "1 x J"), ...
                'DiagramSymbol', struct('Type', "default", 'Label', "empiricalbelief"), 'ComposesFrom', {{}});
        end
    end
end
