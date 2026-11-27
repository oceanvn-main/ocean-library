classdef FictitiousPlaySolver < OceanStats.GameTheory.Solvers.ISolver
    % FICTITIOUSPLAYSOLVER  Every player of a finite normal-form game learns by fictitious play against the others for
    % a number of rounds (Brown 1951). The players' empirical frequencies converge to a Nash equilibrium in two-player
    % zero-sum games (Robinson 1951), and in other classes (potential games, 2 x 2 games); in general they need not
    % (Shapley 1964), which is why the history records the exploitability of the empirical profile each round:
    %
    %   exploitability(x) = sum_i [ max_{s_i} u_i(s_i, x_-i) - u_i(x) ]        (0 exactly at a Nash equilibrium)
    %
    %   g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', [1 -1; -1 1], 'PayoffB', [-1 1; 1 -1]);
    %   s = OceanStats.GameTheory.Solvers.FictitiousPlaySolver('Rounds', 5000);
    %   r = s.solve(g);              % r.Frequencies{i} (1 x m_i), r.Exploitability (final), r.Profiles (Rounds x N)
    %   h = s.getHistory();          % h.Exploitability (1 x Rounds)
    %
    % Players move simultaneously each round and best-respond to the beliefs held BEFORE the round. Ties are broken
    % by the lowest index (deterministic) unless a stream is given, then uniformly.

    properties (SetAccess = private)
        Rounds double = 1000
        Prior = 0
        Response string = "best"
        Temperature double = 0.1
        Stream = []
    end

    methods
        function obj = FictitiousPlaySolver(options)
            arguments
                options.Rounds (1,1) double {mustBeInteger, mustBePositive} = 1000
                options.Prior = 0
                options.Response (1,1) string {mustBeMember(options.Response, ["best" "smooth"])} = "best"
                options.Temperature (1,1) double {mustBePositive} = 0.1
                options.Stream = []
            end
            obj.Rounds = options.Rounds;
            obj.Prior = options.Prior;
            obj.Response = options.Response;
            obj.Temperature = options.Temperature;
            obj.Stream = options.Stream;
        end

        function result = solve(obj, game)
            arguments
                obj
                game (1,1) OceanStats.GameTheory.Core.NormalFormGame
            end
            N = game.numPlayers();
            m = arrayfun(@(i) game.numStrategies(i), 1:N);
            players = arrayfun(@(i) OceanStats.GameTheory.Learning.FictitiousPlay(game, i, 'Prior', obj.Prior, ...
                'Response', obj.Response, 'Temperature', obj.Temperature), 1:N, 'UniformOutput', false);
            counts = arrayfun(@(k) zeros(1, k), m, 'UniformOutput', false);
            profiles = zeros(obj.Rounds, N);
            expl = zeros(1, obj.Rounds);
            for t = 1:obj.Rounds
                profile = zeros(1, N);
                for i = 1:N
                    if isempty(obj.Stream)
                        sigma = players{i}.responseProbabilities();
                        if obj.Response == "smooth"
                            c = cumsum(sigma);  profile(i) = find(0.5 * c(end) <= c, 1);   % deterministic median draw
                        else
                            profile(i) = find(sigma > 0, 1);                               % lowest-index tie-break
                        end
                    else
                        profile(i) = players{i}.act(obj.Stream);
                    end
                end
                for i = 1:N, players{i}.observe(profile); end
                for i = 1:N, counts{i}(profile(i)) = counts{i}(profile(i)) + 1; end
                profiles(t, :) = profile;
                freq = cellfun(@(c) c / t, counts, 'UniformOutput', false);
                expl(t) = OceanStats.GameTheory.Solvers.FictitiousPlaySolver.exploitability(game, freq);
            end
            result = struct('Frequencies', {cellfun(@(c) c / obj.Rounds, counts, 'UniformOutput', false)}, ...
                'Exploitability', expl(end), 'Profiles', profiles);
            obj.History_ = struct('Exploitability', expl, 'Rounds', obj.Rounds);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = ["s_i^{t+1} \in BR_i(\hat{x}_{-i}^t)", ...
                "\mathrm{expl}(x) = \sum_i \left[\max_{s_i} u_i(s_i, x_{-i}) - u_i(x)\right]"];
            meta.Symbols = [struct('Symbol', '\hat{x}_{-i}^t', 'Description', 'opponents'' empirical frequencies after t rounds'), ...
                struct('Symbol', '\mathrm{expl}', 'Description', 'exploitability (Nash gap) of a mixed profile')];
            meta.Assumptions = {sprintf('%d rounds, simultaneous best responses, response "%s"', obj.Rounds, obj.Response), ...
                'convergence guaranteed for 2-player zero-sum games (Robinson 1951), not in general (Shapley 1964)'};
            meta.IsSymbolic = true;
            meta.Reference = "Brown (1951); Robinson (1951); Shapley (1964); Fudenberg & Levine (1998)";
        end

        function spec = getPipelineSpec(~)
            spec = struct('ComponentName', "FictitiousPlaySolver", 'Kind', "NONVIRTUAL", ...
                'Inputs', struct('Name', "game", 'Domain', "GameTheory-NormalForm", ...
                    'Dtype', "OceanStats.GameTheory.Core.NormalFormGame", 'Shape', "1 x 1"), ...
                'Outputs', struct('Name', "frequencies", 'Domain', "GameTheory-MixedProfile", 'Dtype', "cell", 'Shape', "1 x N"), ...
                'DiagramSymbol', struct('Type', "default", 'Label', "fictitiousplaysolver"), ...
                'ComposesFrom', {{'OceanStats.GameTheory.Learning.FictitiousPlay'}});
        end
    end

    methods (Static)
        function e = exploitability(game, x)
            % EXPLOITABILITY  sum_i [ best response value against x_-i  -  value of x ] for a mixed profile x.
            N = game.numPlayers();
            e = 0;
            for i = 1:N
                T = game.playerTensor(i);
                for j = N:-1:1                                    % contract every opponent: U_i(s_i) vector
                    if j == i, continue; end
                    n = max(ndims(T), N);
                    sz = size(T, 1:n);
                    perm = [1:j - 1, j + 1:n, j];
                    T = reshape(reshape(permute(T, perm), [], sz(j)) * x{j}(:), [sz(1:j - 1), sz(j + 1:n), 1]);
                end
                U = reshape(T, 1, []);
                e = e + max(U) - U * x{i}(:);
            end
        end
    end
end
