classdef FictitiousPlay < handle
    % FICTITIOUSPLAY  One player of a finite game learning by fictitious play (Brown 1951): it keeps an empirical belief
    % about every opponent and plays a best response to that belief.
    %
    %   expected payoff  U_i(s_i) = sum_{s_-i} u_i(s_i, s_-i) * prod_{j ~= i} p_j(s_j)
    %   best response    BR_i     = argmax_{s_i} U_i(s_i)                          ('Response', "best")
    %   smooth response  sigma(s_i) proportional to exp(U_i(s_i) / tau)           ('Response', "smooth", 'Temperature', tau)
    %
    % The expected payoff is computed by contracting the player's payoff tensor with each opponent's belief in turn
    % (one tensor-vector product per opponent), so it works for any number of players and any finite strategy sets.
    %
    %   g  = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', [1 -1; -1 1], 'PayoffB', [-1 1; 1 -1]);
    %   fp = OceanStats.GameTheory.Learning.FictitiousPlay(g, 1);          % player 1 of matching pennies
    %   fp.observe([1 2]);                    % a full profile was played (own entry is ignored)
    %   s  = fp.act(stream);                  % best response (ties broken uniformly with the stream)
    %   U  = fp.expectedPayoffs();            % 1 x m_i
    %
    % Also usable without a NormalFormGame: FictitiousPlay(U_i, i) with U_i = this player's payoff tensor (dimension k
    % = player k's strategies), e.g. a guess-vs-opponent's-put matrix inside an agent.

    properties (SetAccess = private)
        Player double                         % this player's index i
        Payoff double                         % u_i as an N-dimensional tensor (dimension k = player k's strategies)
        Belief                                % OceanStats.GameTheory.Learning.EmpiricalBelief over the opponents
        Opponents double                      % indices of the opponents, in belief order
        Response string = "best"
        Temperature double = 0.1
    end

    methods
        function obj = FictitiousPlay(source, player, options)
            arguments
                source                          % a NormalFormGame, or this player's payoff tensor u_i
                player (1,1) double {mustBeInteger, mustBePositive}
                options.Prior = 1
                options.Response (1,1) string {mustBeMember(options.Response, ["best" "smooth"])} = "best"
                options.Temperature (1,1) double {mustBePositive} = 0.1
            end
            if isa(source, 'OceanStats.GameTheory.Core.NormalFormGame')
                if player > source.numPlayers()
                    error('OceanStats:GameTheory:BadPlayer', 'The game has %d players, not %d.', source.numPlayers(), player);
                end
                payoff = source.playerTensor(player);
            elseif isnumeric(source) && isreal(source)
                payoff = double(source);
            else
                error('OceanStats:GameTheory:BadPayoff', 'Give a NormalFormGame or the player''s payoff tensor.');
            end
            obj.init_(payoff, player, options);
        end

        function U = expectedPayoffs(obj)
            % EXPECTEDPAYOFFS  U_i(s_i) for every own strategy under the current beliefs (1 x m_i).
            T = obj.Payoff;
            n = ndims(T);
            if n < max([obj.Player, obj.Opponents]), n = max([obj.Player, obj.Opponents]); end
            beliefs = obj.Belief.all();
            % contract opponents from the highest dimension down, so the remaining dimensions keep their positions
            [~, order] = sort(obj.Opponents, 'descend');
            for k = order
                d = obj.Opponents(k);
                p = beliefs{k};
                sz = size(T, 1:n);
                perm = [1:d - 1, d + 1:n, d];
                M = reshape(permute(T, perm), [], sz(d));
                T = reshape(M * p(:), [sz(1:d - 1), sz(d + 1:n), 1]);
                n = n - 1;
            end
            U = reshape(T, 1, []);
        end

        function sigma = responseProbabilities(obj)
            % RESPONSEPROBABILITIES  The mixed response: uniform over the best responses, or the logit (smooth) one.
            U = obj.expectedPayoffs();
            if obj.Response == "smooth"
                z = (U - max(U)) / obj.Temperature;
                sigma = exp(z) / sum(exp(z));
            else
                best = abs(U - max(U)) <= 1e-12 * max(1, abs(max(U)));
                sigma = best / nnz(best);
            end
        end

        function s = act(obj, stream)
            % ACT  Draw this player's strategy from the response (one uniform draw from stream).
            arguments
                obj
                stream (1,1) RandStream
            end
            c = cumsum(obj.responseProbabilities());
            s = find(rand(stream) * c(end) <= c, 1);
        end

        function observe(obj, profile)
            % OBSERVE  A played profile (1 x N; this player's own entry is ignored; NaN = not seen).
            arguments
                obj
                profile (1,:) double
            end
            obj.Belief.observe(profile(obj.Opponents));
        end

        function P = beliefs(obj)
            P = obj.Belief.all();
        end

        function reset(obj)
            obj.Belief.reset();
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = ["U_i(s_i) = \sum_{s_{-i}} u_i(s_i, s_{-i}) \prod_{j \neq i} p_j(s_j)", ...
                "BR_i = \arg\max_{s_i} U_i(s_i)", "\sigma_i(s_i) \propto \exp(U_i(s_i)/\tau)"];
            meta.Symbols = [struct('Symbol', 'u_i', 'Description', 'player i payoff'), ...
                struct('Symbol', 'p_j', 'Description', 'empirical belief about opponent j (EmpiricalBelief)'), ...
                struct('Symbol', '\tau', 'Description', 'temperature of the smooth (logit) response')];
            meta.Assumptions = {sprintf('player %d of %d; response "%s"', obj.Player, numel(obj.Opponents) + 1, obj.Response), ...
                'opponents treated as stationary and independent (classical fictitious play)'};
            meta.IsSymbolic = true;
            meta.Reference = "Brown (1951) Iterative solution of games by fictitious play; Robinson (1951); " + ...
                "Fudenberg & Levine (1998), ch. 2 and 4 (smooth fictitious play)";
        end

        function spec = getPipelineSpec(~)
            spec = struct('ComponentName', "FictitiousPlay", 'Kind', "NONVIRTUAL", ...
                'Inputs', struct('Name', "profile", 'Domain', "GameTheory-Profile", 'Dtype', "double", 'Shape', "1 x N"), ...
                'Outputs', struct('Name', "strategy", 'Domain', "GameTheory-Strategy", 'Dtype', "double", 'Shape', "1 x 1"), ...
                'DiagramSymbol', struct('Type', "default", 'Label', "fictitiousplay"), ...
                'ComposesFrom', {{'OceanStats.GameTheory.Learning.EmpiricalBelief'}});
        end
    end

    methods (Access = private)
        function init_(obj, payoff, player, options)
            N = max(ndims(payoff), player);
            if ~all(isfinite(payoff(:)))
                error('OceanStats:GameTheory:BadPayoff', 'Payoffs must be finite.');
            end
            sz = size(payoff, 1:N);
            obj.Player = player;
            obj.Payoff = payoff;
            obj.Opponents = setdiff(1:N, player);
            obj.Belief = OceanStats.GameTheory.Learning.EmpiricalBelief(sz(obj.Opponents), 'Prior', options.Prior);
            obj.Response = options.Response;
            obj.Temperature = options.Temperature;
        end
    end
end
