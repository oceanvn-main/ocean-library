classdef UCB1Selection
    % UCB1SELECTION  Upper-confidence tree policy for MCTS selection (Sutton & Barto 2nd ed., Section 2.7 UCB
    % action selection, used in the tree as in Section 8.11; Kocsis & Szepesvári 2006, "UCT").
    %
    %   rule = OceanAlgo.StateSearch.UCB1Selection(c);
    %   k    = rule.select(Q, N, stream)
    %     Q, N   1 x K action-value estimates and visit counts of the edges of one node
    %     k      index into Q/N of the chosen edge
    %
    %   untried edges (N = 0) first, chosen uniformly at random from the stream;
    %   otherwise  k = argmax_k  Q(k) + c * sqrt( ln(Σ N) / N(k) )      (ties -> lowest index)
    %
    % The exploration constant c is scale dependent: UCB1's c = sqrt(2) assumes returns in [0, 1]. Set c
    % relative to the spread of returns of the problem (not in the text: a practical note).
    properties (SetAccess = immutable)
        C (1,1) double
    end
    methods
        function obj = UCB1Selection(c)
            arguments
                c (1,1) double {mustBeNonnegative, mustBeFinite} = sqrt(2)
            end
            obj.C = c;
        end
        function k = select(obj, Q, N, stream)
            untried = find(N == 0);
            if ~isempty(untried)
                k = untried(randi(stream, numel(untried)));
                return
            end
            u = Q + obj.C * sqrt(log(sum(N)) ./ N);
            [~, k] = max(u);
        end
    end
end
