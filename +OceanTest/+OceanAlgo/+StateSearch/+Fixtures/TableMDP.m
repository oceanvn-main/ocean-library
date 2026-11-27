classdef TableMDP < OceanAlgo.StateSearch.IDistributionModel & OceanAlgo.StateSearch.ISampleModel
    % TABLEMDP  Small finite MDP for engine tests: both a distribution and a sample model from one table.
    %   T{s,a} = struct array (Next, Reward, Terminated, Probability), or [] when a is illegal in s.
    % Also offers exact oracles: valueIterationSweeps (synchronous) and policyValue (linear solve).
    properties (SetAccess = immutable)
        T cell
    end
    methods
        function obj = TableMDP(T), obj.T = T; end
        function a = actions(obj, s), a = find(~cellfun(@isempty, obj.T(s, :))); end
        function k = key(~, s), k = s; end
        function o = outcomes(obj, s, a), o = obj.T{s, a}; end
        function o = sample(obj, s, a, stream)
            d = obj.T{s, a};
            k = find(rand(stream) <= cumsum([d.Probability]), 1, 'first');
            if isempty(k), k = numel(d); end
            o = struct('Next', d(k).Next, 'Reward', d(k).Reward, 'Terminated', d(k).Terminated);
        end

        function [V, Q] = valueIterationSweeps(obj, V0, gamma, sweeps)
            % V_k = max_a Σ p (r + γ V_{k-1}(s')) for all s, synchronously; no-action states keep V0.
            S = size(obj.T, 1); A = size(obj.T, 2); V = V0(:);
            Q = nan(S, A);
            for k = 1:sweeps
                Q = nan(S, A);
                for s = 1:S
                    for a = obj.actions(s)
                        Q(s, a) = obj.q_(s, a, V, gamma);
                    end
                end
                Vn = V0(:);
                has = any(~isnan(Q), 2);
                Vn(has) = max(Q(has, :), [], 2);
                V = Vn;
            end
        end

        function V = policyValue(obj, pi, gamma, leaf)
            % Exact v_pi for a deterministic policy vector pi (pi(s) = action), solving v = r_pi + γ P_pi v.
            % States with no legal action have value leaf(s).
            S = size(obj.T, 1); P = zeros(S); r = zeros(S, 1); b = zeros(S, 1);
            for s = 1:S
                if isempty(obj.actions(s)), b(s) = leaf(s); continue; end
                d = obj.T{s, pi(s)};
                for k = 1:numel(d)
                    r(s) = r(s) + d(k).Probability * d(k).Reward;
                    if ~d(k).Terminated
                        P(s, d(k).Next) = P(s, d(k).Next) + d(k).Probability;
                    end
                end
            end
            noAct = arrayfun(@(s) isempty(obj.actions(s)), (1:S)');
            M = eye(S) - gamma * P; rhs = r;
            M(noAct, :) = 0; M(sub2ind([S S], find(noAct), find(noAct))) = 1; rhs(noAct) = b(noAct);
            V = M \ rhs;
        end
    end
    methods (Access = private)
        function q = q_(obj, s, a, V, gamma)
            d = obj.T{s, a}; q = 0;
            for k = 1:numel(d)
                c = 0; if ~d(k).Terminated, c = V(d(k).Next); end
                q = q + d(k).Probability * (d(k).Reward + gamma * c);
            end
        end
    end
    methods (Static)
        function m = tiny()
            % The RTDP fixture MDP: s1 a1 -> (s2, r=2, p=.25) | (end, r=4, p=.75); s1 a2 -> (end, r=3);
            % s2 a1 -> (end, r=1); s3 has no action.
            o = @(n, r, t, p) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', p);
            T = cell(3, 2);
            T{1,1} = [o(2, 2, false, 0.25), o(3, 4, true, 0.75)];
            T{1,2} = o(3, 3, true, 1);
            T{2,1} = o(3, 1, true, 1);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
        end
        function m = random(S, A, branching, stream)
            % Random episodic MDP (S&B Figure 8.8 style): each (s,a) leads to `branching` successors uniformly;
            % every transition terminates with probability 0.1; rewards ~ N(0,1).
            o = @(n, r, t, p) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', p);
            T = cell(S, A);
            for s = 1:S
                for a = 1:A
                    nexts = randi(stream, S, 1, branching);
                    rows = repmat(o(1, 0, false, 1), 1, 0);
                    for k = 1:branching
                        rows(end+1) = o(nexts(k), randn(stream), false, 0.9 / branching); %#ok<AGROW>
                    end
                    rows(end+1) = o(1, randn(stream), true, 0.1); %#ok<AGROW>
                    T{s, a} = rows;
                end
            end
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
        end
    end
end
