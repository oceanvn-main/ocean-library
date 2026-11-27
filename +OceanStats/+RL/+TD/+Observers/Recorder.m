classdef Recorder < OceanStats.RL.TD.Observers.IObserver
    % RECORDER  Keeps every episode (state indices, actions, rewards, double-estimator coin flips, terminated / truncated)
    % in Episodes, the format the independent Python replay validates. A terminal outside the state table is recorded
    % as state 0, never as a real state index.
    properties (SetAccess = private)
        Episodes cell = {}
    end
    properties (Access = private)
        Current struct = struct()
    end
    methods
        function startRun(obj), obj.Episodes = {};  obj.Current = struct(); end
        function ep = episodes(obj), ep = obj.Episodes; end
        function begin(obj, s0, a0)
            obj.Current = struct('S', s0, 'A', a0, 'R', zeros(0, 1), 'Coin', zeros(0, 1));
        end
        function step(obj, r, s2, a2, outsideTable)
            if outsideTable, s2 = 0; end
            obj.Current.R(end+1, 1) = r;  obj.Current.S(end+1, 1) = s2;
            if a2 > 0, obj.Current.A(end+1, 1) = a2; end
        end
        function coin(obj, c)
            obj.Current.Coin(end+1, 1) = c;
        end
        function action(obj, a)
            obj.Current.A(end+1, 1) = a;
        end
        function finish(obj, terminated, truncated)
            rec = obj.Current;
            rec.Terminated = terminated;  rec.Truncated = truncated;
            obj.Episodes{end+1} = rec;
        end
        function finishArrays(obj, S, A, R, terminated, truncated, outsideLast)
            if outsideLast && terminated, S(end) = 0; end
            obj.Episodes{end+1} = struct('S', S(:), 'A', A(:), 'R', R(:), 'Coin', zeros(0, 1), ...
                'Terminated', terminated, 'Truncated', truncated);
        end
    end
end
