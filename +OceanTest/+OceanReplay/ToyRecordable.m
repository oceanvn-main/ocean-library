classdef ToyRecordable < OceanReplay.Record.IRecordable
    % TOYRECORDABLE  Deterministic test domain for OceanReplay (Master plan G1).
    %   Two nested loops: outer axis t = 10, 8, ..., 0 (DESCENDING, like a
    %   diffusion sampler), inner axis k = 1, 2, 3. It offers one channel of
    %   every kind, a channel it never provides ("ghost"), and one view that
    %   draws random numbers from BOTH its stream and the global generator.
    %   The run itself also uses the global generator, so an unguarded view
    %   would change the output (the invariance test would catch it).
    %   Every provider call is counted and its value remembered (Calls, History).

    properties
        OpenAxis (1,1) logical = false      % declare t without values (filled as visited)
        Seed (1,1) double = 5
        Calls                               % name -> number of provider evaluations
        History                             % name -> struct array (Coords, Value)
    end

    properties (Constant)
        T = [10 8 6 4 2 0]
        K = [1 2 3]
    end

    methods
        function obj = ToyRecordable(opts)
            arguments
                opts.OpenAxis (1,1) logical = false
                opts.Seed (1,1) double = 5
            end
            obj.OpenAxis = opts.OpenAxis; obj.Seed = opts.Seed;
            obj.Calls = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.History = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function axes = axesInfo(obj)
            t = obj.T; if obj.OpenAxis, t = []; end
            axes = [OceanReplay.Core.Axis("t", t, 'Symbol', "t", 'Description', "outer step, descending"), ...
                    OceanReplay.Core.Axis("k", obj.K, 'Symbol', "k", 'Description', "inner index")];
        end

        function [c, v] = catalog(~)
            C = @OceanReplay.Core.ChannelInfo;
            c = [C("energy", "scalar", 'Axes', "t", 'Symbol', "E", 'Description', "sum of squares of x"), ...
                 C("vec", "vector", 'Axes', "t", 'ValueShape', 4), ...
                 C("mat", "matrix", 'Axes', ["t" "k"], 'ValueShape', [2 3], 'RowLabels', ["r1" "r2"]), ...
                 C("img", "image", 'Axes', "t", 'ValueShape', [2 3 2], 'DisplayRange', [-5 5]), ...
                 C("seq", "sequence", 'Axes', "t", 'ValueShape', [5 2]), ...
                 C("pts", "points", 'ValueShape', [4 2]), ...
                 C("tok", "tokens", 'Axes', "k", 'ValueShape', 3), ...
                 C("label", "vector", 'Axes', "t", 'ValueShape', 4, 'Dtype', "int32", 'Categories', ["a" "b" "c"]), ...
                 C("ghost", "scalar", 'Axes', "t")];
            v = OceanReplay.Core.ViewInfo("noisy", "image", 'Axes', "t", 'ValueShape', [2 3 2], 'DisplayRange', [-5 5], ...
                'Compute', @(s, p, r) s.img() + p.scale * randn(r, 2, 3, 2) + 0 * rand(1), ...
                'HowMade', "img plus Gaussian noise from the view stream (test only)", 'Reference', "test fixture", ...
                'InTexts', false);
        end

        function out = run(obj, rec)
            s = RandStream('mt19937ar', 'Seed', obj.Seed);
            x = zeros(2, 3, 2);
            total = 0;
            for t = obj.T
                x = 0.5 * x + randn(s, size(x)) + 0.01 * randn(1);      % global draw on purpose
                for k = obj.K
                    y = x(:, :, 1) * k;
                    total = total + sum(y(:));
                    xs = x; tt = t; kk = k;
                    st = struct( ...
                        'energy', @() obj.tick("energy", [tt kk], sum(xs(:) .^ 2)), ...
                        'vec',    @() obj.tick("vec",    [tt kk], [tt; xs(1); xs(end); sum(xs(:))]), ...
                        'mat',    @() obj.tick("mat",    [tt kk], xs(:, :, 1) * kk), ...
                        'img',    @() obj.tick("img",    [tt kk], xs), ...
                        'seq',    @() obj.tick("seq",    [tt kk], [(1:5)' * tt, (1:5)' * xs(1)]), ...
                        'pts',    @() obj.tick("pts",    [tt kk], reshape(1:8, 4, 2)), ...
                        'tok',    @() obj.tick("tok",    [tt kk], ["w" + kk, "ñ", "中"]), ...
                        'label',  @() obj.tick("label",  [tt kk], int32([1 2 3 mod(tt, 3) + 1])));
                    rec.point(struct('t', t, 'k', k), st);
                end
            end
            out = struct('x', x, 'total', total, 'lastGlobal', rand(1));
        end

        function v = tick(obj, name, coords, v)
            key = char(name);
            if ~isKey(obj.Calls, key), obj.Calls(key) = 0; obj.History(key) = struct('Coords', {}, 'Value', {}); end
            obj.Calls(key) = obj.Calls(key) + 1;
            h = obj.History(key); h(end+1) = struct('Coords', coords, 'Value', v); obj.History(key) = h;
        end

        function n = calls(obj, name)
            n = 0; if isKey(obj.Calls, char(name)), n = obj.Calls(char(name)); end
        end
    end

    methods (Static)
        function spec = fullSpec()
            % Every provided channel at every point, plus the view at two steps.
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("all", 0, ["energy" "vec" "mat" "img" "seq" "pts" "tok" "label"]) ...
                .addView("noisy", "noisy", struct('scale', 0.1), struct('t', S.list([8 2])));
        end
    end
end
