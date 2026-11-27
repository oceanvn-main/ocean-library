classdef Sampling
    % SAMPLING  Which points of ONE axis to record (Format spec Sec. 5.1).
    %   Rules act on axis VALUES (the real step / epoch / lambda), except
    %   "every" with KIndex, which counts the run-order index (0-based) for
    %   non-integer axes.
    %
    %     all                      every value
    %     every(k, from, to)       values v with (v - from) a multiple of k and
    %                              v in [min(from,to), max(from,to)]; from
    %                              defaults to 0 (anchor), the range to (-Inf, Inf)
    %     everyIndex(k)            run-order index a multiple of k
    %     list(values)             exactly these values (not on a closed axis -> error)
    %     window(center, half)     values within center +/- half
    %     union(r1, r2, ...)       any of the sub-rules
    %
    %   Example:
    %     s = OceanReplay.Core.Sampling.union(OceanReplay.Core.Sampling.every(20), ...
    %                                        OceanReplay.Core.Sampling.window(51, 1));
    %     mask = s.select(1000:-1:0);      % steps 1000, 980, ..., 0 plus 50, 51, 52

    properties (SetAccess = private)
        Rule (1,1) string = "all"
        K = []
        KIndex = []
        From = []
        To = []
        Values (1,:) double = []
        Center = []
        Half = []
        Rules (1,:) cell = {}
    end

    properties (Constant, Access = private)
        RelTol = 1e-9
    end

    methods (Static)
        function s = all()
            s = OceanReplay.Core.Sampling();
        end

        function s = every(k, from, to)
            arguments
                k (1,1) double {mustBePositive}
                from (1,1) double = 0
                to (1,1) double = NaN
            end
            s = OceanReplay.Core.Sampling(); s.Rule = "every"; s.K = k; s.From = from; s.To = to;
        end

        function s = everyIndex(k)
            arguments
                k (1,1) double {mustBePositive, mustBeInteger}
            end
            s = OceanReplay.Core.Sampling(); s.Rule = "every"; s.KIndex = k;
        end

        function s = list(values)
            arguments
                values (1,:) double
            end
            s = OceanReplay.Core.Sampling(); s.Rule = "list"; s.Values = values;
        end

        function s = window(center, half)
            arguments
                center (1,1) double
                half (1,1) double {mustBeNonnegative}
            end
            s = OceanReplay.Core.Sampling(); s.Rule = "window"; s.Center = center; s.Half = half;
        end

        function s = union(varargin)
            s = OceanReplay.Core.Sampling(); s.Rule = "union"; s.Rules = varargin;
            for i = 1:numel(varargin)
                if ~isa(varargin{i}, 'OceanReplay.Core.Sampling')
                    error('OceanReplay:Sampling:union', 'union() takes Sampling rules.');
                end
            end
        end

        function s = fromStruct(t)
            % From a decoded JSON object {"rule": ..., ...}.
            if ~isfield(t, 'rule'), error('OceanReplay:Sampling:rule', 'A sampling rule needs a "rule" field.'); end
            S = OceanReplay.Core.Sampling;
            switch string(t.rule)
                case "all", s = S.all();
                case "every"
                    if isfield(t, 'kIndex'), s = S.everyIndex(t.kIndex); return; end
                    from = 0; to = NaN;
                    if isfield(t, 'from'), from = t.from; end
                    if isfield(t, 'to'), to = t.to; end
                    s = S.every(t.k, from, to);
                case "list", s = S.list(OceanReplay.Core.Sampling.numList(t.values));
                case "window", s = S.window(t.center, t.half);
                case "union"
                    r = t.rules;
                    if isstruct(r), r = num2cell(r); end
                    subs = cellfun(@(x) OceanReplay.Core.Sampling.fromStruct(x), r, 'UniformOutput', false);
                    s = S.union(subs{:});
                otherwise
                    error('OceanReplay:Sampling:rule', 'Unknown sampling rule "%s" (all | every | list | window | union).', string(t.rule));
            end
        end
    end

    methods
        function t = toStruct(obj)
            % JSON-ready struct (lists as cells so they always encode as arrays).
            t = struct('rule', obj.Rule);
            switch obj.Rule
                case "every"
                    if ~isempty(obj.KIndex), t.kIndex = obj.KIndex;
                    else
                        t.k = obj.K; t.from = obj.From;
                        if ~isnan(obj.To), t.to = obj.To; end
                    end
                case "list", t.values = num2cell(obj.Values);
                case "window", t.center = obj.Center; t.half = obj.Half;
                case "union", t.rules = cellfun(@(r) r.toStruct(), obj.Rules, 'UniformOutput', false);
            end
        end

        function tf = contains(obj, value, index)
            % True when the point with axis value `value` (run-order index `index`, 0-based) is selected.
            arguments
                obj
                value (1,1) double
                index (1,1) double = NaN
            end
            tol = obj.RelTol * max(1, abs(value));
            switch obj.Rule
                case "all", tf = true;
                case "every"
                    if ~isempty(obj.KIndex)
                        tf = ~isnan(index) && mod(index, obj.KIndex) == 0;
                    else
                        r = mod(value - obj.From, obj.K);
                        tf = min(r, obj.K - r) <= tol;
                        if ~isnan(obj.To)
                            tf = tf && value >= min(obj.From, obj.To) - tol && value <= max(obj.From, obj.To) + tol;
                        end
                    end
                case "list", tf = any(abs(obj.Values - value) <= tol);
                case "window", tf = abs(value - obj.Center) <= obj.Half + tol;
                case "union", tf = any(cellfun(@(r) r.contains(value, index), obj.Rules));
            end
        end

        function mask = select(obj, values)
            % Logical mask over a known axis (values in run order).
            arguments
                obj
                values (1,:) double
            end
            mask = false(size(values));
            for i = 1:numel(values)
                mask(i) = obj.contains(values(i), i - 1);
            end
        end

        function bad = missing(obj, values)
            % "list" values (also inside unions) that are not on an axis with these values.
            bad = [];
            switch obj.Rule
                case "list"
                    for v = obj.Values
                        if ~any(abs(values - v) <= obj.RelTol * max(1, abs(v))), bad(end+1) = v; end %#ok<AGROW>
                    end
                case "union"
                    for i = 1:numel(obj.Rules), bad = [bad, obj.Rules{i}.missing(values)]; end %#ok<AGROW>
            end
        end
    end

    methods (Static, Access = private)
        function v = numList(x)
            if iscell(x), v = cell2mat(x(:).'); else, v = reshape(double(x), 1, []); end
        end
    end
end
