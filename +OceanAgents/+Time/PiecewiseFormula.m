classdef PiecewiseFormula < handle
    % PIECEWISEFORMULA  A value that is a function of time, stored as immutable segments (parent sections 4 and 8).
    % Each segment has an anchor (T0, V0), parameters and a formula V(t) = Fcn(t - T0, V0, Params). A deposit, rate change
    % or other discontinuity CLOSES one segment and opens another, so any past time can still be answered (an anchor
    % alone could not answer before the latest rebase). Queries are pure: no RNG, no learning, no side effects.
    %
    %   f = OceanAgents.Time.PiecewiseFormula(0, 100, struct('rate', -2));   % default linear: V0 + rate*dt
    %   f.valueAt(10)                      % 80
    %   f.rebase(10, struct('rate', -1), 'Delta', 50);   % at t=10: value 80 + 50 deposit, new slope
    %   f.valueAt(5)                       % 90 (old segment, still exact);  f.valueAt(12) -> 128
    %   f.nextCrossing(0)                                % time the CURRENT segment reaches level 0 (Inf if never)
    %
    % Threshold effects must happen even if nobody reads the value: the owner of the world's calendar (a kernel
    % predictor) schedules the crossing, using nextCrossing. Root finding is exact for the default linear formula; a
    % custom Fcn must supply 'Root' = @(level, t0, v0, params) -> time.
    properties (SetAccess = private)
        Segments struct = struct('T0', {}, 'V0', {}, 'Params', {}, 'Fcn', {}, 'Root', {})
    end
    methods
        function obj = PiecewiseFormula(t0, v0, params, opts)
            arguments
                t0 (1,1) double
                v0 (1,1) double
                params (1,1) struct
                opts.Fcn function_handle = @(dt, v0, p) v0 + p.rate .* dt
                opts.Root = []
            end
            if isempty(opts.Root)
                root = @(level, t0, v0, p) linearRoot_(level, t0, v0, p);
                if ~isfield(params, 'rate'), root = []; end
            else
                root = opts.Root;
            end
            obj.Segments = struct('T0', t0, 'V0', v0, 'Params', params, 'Fcn', opts.Fcn, 'Root', {root});
        end

        function v = valueAt(obj, t)
            [k, ~] = obj.segmentAt_(t);
            s = obj.Segments(k);
            v = s.Fcn(t - s.T0, s.V0, s.Params);
        end

        function rebase(obj, t, params, opts)
            % REBASE  Close the current segment at t and open a new one anchored at the value there (+ Delta).
            arguments
                obj
                t (1,1) double
                params (1,1) struct
                opts.Delta (1,1) double = 0
                opts.Fcn function_handle = obj.Segments(end).Fcn
                opts.Root = obj.Segments(end).Root
            end
            if t < obj.Segments(end).T0
                error('OceanAgents:Time:Retroactive', 'Rebase at %g precedes the open segment (%g).', t, obj.Segments(end).T0);
            end
            v = obj.valueAt(t) + opts.Delta;
            obj.Segments(end+1) = struct('T0', t, 'V0', v, 'Params', params, 'Fcn', opts.Fcn, 'Root', {opts.Root});
        end

        function t = nextCrossing(obj, level)
            % NEXTCROSSING  Time >= the open segment's anchor at which the open segment reaches LEVEL (Inf if never).
            s = obj.Segments(end);
            if isempty(s.Root)
                error('OceanAgents:Time:NoRootDetection', 'This formula declares no Root; threshold events cannot be predicted.');
            end
            t = s.Root(level, s.T0, s.V0, s.Params);
        end

        function n = segmentCount(obj), n = numel(obj.Segments); end
    end
    methods (Access = private)
        function [k, s] = segmentAt_(obj, t)
            T0 = [obj.Segments.T0];
            if t < T0(1)
                error('OceanAgents:Time:NoCoverage', 'Time %g precedes the first segment (%g).', t, T0(1));
            end
            k = find(T0 <= t, 1, 'last');
            s = obj.Segments(k);
        end
    end
end

function t = linearRoot_(level, t0, v0, p)
if p.rate == 0
    if v0 == level, t = t0; else, t = Inf; end
    return
end
dt = (level - v0) / p.rate;
if dt < 0, t = Inf; else, t = t0 + dt; end
end
