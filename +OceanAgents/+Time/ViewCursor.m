classdef ViewCursor < handle
    % VIEWCURSOR  A read-only view position on one time domain (parent sections 1, 3.2 and 7.1). Moving it never touches
    % the live world; it only changes which historical time attached views evaluate.
    %
    %   c = OceanAgents.Time.ViewCursor("arm", 'Span', [0 100]);
    %   g = c.seek(40, "after");        % SeekView: a NEWER generation; results of older generations are stale
    %   c.play(-2); c.advance(0.5);     % reverse playback: wall time increases, domain time decreases (Target 39)
    %   c.isCurrent(g)                  % false once another seek happened
    %
    % Boundary "after" shows the completed commit at or before the target (default); "before" the state before events
    % at the target. Playback rate may be negative; the target is clamped to Span and playback stops at either end.
    properties (SetAccess = private)
        DomainID string
        Target double = 0
        Boundary string = "after"
        Generation double = 0
        Rate double = 0
        Span double = [-Inf Inf]
    end
    methods
        function obj = ViewCursor(domainID, opts)
            arguments
                domainID (1,1) string
                opts.Span (1,2) double = [-Inf Inf]
                opts.Target (1,1) double = 0
            end
            obj.DomainID = domainID; obj.Span = opts.Span;
            obj.Target = min(max(opts.Target, opts.Span(1)), opts.Span(2));
        end

        function g = seek(obj, t, boundary)
            arguments
                obj
                t (1,1) double
                boundary (1,1) string = "after"
            end
            obj.Target = min(max(t, obj.Span(1)), obj.Span(2));
            obj.Boundary = boundary;
            obj.Generation = obj.Generation + 1;
            g = obj.Generation;
        end

        function setSpan(obj, span), obj.Span = span; end

        function play(obj, rate), obj.Rate = rate; end
        function pause(obj), obj.Rate = 0; end

        function g = advance(obj, wallDt)
            % ADVANCE  Playback step: Target += Rate * wallDt (wall time is always positive). Generation changes only
            % when the target actually moved, so a paused cursor leaves attached views untouched.
            arguments
                obj
                wallDt (1,1) double {mustBeNonnegative}
            end
            g = obj.Generation;
            if obj.Rate == 0, return, end
            t = min(max(obj.Target + obj.Rate * wallDt, obj.Span(1)), obj.Span(2));
            if t ~= obj.Target
                obj.seek(t, obj.Boundary);
                g = obj.Generation;
            end
            if t == obj.Span(1) || t == obj.Span(2), obj.Rate = 0; end
        end

        function tf = isCurrent(obj, generation)
            tf = generation == obj.Generation;
        end
    end
end
