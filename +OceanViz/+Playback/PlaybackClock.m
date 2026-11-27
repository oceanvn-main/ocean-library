classdef PlaybackClock < handle
    % PLAYBACKCLOCK  Presentation time (GFX2): maps wall-clock time to simulation time. Pause, seek, speed change and
    %   reverse (negative speed) rebase the mapping, so they never advance or alter the recorded world. The wall clock
    %   is injectable (a function returning seconds) so behaviour is testable without waiting.
    %
    %   clk = OceanViz.Playback.PlaybackClock([0 10], 'Wall', @() toc(t0), 'Speed', 2);
    %   clk.play(); t = clk.now();  clk.seek(7); clk.pause(); clk.setSpeed(-1);
    % Time is clamped to the span; reaching an end pauses the clock. Every seek/pause/speed change bumps Generation,
    % which a consumer compares with the generation it started work under (see FrameSource.newRequest).

    properties (SetAccess = private)
        Span (1,2) double
        Speed (1,1) double = 1
        Playing (1,1) logical = false
        Generation (1,1) double = 0
    end

    properties (Access = private)
        Wall_
        T0_ (1,1) double          % presentation time at the last rebase
        W0_ (1,1) double          % wall time at the last rebase
    end

    methods
        function obj = PlaybackClock(span, opts)
            arguments
                span (1,2) double
                opts.Wall (1,1) function_handle = @() posixtime(datetime('now'))
                opts.Speed (1,1) double = 1
            end
            if span(2) < span(1), error('OceanViz:Playback:badSpan', 'Span must be [t0 t1] with t1 >= t0.'); end
            obj.Span = span; obj.Wall_ = opts.Wall; obj.Speed = opts.Speed; obj.T0_ = span(1); obj.W0_ = obj.Wall_();
        end

        function t = now(obj)
            if obj.Playing
                t = obj.T0_ + obj.Speed * (obj.Wall_() - obj.W0_);
                if t >= obj.Span(2) || t <= obj.Span(1)
                    t = min(max(t, obj.Span(1)), obj.Span(2));
                    obj.T0_ = t; obj.Playing = false; obj.Generation = obj.Generation + 1;   % end reached: pause
                end
            else
                t = obj.T0_;
            end
        end

        function play(obj)
            if obj.Playing, return; end
            obj.W0_ = obj.Wall_(); obj.Playing = true;
        end

        function pause(obj)
            obj.T0_ = obj.now(); obj.Playing = false; obj.Generation = obj.Generation + 1;
        end

        function seek(obj, t)
            obj.T0_ = min(max(t, obj.Span(1)), obj.Span(2)); obj.W0_ = obj.Wall_();
            obj.Generation = obj.Generation + 1;
        end

        function setSpeed(obj, speed)
            obj.T0_ = obj.now(); obj.W0_ = obj.Wall_(); obj.Speed = speed; obj.Generation = obj.Generation + 1;
        end
    end
end
