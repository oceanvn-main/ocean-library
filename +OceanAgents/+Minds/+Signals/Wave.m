classdef (Sealed) Wave
    % WAVE  A flow described, not stored: its value at any time is computed from a few numbers, so moving the clock
    % costs nothing and a threshold crossing is solved exactly instead of found by stepping (design D3, D13; AB1 3.3).
    %
    %   w = OceanAgents.Minds.Signals.Wave.sine(2, 0.25, 'Offset', 1);      % 1 + 2 sin(2 pi 0.25 (t - T0))
    %   w.valueAt(0:0.5:4)
    %   t = w.nextCrossing(2.5, 0, "up");                                   % first upward crossing of 2.5 after t = 0
    %
    %   Form  "constant": Offset
    %         "ramp"    : Offset + Slope (t - T0)
    %         "sine"    : Offset + Amplitude sin(2 pi Frequency (t - T0) + Phase)
    % A crossing is a strict change of side; touching the level without crossing it (a sine peak) is not a crossing.

    properties (SetAccess = immutable)
        Form string = "constant"
        Offset double = 0
        Slope double = 0
        Amplitude double = 0
        Frequency double = 0
        Phase double = 0
        T0 double = 0
    end

    properties (Constant)
        Forms = ["constant" "ramp" "sine"]
    end

    methods
        function obj = Wave(options)
            arguments
                options.Form = "constant"
                options.Offset = 0
                options.Slope = 0
                options.Amplitude = 0
                options.Frequency = 0
                options.Phase = 0
                options.T0 = 0
            end
            bad = @(msg) error('OceanAgents:Signals:BadValue', 'Wave: %s', msg);
            obj.Form = string(options.Form);
            if ~isscalar(obj.Form) || ~any(obj.Form == obj.Forms), bad('unknown Form.'); end
            names = ["Offset" "Slope" "Amplitude" "Frequency" "Phase" "T0"];
            for n = names
                v = options.(n);
                if ~(isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v)), bad(n + " must be a finite real scalar."); end
                obj.(n) = double(v);
            end
            if obj.Form == "sine" && obj.Frequency < 0, bad('Frequency must be non-negative.'); end
        end

        function v = valueAt(obj, t)
            switch obj.Form
                case "constant", v = obj.Offset + 0 * t;
                case "ramp",     v = obj.Offset + obj.Slope * (t - obj.T0);
                otherwise,       v = obj.Offset + obj.Amplitude * sin(2 * pi * obj.Frequency * (t - obj.T0) + obj.Phase);
            end
        end

        function t = nextCrossing(obj, level, after, direction)
            % NEXTCROSSING  First time strictly after `after` at which the wave crosses `level`; Inf when it never does.
            % direction: "up" | "down" | "either".
            arguments
                obj
                level (1,1) double {mustBeFinite}
                after (1,1) double {mustBeFinite}
                direction (1,1) string {mustBeMember(direction, ["up" "down" "either"])} = "either"
            end
            t = Inf;
            switch obj.Form
                case "constant"
                    return
                case "ramp"
                    if obj.Slope == 0, return; end
                    if (direction == "up" && obj.Slope < 0) || (direction == "down" && obj.Slope > 0), return; end
                    tc = obj.T0 + (level - obj.Offset) / obj.Slope;
                    if tc > after, t = tc; end
                otherwise
                    A = obj.Amplitude;  f = obj.Frequency;
                    if A == 0 || f == 0, return; end
                    s = (level - obj.Offset) / A;
                    if abs(s) >= 1, return; end                       % never reached, or only touched at a peak
                    base = [asin(s), pi - asin(s)];                   % angles where sin = s
                    rising = A * cos(base) > 0;                       % the wave increases there
                    x = 2 * pi * f * (after - obj.T0) + obj.Phase;    % angle at `after`
                    for k = 1:2
                        if (direction == "up" && ~rising(k)) || (direction == "down" && rising(k)), continue; end
                        m = floor((x - base(k)) / (2 * pi)) + 1;
                        tc = obj.T0 + (base(k) + 2 * pi * m - obj.Phase) / (2 * pi * f);
                        if tc <= after, tc = tc + 1 / f; end
                        t = min(t, tc);
                    end
            end
        end
    end

    methods
        function t = nextReach(obj, level, after, direction)
            % NEXTREACH  First time t >= after at which the wave is at or beyond level on the given side: "up" means
            % value >= level, "down" means value <= level. Touching counts (a peak equal to the level is reached).
            % This is the condition a threshold acts on, so skipping to it equals stepping (AB1 I8; critic finding 5).
            arguments
                obj
                level (1,1) double {mustBeFinite}
                after (1,1) double {mustBeFinite}
                direction (1,1) string {mustBeMember(direction, ["up" "down"])}
            end
            v0 = obj.valueAt(after);
            if (direction == "up" && v0 >= level) || (direction == "down" && v0 <= level)
                t = after;
                return
            end
            tol = 1e-9 * max(1, abs(after));
            % a crossing within rounding of `after` is AT `after` (the sample there fell a few ulps short); searching
            % from `after` itself would skip it and lose a whole period (critic AB1 round 2, I8)
            t = max(obj.nextCrossing(level, after - tol, direction), after);
            if obj.Form == "sine" && obj.Amplitude ~= 0 && obj.Frequency > 0
                s = (level - obj.Offset) / obj.Amplitude;
                if abs(abs(s) - 1) <= 1e-12                         % the level is a peak or a trough: touched only
                    theta = sign(s) * pi / 2;
                    x = 2 * pi * obj.Frequency * (after - tol - obj.T0) + obj.Phase;
                    m = ceil((x - theta) / (2 * pi));
                    tc = obj.T0 + (theta + 2 * pi * m - obj.Phase) / (2 * pi * obj.Frequency);
                    if tc < after - tol, tc = tc + 1 / obj.Frequency; end
                    t = min(t, max(tc, after));
                end
            end
        end
    end

    methods (Static)
        function w = constant(value)
            w = OceanAgents.Minds.Signals.Wave('Form', "constant", 'Offset', value);
        end

        function w = ramp(offset, slope, options)
            arguments
                offset
                slope
                options.T0 = 0
            end
            w = OceanAgents.Minds.Signals.Wave('Form', "ramp", 'Offset', offset, 'Slope', slope, 'T0', options.T0);
        end

        function w = sine(amplitude, frequency, options)
            arguments
                amplitude
                frequency
                options.Offset = 0
                options.Phase = 0
                options.T0 = 0
            end
            w = OceanAgents.Minds.Signals.Wave('Form', "sine", 'Amplitude', amplitude, 'Frequency', frequency, ...
                'Offset', options.Offset, 'Phase', options.Phase, 'T0', options.T0);
        end
    end
end
