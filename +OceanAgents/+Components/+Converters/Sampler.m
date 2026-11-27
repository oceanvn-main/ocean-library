classdef Sampler < OceanAgents.Components.IConverter
    % SAMPLER  A wave -> a number stream: when triggered at time t it emits the wave's value at t; nextChange gives the
    % next sampling instant T0 + k / Rate (AB1 section 3.6).
    %
    %   s = OceanAgents.Components.Converters.Sampler(2);        % two samples per unit time, sense role
    %   replies = s.receive(waveSignal);                         % one sample at the signal's stamped time

    properties (SetAccess = private)
        Rate double
    end

    properties (Access = private)
        T0_ = 0
    end

    methods
        function obj = Sampler(rate, options)
            arguments
                rate (1,1) double {mustBePositive, mustBeFinite}
                options.Role (1,1) string = "sense"
                options.ID (1,1) string = "sampler"
            end
            T = @(varargin) OceanAgents.Minds.Signals.SignalType(varargin{:});
            in = T('Shape', "wave", 'ValueKind', "number", 'Role', options.Role);
            out = T('Shape', "stream", 'ValueKind', "number", 'Role', options.Role, 'Rate', rate);
            obj.setDeclarations_(options.ID, in, out, ["calendar" "clock"]);
            obj.InType = in;
            obj.OutType = out;
            obj.Rate = rate;
        end

        function t = nextChange(obj, after)
            k = floor((after - obj.T0_) * obj.Rate) + 1;
            t = obj.T0_ + k / obj.Rate;
            if t <= after, t = t + 1 / obj.Rate; end
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(~, signal)
            ok = ~isnan(signal.Time);
            reason = "a sample needs a stamped time";
        end

        function out = act(obj, signal)
            obj.T0_ = signal.Value.T0;
            out = OceanAgents.Minds.Signals.Signal.make(obj.OutType, signal.Value.valueAt(signal.Time));
        end
    end
end
