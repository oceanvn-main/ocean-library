classdef Scheduler
    % SCHEDULER  Drives one component through integer ticks according to its time binding (design D13; AB1 I7, I8):
    %   "clock"    : the component is visited at every tick 0, 1, ..., Horizon
    %   "calendar" : it is visited at tick 0 and then only at component.nextTick(t) (skipping the ticks in between)
    % The binding changes how many visits are made (cost), never which signals come out (results).
    %
    %   c = OceanAgents.Components.Converters.ThresholdConverter(1.2, 0.3);
    %   c.receive(waveSignal);  c.useBinding("calendar");
    %   [ticks, out, visits] = OceanAgents.Components.Scheduler.run(c, @(t) reading(t), 400);

    methods (Static)
        function [ticks, out, visits] = run(component, readingAt, horizon)
            arguments
                component (1,1) OceanAgents.Components.IComponent
                readingAt (1,1) function_handle
                horizon (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if ~any(component.Binding == ["clock" "calendar"])
                error('OceanAgents:Components:BadBinding', 'Scheduler drives "clock" or "calendar" bindings, not "%s".', ...
                    component.Binding);
            end
            ticks = zeros(1, 0);
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
            visits = 0;
            t = 0;
            while t <= horizon
                emitted = component.receive(readingAt(t));
                visits = visits + 1;
                for e = emitted
                    if e.isRefusal(), continue; end
                    ticks(end + 1) = t; %#ok<AGROW>
                    out(end + 1) = e; %#ok<AGROW>
                end
                if component.Binding == "clock"
                    t = t + 1;
                else
                    t = component.nextTick(t);
                end
            end
        end
    end
end
