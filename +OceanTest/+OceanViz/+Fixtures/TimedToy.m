classdef TimedToy < OceanReplay.Record.IRecordable
    % TIMEDTOY  Recordable with explicit commit times and episodes (irregular spacing, two episodes), for playback tests.
    properties
        Times (1,:) double = [0 0.5 0.6 3 3.1 10]
        Episodes (1,:) double = [1 1 1 1 1 1]
    end
    methods
        function obj = TimedToy(times, episodes)
            if nargin > 0, obj.Times = times; obj.Episodes = episodes; end
        end
        function axes = axesInfo(~)
            axes = OceanReplay.Core.Axis("commit", []);
        end
        function [c, v] = catalog(~)
            C = @OceanReplay.Core.ChannelInfo;
            c = [C("tick", "scalar", 'Axes', "commit"), C("episode", "scalar", 'Axes', "commit"), ...
                 C("position", "points", 'Axes', "commit", 'ValueShape', [1 2])];
            v = OceanReplay.Core.ViewInfo.empty;
        end
        function out = run(obj, recorder)
            for i = 1:numel(obj.Times)
                x = [i 100 + i]; t = obj.Times(i); e = obj.Episodes(i);
                recorder.point(struct('commit', i), struct('tick', @() t, 'episode', @() e, 'position', @() x));
            end
            out = obj.Times;
        end
    end
end
