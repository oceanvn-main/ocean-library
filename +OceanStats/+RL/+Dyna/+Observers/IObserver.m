classdef (Abstract) IObserver < handle
    % IOBSERVER Passive Dyna event sink.
    methods
        function startRun(~), end
        function realTransition(~, ~), end
        function modelUpdated(~, ~), end
        function valueUpdated(~, ~), end
        function finishEpisode(~, ~), end
        function finishRun(~, ~), end
    end
end
