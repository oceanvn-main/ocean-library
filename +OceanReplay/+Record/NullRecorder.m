classdef NullRecorder < handle
    % NULLRECORDER  Records nothing. Pass it to IRecordable.run to execute a
    %   domain's computation exactly as it would run without recording
    %   (the reference side of verifyInvariance).
    %
    %   Example:
    %     out = recordable.run(OceanReplay.Record.NullRecorder());

    methods
        function point(~, ~, ~)
        end
    end
end
