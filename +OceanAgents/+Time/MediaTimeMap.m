classdef MediaTimeMap < handle
    % MEDIATIMEMAP  Timestamped media (video) mapped to experiment time (parent sections 5 and 9, TIME4).
    % Frame index alone is insufficient for variable-rate media: each frame carries a presentation timestamp (PTS, media
    % clock); an affine map  experiment = Offset + Scale * pts  ties the media clock to the domain, with a Revision that
    % changes whenever the mapping changes (cache keys include it).
    %
    %   m = OceanAgents.Time.MediaTimeMap(pts, 'Keyframes', [1 5 9], 'Offset', 2, 'Scale', 1);
    %   i = m.frameAt(7.3)            % latest frame with experiment time <= 7.3 (hold); 0 if before the first frame
    %   m.decodeStart(i)              % keyframe to decode from; reverse playback buffers decodeStart..i first
    %   m.block(i)                    % frame indices decodeStart(i):i, a bounded reverse-display block
    %   m.remap('Offset', 3)          % new Offset/Scale, Revision + 1
    %
    % Random access is not constant-time: reverse display costs decoding from the previous keyframe.
    properties (SetAccess = private)
        PTS double
        Keyframes double
        Offset double = 0
        Scale double = 1
        Revision double = 0
    end
    methods
        function obj = MediaTimeMap(pts, opts)
            arguments
                pts (:,1) double
                opts.Keyframes (:,1) double = 1
                opts.Offset (1,1) double = 0
                opts.Scale (1,1) double {mustBePositive} = 1
            end
            if any(diff(pts) <= 0), error('OceanAgents:Time:BadValue', 'PTS must be strictly increasing (presentation order).'); end
            if isempty(opts.Keyframes) || opts.Keyframes(1) ~= 1 || any(opts.Keyframes < 1 | opts.Keyframes > numel(pts))
                error('OceanAgents:Time:BadValue', 'Keyframes must be valid frame indices and include frame 1.');
            end
            obj.PTS = pts; obj.Keyframes = sort(unique(opts.Keyframes)); obj.Offset = opts.Offset; obj.Scale = opts.Scale;
        end

        function t = experimentTime(obj, i), t = obj.Offset + obj.Scale * obj.PTS(i); end

        function i = frameAt(obj, t)
            pts = (t - obj.Offset) / obj.Scale;
            i = find(obj.PTS <= pts, 1, 'last');
            if isempty(i), i = 0; end
        end

        function k = decodeStart(obj, i)
            k = obj.Keyframes(find(obj.Keyframes <= i, 1, 'last'));
        end

        function idx = block(obj, i), idx = obj.decodeStart(i):i; end

        function remap(obj, opts)
            arguments
                obj
                opts.Offset (1,1) double = obj.Offset
                opts.Scale (1,1) double {mustBePositive} = obj.Scale
            end
            obj.Offset = opts.Offset; obj.Scale = opts.Scale; obj.Revision = obj.Revision + 1;
        end
    end
end
