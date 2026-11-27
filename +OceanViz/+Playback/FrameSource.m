classdef FrameSource < handle
    % FRAMESOURCE  Time-indexed, causal reads of a Replay recording (GFX2). Unlike Recording.get (nearest stored
    %   point, for exact inspection) at(t) HOLDS the latest stored commit whose time is <= t: it never shows a sample
    %   from the future, never extrapolates, and reports staleness and gaps.
    %
    %   src = OceanViz.Playback.FrameSource(rec, ["position" "ended"], 'TimeChannel', "tick", 'CacheBytes', 2e6);
    %   f = src.at(2.5, 'Episode', 1);   f.Available, f.Commit, f.Time, f.Staleness, f.Gap, f.Values.position
    %   g = src.newRequest();            src.prefetch(commits, g)     % superseded by the next newRequest
    %
    % Time index: the commit axis, a scalar time channel and an optional scalar episode channel over that axis.
    % Times must be nondecreasing within an episode (else error); spacing may be irregular, any seek order is fine.
    % The decoded-value cache is byte-limited (least recently used evicted). Prefetch is cooperative: it checks the
    % request generation before every read and stops as soon as a newer request exists.

    properties (SetAccess = private)
        Recording
        Channels (1,:) string
        CacheBytes (1,1) double
        GapThreshold (1,1) double
        Generation (1,1) double = 0
        Hits (1,1) double = 0
        Misses (1,1) double = 0
        CachedBytes (1,1) double = 0
    end

    properties (Access = private)
        Commit_ double            % stored commits, recording order
        Time_ double
        Episode_ double
        Cache_ = []               % containers.Map key -> value
        Order_ string = strings(1, 0)   % LRU order, oldest first
    end

    methods
        function obj = FrameSource(rec, channels, opts)
            arguments
                rec (1,1) OceanReplay.Core.Recording
                channels (1,:) string
                opts.TimeChannel (1,1) string = "tick"
                opts.EpisodeChannel (1,1) string = "episode"
                opts.CacheBytes (1,1) double {mustBePositive} = 4e6
                opts.GapThreshold (1,1) double = Inf
            end
            obj.Recording = rec; obj.Channels = channels; obj.CacheBytes = opts.CacheBytes;
            obj.GapThreshold = opts.GapThreshold;
            [c, t] = rec.series(opts.TimeChannel);
            obj.Commit_ = c(:); obj.Time_ = t(:);
            if any(rec.names() == opts.EpisodeChannel)
                [c2, e] = rec.series(opts.EpisodeChannel);
                [~, loc] = ismember(obj.Commit_, c2(:));
                if any(loc == 0), error('OceanViz:Playback:coverage', 'Episode channel misses stored commits of the time channel.'); end
                obj.Episode_ = e(loc);
            else
                obj.Episode_ = ones(size(obj.Commit_));
            end
            for e = unique(obj.Episode_).'
                k = find(obj.Episode_ == e);
                [~, o] = sort(obj.Commit_(k)); k = k(o);
                if any(diff(obj.Time_(k)) < 0)
                    error('OceanViz:Playback:timeOrder', 'Time is not nondecreasing within episode %g.', e);
                end
            end
            obj.Cache_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function e = episodes(obj)
            e = unique(obj.Episode_).';
        end

        function [t0, t1] = span(obj, episode)
            t = obj.Time_(obj.Episode_ == episode); t0 = min(t); t1 = max(t);
        end

        function f = at(obj, t, opts)
            % Causal hold: the newest stored commit of the episode with time <= t.
            arguments
                obj
                t (1,1) double
                opts.Episode (1,1) double = NaN
            end
            e = opts.Episode; if isnan(e), e = obj.Episode_(1); end
            k = find(obj.Episode_ == e);
            [~, o] = sort(obj.Commit_(k)); k = k(o);
            j = find(obj.Time_(k) <= t, 1, 'last');
            f = struct('Available', false, 'Episode', e, 'Commit', NaN, 'Time', NaN, 'RequestedTime', t, ...
                'Staleness', NaN, 'Gap', false, 'Values', struct());
            if isempty(j), return; end                         % before the first stored sample: nothing to show
            commit = obj.Commit_(k(j));
            f.Available = true; f.Commit = commit; f.Time = obj.Time_(k(j));
            f.Staleness = t - f.Time; f.Gap = f.Staleness > obj.GapThreshold;
            for n = obj.Channels
                f.Values.(n) = obj.value_(n, commit);
            end
        end

        function g = newRequest(obj)
            % Start a new request generation (a seek or pause); earlier prefetches stop at their next check.
            obj.Generation = obj.Generation + 1; g = obj.Generation;
        end

        function n = prefetch(obj, commits, generation)
            % Load the channels of the given commits into the cache while `generation` is still current.
            n = 0;
            for c = reshape(commits, 1, [])
                if generation ~= obj.Generation, return; end
                for ch = obj.Channels, obj.value_(ch, c); end
                n = n + 1;
            end
        end
    end

    methods (Access = private)
        function v = value_(obj, name, commit)
            key = char(name + "|" + commit);
            if isKey(obj.Cache_, key)
                v = obj.Cache_(key); obj.Hits = obj.Hits + 1;
                obj.Order_(obj.Order_ == key) = []; obj.Order_(end+1) = key;
                return
            end
            obj.Misses = obj.Misses + 1;
            v = obj.Recording.get(name, struct('commit', commit), "exact");
            obj.Cache_(key) = v; obj.Order_(end+1) = key; obj.CachedBytes = obj.CachedBytes + numel(v) * 8;
            while obj.CachedBytes > obj.CacheBytes && numel(obj.Order_) > 1
                old = char(obj.Order_(1)); obj.Order_(1) = [];
                obj.CachedBytes = obj.CachedBytes - numel(obj.Cache_(old)) * 8;
                remove(obj.Cache_, old);
            end
        end
    end
end
