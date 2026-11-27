classdef CommitLog < handle
    % COMMITLOG  History source of one time domain: the state stored at each committed instant (TIME0 section 5, capture
    % boundary; parent section 7.1). Reads hold the latest commit at or before the target; nothing is extrapolated and
    % outside the stored coverage the answer is "unavailable", never an invented state.
    %
    %   log = OceanAgents.Time.CommitLog('MaxCommits', 1e5);
    %   log.append(ctx, struct('pos', 3));          % KernelDomain does this at each commit
    %   r = log.at(7, "after");                     % r.Status "ok"|"unavailable", r.Context, r.State
    %   r = log.at(7, "before");                    % last commit strictly before t (state before events at t)
    %
    % Times are nondecreasing (reset and first step may share a time); the Commit counter is the identity. A byte-style
    % bound is kept as MaxCommits: the oldest commits are dropped and Coverage moves (memory stays bounded).
    properties (SetAccess = private)
        Version double = 0
        Dropped double = 0
    end
    properties
        MaxCommits double = Inf
    end
    properties (Access = private)
        Time_ double = zeros(0, 1)
        Ctx_ cell = cell(0, 1)
        Ep_ double = zeros(0, 1)       % episode of each commit (nondecreasing); time restarts with a new episode
        State_ cell = cell(0, 1)
    end
    methods
        function obj = CommitLog(opts)
            arguments
                opts.MaxCommits (1,1) double = Inf
            end
            obj.MaxCommits = opts.MaxCommits;
        end

        function append(obj, ctx, state)
            if ~isempty(obj.Time_)
                if ctx.Episode < obj.Ep_(end) || (ctx.Episode == obj.Ep_(end) && ctx.Time < obj.Time_(end))
                    error('OceanAgents:Time:Retroactive', 'CommitLog: commit at %g (episode %g) after %g (episode %g).', ...
                        ctx.Time, ctx.Episode, obj.Time_(end), obj.Ep_(end));
                end
            end
            obj.Ep_(end+1, 1) = ctx.Episode;
            obj.Time_(end+1, 1) = ctx.Time;
            obj.Ctx_{end+1, 1} = ctx;
            obj.State_{end+1, 1} = state;
            obj.Version = obj.Version + 1;
            over = numel(obj.Time_) - obj.MaxCommits;
            if over > 0
                obj.Time_(1:over) = []; obj.Ep_(1:over) = []; obj.Ctx_(1:over) = []; obj.State_(1:over) = [];
                obj.Dropped = obj.Dropped + over;
            end
        end

        function n = count(obj), n = numel(obj.Time_); end

        function c = coverage(obj, episode)
            % COVERAGE  [first last] stored time of an episode (default: the latest), [] when empty.
            arguments
                obj
                episode double = []
            end
            [a, b] = obj.block_(episode);
            if a == 0, c = []; else, c = [obj.Time_(a) obj.Time_(b)]; end
        end

        function r = at(obj, t, boundary, episode)
            % AT  Read within one episode (default: the latest); times of different episodes are never mixed.
            arguments
                obj
                t (1,1) double
                boundary (1,1) string = "after"
                episode double = []
            end
            [a, b] = obj.block_(episode);
            if a == 0
                k = 0;
            elseif boundary == "after"
                k = lastLE_(obj.Time_(a:b), t, false);
            elseif boundary == "before"
                k = lastLE_(obj.Time_(a:b), t, true);
            else
                error('OceanAgents:Time:BadValue', 'boundary must be "after" or "before".');
            end
            if k > 0, k = k + a - 1; end
            if k == 0
                r = struct('Status', "unavailable", 'Context', [], 'State', [], 'Staleness', NaN);
            else
                r = struct('Status', "ok", 'Context', obj.Ctx_{k}, 'State', obj.State_{k}, ...
                    'Staleness', t - obj.Time_(k));
            end
        end
    end
    methods (Access = private)
        function [a, b] = block_(obj, episode)
            % First and last index of the episode's commits (0, 0 when none).
            a = 0; b = 0;
            if isempty(obj.Ep_), return, end
            if isempty(episode), episode = obj.Ep_(end); end
            idx = find(obj.Ep_ == episode);
            if ~isempty(idx), a = idx(1); b = idx(end); end
        end
    end
end

function k = lastLE_(x, t, strict)
% Largest index with x(k) <= t (or < t when strict); 0 when none. x is nondecreasing; binary search.
lo = 0; hi = numel(x);
while lo < hi
    mid = ceil((lo + hi) / 2);
    if (strict && x(mid) < t) || (~strict && x(mid) <= t)
        lo = mid;
    else
        hi = mid - 1;
    end
end
k = lo;
end
