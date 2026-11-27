classdef QueryCache < handle
    % QUERYCACHE  Memoises PURE query results (parent section 8). The key must carry everything the answer depends on:
    % branch, episode, domain, commit, source version and channel; use QueryCache.key to build it. Bounded by entry count
    % (least recently used out), so memory stays bounded.
    %
    %   c = OceanAgents.Time.QueryCache('Capacity', 256);
    %   k = OceanAgents.Time.QueryCache.key(ctx, "position");
    %   [hit, v] = c.get(k);   if ~hit, v = compute(); c.put(k, v); end
    properties (SetAccess = private)
        Hits double = 0
        Misses double = 0
    end
    properties (Access = private)
        Capacity_ double = 256
        Keys_ string = strings(0, 1)      % least recently used first
        Vals_ cell = cell(0, 1)
    end
    methods (Static)
        function k = key(ctx, channel, extra)
            arguments
                ctx
                channel (1,1) string
                extra (1,1) string = ""
            end
            k = strjoin([ctx.RunID, ctx.BranchID, string(ctx.Episode), ctx.DomainID, string(ctx.Commit), ...
                string(ctx.MappingRevision), string(ctx.SourceVersion), string(ctx.ActiveSetRevision), channel, extra], "|");
        end
    end
    methods
        function obj = QueryCache(opts)
            arguments
                opts.Capacity (1,1) double {mustBePositive} = 256
            end
            obj.Capacity_ = opts.Capacity;
        end

        function [hit, v] = get(obj, k)
            i = find(obj.Keys_ == k, 1);
            if isempty(i)
                hit = false; v = []; obj.Misses = obj.Misses + 1;
                return
            end
            hit = true; v = obj.Vals_{i}; obj.Hits = obj.Hits + 1;
            obj.Keys_(end+1, 1) = obj.Keys_(i); obj.Vals_{end+1, 1} = v;
            obj.Keys_(i) = []; obj.Vals_(i) = [];
        end

        function put(obj, k, v)
            i = find(obj.Keys_ == k, 1);
            if ~isempty(i), obj.Keys_(i) = []; obj.Vals_(i) = []; end
            obj.Keys_(end+1, 1) = k; obj.Vals_{end+1, 1} = v;
            while numel(obj.Keys_) > obj.Capacity_
                obj.Keys_(1) = []; obj.Vals_(1) = [];
            end
        end

        function n = count(obj), n = numel(obj.Keys_); end
        function clear(obj), obj.Keys_ = strings(0, 1); obj.Vals_ = cell(0, 1); end
    end
end
