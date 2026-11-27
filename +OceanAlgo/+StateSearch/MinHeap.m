classdef MinHeap < handle
    % MINHEAP  Binary min-heap of (priority row, payload id). Priority rows are compared lexicographically, so
    % ties on the first key are broken by the next ones, e.g. [f, -g, insertionOrder] for A*.
    %
    %   h = OceanAlgo.StateSearch.MinHeap(numKeys);
    %   h.push(id, keys)              keys: 1 x numKeys finite doubles
    %   [id, keys] = h.pop()          smallest entry (error when empty)
    %   [id, keys] = h.peek()
    %   n = h.count();  tf = h.isEmpty()
    % Entries are never updated in place: callers push again and skip stale entries on pop ("lazy deletion").
    % Storage grows by capacity doubling. push/pop are O(log n).
    properties (SetAccess = private)
        NumKeys (1,1) double
    end
    properties (Access = private)
        Keys_ double
        Ids_ (:,1) double
        N_ (1,1) double = 0
    end
    methods
        function obj = MinHeap(numKeys, capacity)
            arguments
                numKeys (1,1) double {mustBeInteger, mustBePositive} = 1
                capacity (1,1) double {mustBeInteger, mustBePositive} = 64
            end
            obj.NumKeys = numKeys;
            obj.Keys_ = zeros(capacity, numKeys);
            obj.Ids_ = zeros(capacity, 1);
        end
        function push(obj, id, keys)
            if numel(keys) ~= obj.NumKeys || ~all(isfinite(keys))
                error('OceanAlgo:StateSearch:BadPriority', 'Priority must be %d finite numbers.', obj.NumKeys);
            end
            if obj.N_ == size(obj.Keys_, 1)
                obj.Keys_ = [obj.Keys_; zeros(size(obj.Keys_))];
                obj.Ids_ = [obj.Ids_; zeros(size(obj.Ids_))];
            end
            obj.N_ = obj.N_ + 1;
            obj.Keys_(obj.N_, :) = keys(:)';
            obj.Ids_(obj.N_) = id;
            obj.siftUp_(obj.N_);
        end
        function [id, keys] = pop(obj)
            [id, keys] = obj.peek();
            obj.Keys_(1, :) = obj.Keys_(obj.N_, :);
            obj.Ids_(1) = obj.Ids_(obj.N_);
            obj.N_ = obj.N_ - 1;
            if obj.N_ > 1, obj.siftDown_(1); end
        end
        function [id, keys] = peek(obj)
            if obj.N_ == 0
                error('OceanAlgo:StateSearch:EmptyHeap', 'The heap is empty.');
            end
            id = obj.Ids_(1); keys = obj.Keys_(1, :);
        end
        function n = count(obj), n = obj.N_; end
        function tf = isEmpty(obj), tf = obj.N_ == 0; end
    end
    methods (Access = private)
        function tf = less_(obj, i, j)
            % Lexicographic comparison of rows i and j.
            d = obj.Keys_(i, :) - obj.Keys_(j, :);
            k = find(d ~= 0, 1, 'first');
            tf = ~isempty(k) && d(k) < 0;
        end
        function swap_(obj, i, j)
            obj.Keys_([i j], :) = obj.Keys_([j i], :);
            obj.Ids_([i j]) = obj.Ids_([j i]);
        end
        function siftUp_(obj, i)
            while i > 1
                p = floor(i / 2);
                if ~obj.less_(i, p), return; end
                obj.swap_(i, p); i = p;
            end
        end
        function siftDown_(obj, i)
            while true
                l = 2 * i; r = l + 1; m = i;
                if l <= obj.N_ && obj.less_(l, m), m = l; end
                if r <= obj.N_ && obj.less_(r, m), m = r; end
                if m == i, return; end
                obj.swap_(i, m); i = m;
            end
        end
    end
end
