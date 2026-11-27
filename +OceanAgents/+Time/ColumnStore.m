classdef ColumnStore < handle
    % COLUMNSTORE  Homogeneous component column with stable entity identity and an active row set (parent sections 4, 8).
    % One batched update replaces a per-entity loop; deleted entities leave no stale row visible and their row is reused.
    %
    %   cs = OceanAgents.Time.ColumnStore();
    %   cs.add(7, 100); cs.add(9, 50);
    %   cs.applyBatch(@(v) v - 2 * 3);       % one vectorised call over the ACTIVE rows (elementwise: bitwise equal to a loop)
    %   cs.setActive(9, false);               % dormant rows are skipped
    %   cs.value(7)
    %
    % Equivalence level: bitwise for elementwise updates; a reduction (sum) may change order in a batch and then carries
    % a tolerance policy, not bitwise identity.
    properties (SetAccess = private)
        IDs double = zeros(0, 1)          % entity ID per row (0 = free row)
        Values double = zeros(0, 1)
        Active logical = false(0, 1)
    end
    methods
        function row = add(obj, id, value)
            if any(obj.IDs == id), error('OceanAgents:Time:BadValue', 'Entity %g already present.', id); end
            row = find(obj.IDs == 0, 1);
            if isempty(row)
                row = numel(obj.IDs) + 1;
            end
            obj.IDs(row, 1) = id; obj.Values(row, 1) = value; obj.Active(row, 1) = true;
        end

        function remove(obj, id)
            row = obj.row_(id);
            obj.IDs(row) = 0; obj.Values(row) = NaN; obj.Active(row) = false;
        end

        function setActive(obj, id, tf), obj.Active(obj.row_(id)) = tf; end

        function v = value(obj, id), v = obj.Values(obj.row_(id)); end

        function n = applyBatch(obj, fn)
            m = obj.Active & obj.IDs ~= 0;
            obj.Values(m) = fn(obj.Values(m));
            n = nnz(m);
        end
    end
    methods (Access = private)
        function r = row_(obj, id)
            r = find(obj.IDs == id, 1);
            if isempty(r), error('OceanAgents:Time:BadValue', 'Unknown entity %g.', id); end
        end
    end
end
