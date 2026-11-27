classdef TableMemory < OceanAgents.Memory.IMemory
    % TABLEMEMORY  An action-value table kept by label: the agent's TabularQ, with the action label of every column.
    % The table is written only by its bound learner, which holds the same TabularQ (the learner's own update is the
    % one sanctioned write path, checked when the writer is bound). Queries by signal (AB1 section 5):
    %
    %   struct('Op', "value", 'State', s, 'Label', "left")   -> Value
    %   struct('Op', "row",   'State', s)                    -> Values (1 x A), Labels
    %   struct('Op', "best",  'State', s)                    -> Label (first maximum), Value
    %
    % After a body swap, TableMemory.transfer keeps the values of the labels both bodies share; new labels start at
    % the initial value (actions are stored by label, never by column position).

    properties (SetAccess = private)
        Labels string
    end

    properties (Access = private)
        Table                                   % OceanStats.RL.TD.Values.TabularQ (handle; never handed out)
    end

    methods
        function obj = TableMemory(id, table, labels, options)
            arguments
                id (1,1) string
                table (1,1) OceanStats.RL.TD.Values.TabularActionValues
                labels string
                options.BodySignature = []
            end
            obj@OceanAgents.Memory.IMemory(id, "table", 'BodySignature', options.BodySignature);
            labels = reshape(labels, 1, []);
            if numel(labels) ~= size(table.Q, 2) || numel(unique(labels)) ~= numel(labels)
                error('OceanAgents:Memory:BadQuery', 'TableMemory: one unique label per table column is required.');
            end
            obj.Table = table;
            obj.Labels = labels;
        end

        function tf = isUpdatedBy(obj, learner)
            % ISUPDATEDBY  True when learner updates this very table.
            tf = isprop(learner, 'Q') && learner.Q == obj.Table;
        end

        function Q = values(obj)
            % VALUES  A copy of the table (S x A), by column in Labels order.
            Q = obj.Table.Q;
        end

        function bindLearner(obj, learner, learnerID)
            % BINDLEARNER  Bind the learner that owns the write path; it must update this very table.
            if ~obj.isUpdatedBy(learner)
                error('OceanAgents:Memory:NotWriter', 'TableMemory "%s": the learner does not update this table.', obj.ID);
            end
            obj.bindWriter(learnerID);
        end

    end

    methods (Access = protected)
        function payload = answer(obj, q)
            if ~(isfield(q, 'Op') && isfield(q, 'State')), obj.badQuery_('a query needs Op and State.'); end
            s = q.State;
            if ~(isnumeric(s) && isscalar(s) && s >= 1 && s <= size(obj.Table.Q, 1) && s == fix(s))
                obj.badQuery_('State %s is not a row of the table.', mat2str(s));
            end
            row = obj.Table.estimates(s);
            switch string(q.Op)
                case "value"
                    k = find(obj.Labels == string(q.Label), 1);
                    if isempty(k), obj.badQuery_('unknown label "%s".', string(q.Label)); end
                    payload = struct('Value', row(k));
                case "row"
                    payload = struct('Values', row, 'Labels', obj.Labels);
                case "best"
                    [v, k] = max(row);
                    payload = struct('Label', obj.Labels(k), 'Value', v);
                otherwise
                    obj.badQuery_('unknown Op "%s".', string(q.Op));
            end
        end
    end

    methods (Access = protected)
        function store(obj, ~)
            obj.badQuery_('the table is written by its bound learner''s own update, not by write signals.');
        end
    end

    methods (Static)
        function Q = transfer(old, newLabels, newValid, initialValue)
            % TRANSFER  Values for a new body's labels: shared labels keep their columns, new labels start at
            % initialValue, labels the new body lacks are dropped. Returns a new TabularQ.
            Q = OceanStats.RL.TD.Values.TabularQ(newValid, initialValue);
            T = Q.Q;
            [shared, iOld] = ismember(newLabels, old.Labels);
            T(:, shared) = old.Table.Q(:, iOld(shared));
            T(~newValid) = Q.Q(~newValid);
            Q.setTables(T, zeros(size(T)));
        end
    end
end
