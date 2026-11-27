classdef (Abstract) TabularActionValues < OceanStats.RL.TD.Values.IValueFunction
    % TABULARACTIONVALUES  Shared storage of the dense action-value tables (Q, S x A, invalid actions stay at 0) with
    % per-entry visit counts. TabularQ and DoubleTabularQ differ only in how one transition updates the tables.
    properties (SetAccess = protected)
        Q double = []
        Counts double = []
    end
    methods
        function obj = TabularActionValues(validMask, initialValue)
            arguments
                validMask logical
                initialValue (1,1) double = 0
            end
            obj.Q = initialValue * ones(size(validMask));
            obj.Q(~validMask) = 0;
            obj.Counts = zeros(size(validMask));
        end
        function k = kind(~), k = "action"; end
        function x = estimates(obj, si), x = obj.Q(si, :); end
        function x = behaviourEstimates(obj, si), x = obj.Q(si, :); end
        function x = get(obj, si, ai), x = obj.Q(si, ai); end
        function n = touch(obj, si, ai)
            obj.Counts(si, ai) = obj.Counts(si, ai) + 1;
            n = obj.Counts(si, ai);
        end
        function update(obj, si, ai, alpha, y)
            obj.Q(si, ai) = obj.Q(si, ai) + alpha * (y - obj.Q(si, ai));
        end
        function T = zerosLike(obj), T = zeros(size(obj.Counts)); end
        function applyIncrement(obj, dT), obj.Q = obj.Q + dT; end
        function Q = finalQ(obj), Q = obj.Q; end
        function setTables(obj, Q, Counts)
            % SETTABLES  Restore Q and Counts from saved copies (e.g. a checkpoint). Same size as the current tables,
            % Q finite, Counts non-negative integers. Subclasses with extra tables (DoubleTabularQ) restore those
            % separately. The valid-action mask is not stored, so "invalid actions stay at 0" is the caller's
            % responsibility: restore only tables exported from an identically configured object. Entries where
            % the current table is invalid-by-construction cannot be detected here.
            arguments
                obj
                Q double
                Counts double
            end
            if ~isequal(size(Q), size(obj.Q)) || ~isequal(size(Counts), size(obj.Counts))
                error('OceanStats:RL:TD:Values:SizeMismatch', 'Q and Counts must be %s.', mat2str(size(obj.Q)));
            end
            if ~all(isfinite(Q(:))) || any(Counts(:) < 0 | Counts(:) ~= fix(Counts(:)))
                error('OceanStats:RL:TD:Values:BadTables', 'Q must be finite and Counts non-negative integers.');
            end
            obj.Q = Q;
            obj.Counts = Counts;
        end
    end
end
