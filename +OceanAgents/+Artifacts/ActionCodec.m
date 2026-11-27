classdef ActionCodec
    % ACTIONCODEC  The one sanctioned translation between external action labels and table columns.
    % An environment names its actions by labels (e.g. Discrete(3, Start = 0) -> labels [0 1 2]); a table stores them
    % by column (1..A). Mixing the two up is silent and fatal (label 2 is column 3 here), so every label <-> column
    % conversion in OceanAgents goes through this value object and never through "label == column" arithmetic.
    %
    %   c = OceanAgents.Artifacts.ActionCodec([0 1 2]);
    %   c = OceanAgents.Artifacts.ActionCodec.fromSpace(OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0));
    %   k = c.column(2)          % 3            (unknown label -> OceanAgents:Artifacts:UnknownLabel)
    %   a = c.labelOf(3)         % 2
    %   tf = c.matches([0 1 2])  % true: same labels in the same column order
    %
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 6.1.

    properties (SetAccess = immutable)
        Labels double = zeros(1, 0)     % 1 x A external labels; Labels(k) is the label of column k
    end

    methods
        function obj = ActionCodec(labels)
            arguments
                labels {mustBeNumeric, mustBeReal}
            end
            labels = double(labels);
            if isempty(labels) || ~isvector(labels)
                error('OceanAgents:Core:InvalidRecord', 'ActionCodec needs a non-empty 1 x A vector of labels.');
            end
            if ~all(isfinite(labels))
                error('OceanAgents:Core:NonFinite', 'ActionCodec labels contain NaN or Inf.');
            end
            if numel(unique(labels)) ~= numel(labels)
                error('OceanAgents:Core:InvalidRecord', 'ActionCodec labels must be unique (got %s).', mat2str(labels));
            end
            obj.Labels = reshape(labels, 1, []);
        end

        function n = numActions(obj)
            % NUMACTIONS  A, the number of columns.
            n = numel(obj.Labels);
        end

        function k = column(obj, label)
            % COLUMN  Column(s) of external label(s). Every label must be known (no nearest match, no fallback).
            arguments
                obj
                label {mustBeNumeric}
            end
            [found, k] = ismember(double(label), obj.Labels);
            if isempty(label) || ~all(found(:))
                error('OceanAgents:Artifacts:UnknownLabel', 'Label(s) %s not in the action labels %s.', ...
                    mat2str(label), mat2str(obj.Labels));
            end
        end

        function label = labelOf(obj, k)
            % LABELOF  External label(s) of column(s) k (integers in 1..A).
            arguments
                obj
                k {mustBeNumeric}
            end
            A = numel(obj.Labels);
            if isempty(k) || any(k(:) ~= fix(k(:)) | k(:) < 1 | k(:) > A)
                error('OceanAgents:Artifacts:UnknownLabel', 'Column(s) %s outside 1..%d.', mat2str(k), A);
            end
            label = obj.Labels(k);
        end

        function tf = matches(obj, labels)
            % MATCHES  True when labels are exactly this codec's labels in the same column order.
            tf = isnumeric(labels) && isequal(reshape(double(labels), 1, []), obj.Labels);
        end

        function tf = sameLabelSet(obj, labels)
            % SAMELABELSET  True when labels are the same set as this codec's labels (order may differ).
            tf = isnumeric(labels) && numel(labels) == numel(obj.Labels) && ...
                isequal(sort(reshape(double(labels), 1, [])), sort(obj.Labels));
        end
    end

    methods (Static)
        function obj = fromSpace(space)
            % FROMSPACE  Labels Start:(Start+N-1) of an OceanStats.RL.Env.Spaces.Discrete action space.
            arguments
                space (1,1) OceanStats.RL.Env.Spaces.Discrete
            end
            obj = OceanAgents.Artifacts.ActionCodec(space.Start:(space.Start + space.N - 1));
        end
    end
end
