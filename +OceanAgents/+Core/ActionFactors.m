classdef (Sealed) ActionFactors
    % ACTIONFACTORS  An action made of parts (PG1 contract, factored actions): each factor has a name and its own
    % labels, and the environment's action is the combination. Joint actions are numbered 1..prod(sizes) with the
    % FIRST factor changing fastest (MATLAB's sub2ind order).
    %
    %   f = OceanAgents.Core.ActionFactors(["put" "guess"], {["even" "odd"], ["even" "odd"]});
    %   k = f.joint([2 1]);          % put odd, guess even -> 2
    %   sub = f.parts(3);            % [1 2]: put even, guess odd
    %   m = f.partMask(jointMask, "guess")   % which guess values have at least one permitted joint action

    properties (SetAccess = immutable)
        Names string = strings(1, 0)
        Labels cell = {}
        Sizes double = zeros(1, 0)
    end

    methods
        function obj = ActionFactors(names, labels)
            arguments
                names (1,:) string
                labels (1,:) cell
            end
            if numel(names) ~= numel(labels) || isempty(names) || numel(unique(names)) ~= numel(names)
                error('OceanAgents:Core:InvalidRecord', 'ActionFactors: one unique name per factor, with its labels.');
            end
            obj.Names = names;
            obj.Labels = cellfun(@(x) reshape(string(x), 1, []), labels, 'UniformOutput', false);
            obj.Sizes = cellfun(@numel, obj.Labels);
            if any(obj.Sizes < 1), error('OceanAgents:Core:InvalidRecord', 'ActionFactors: every factor needs a label.'); end
        end

        function n = numJoint(obj)
            n = prod(obj.Sizes);
        end

        function k = joint(obj, subs)
            % JOINT  The joint action number of one value per factor (subs: 1 x F indices).
            c = num2cell(subs);
            k = sub2ind([obj.Sizes 1], c{:});
        end

        function subs = parts(obj, k)
            % PARTS  The factor values (indices) of joint action k.
            c = cell(1, numel(obj.Sizes));
            [c{:}] = ind2sub([obj.Sizes 1], k);
            subs = cell2mat(c);
        end

        function f = index(obj, name)
            f = find(obj.Names == string(name), 1);
            if isempty(f), error('OceanAgents:Core:InvalidRecord', 'No action factor "%s".', name); end
        end

        function m = partMask(obj, jointMask, name)
            % PARTMASK  Values of factor name that appear in at least one permitted joint action.
            f = obj.index(name);
            m = false(1, obj.Sizes(f));
            for k = find(jointMask)
                s = obj.parts(k);
                m(s(f)) = true;
            end
        end
    end
end
