classdef EpisodeMemory < OceanAgents.Memory.IMemory
    % EPISODEMEMORY  Experience kept as plain records, one per step (AB1 section 5). Written by its one writer with
    % records holding at least Episode, Observation, Action, Reward, Terminated, Truncated; queried by signal:
    %
    %   struct('Op', "last",    'Count', k)      -> Records (the k most recent, oldest first)
    %   struct('Op', "episode", 'Episode', e)    -> Records of episode e
    %   struct('Op', "count")                    -> Count

    properties (SetAccess = private)
        Records = struct('Episode', {}, 'Observation', {}, 'Action', {}, 'Reward', {}, 'Terminated', {}, 'Truncated', {})
        Capacity double = Inf
    end

    methods
        function obj = EpisodeMemory(id, options)
            arguments
                id (1,1) string
                options.Capacity (1,1) double {mustBePositive} = Inf
                options.BodySignature = []
            end
            obj@OceanAgents.Memory.IMemory(id, "episodes", 'BodySignature', options.BodySignature);
            obj.Capacity = options.Capacity;
        end

    end

    methods (Access = protected)
        function payload = answer(obj, q)
            if ~isfield(q, 'Op'), obj.badQuery_('a query needs Op.'); end
            switch string(q.Op)
                case "last"
                    k = min(q.Count, numel(obj.Records));
                    payload = struct('Records', obj.Records(end - k + 1:end));
                case "episode"
                    payload = struct('Records', obj.Records([obj.Records.Episode] == q.Episode));
                case "count"
                    payload = struct('Count', numel(obj.Records));
                otherwise
                    obj.badQuery_('unknown Op "%s".', string(q.Op));
            end
        end
    end

    methods (Access = protected)
        function store(obj, r)
            need = {'Episode', 'Observation', 'Action', 'Reward', 'Terminated', 'Truncated'};
            if ~(isstruct(r) && isscalar(r) && all(isfield(r, need)))
                obj.badQuery_('a record needs fields %s.', strjoin(need, ', '));
            end
            r = rmfield(r, setdiff(fieldnames(r), need));
            obj.Records(end + 1) = orderfields(r, obj.Records);
            if numel(obj.Records) > obj.Capacity, obj.Records(1) = []; end
        end
    end
end
