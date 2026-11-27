classdef (Abstract) IMemory < handle
    % IMEMORY  What an agent keeps: values, beliefs, experience. Any part may QUERY it through a signal; only its one
    % bound writer may WRITE it (design D7; AB1_Signal_Body_Contract.md section 5). A memory records the body signature
    % it was learned on (empty = body-independent), so after a body swap the agent knows what is different.
    %
    %   mem.bindWriter("learner")                    % once; a second binding throws Memory:WriterBound
    %   replies = mem.receive(querySignal)           % Role "query"  : record with field Op (+ arguments)
    %   replies = mem.receive(writeSignal)           % Role "command": only from the bound writer (Memory:NotWriter)

    properties (SetAccess = protected)
        ID string = ""
        Kind string = ""
        BodySignature = []                      % OceanAgents.Body.BodySignature or [] (body-independent)
        Accepts = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
    end

    properties (SetAccess = private)
        WriterID string = ""
        RouterID double = 0                     % the router it is attached to now (writes must arrive through it)
    end

    properties (Access = private)
        Owner_ = []                             % the mind it belongs to (handle identity; never exposed)
        AttachedAs_ string = ""                 % its id on the current router
        LastSeq_ double = 0                     % last accepted write (a replayed write is refused)
    end

    methods (Abstract, Access = protected)
        payload = answer(obj, query)             % query: struct with Op; payload: struct (refusal via BadQuery error)
        store(obj, record)                       % reached only through receive(), after the writer check
    end

    methods
        function tf = isOwned(obj)
            tf = ~isempty(obj.Owner_);
        end

        function obj = IMemory(id, kind, options)
            arguments
                id (1,1) string {mustBeNonzeroLengthText}
                kind (1,1) string
                options.BodySignature = []
            end
            obj.ID = id;
            obj.Kind = kind;
            obj.BodySignature = options.BodySignature;
            T = @(role) OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', role);
            obj.Accepts = [T("query") T("command")];
        end

        function bindWriter(obj, writerID)
            arguments
                obj
                writerID (1,1) string {mustBeNonzeroLengthText}
            end
            if strlength(obj.WriterID) > 0
                error('OceanAgents:Memory:WriterBound', 'Memory "%s" is already written by "%s".', obj.ID, obj.WriterID);
            end
            obj.WriterID = writerID;
        end

        function out = receive(obj, signal)
            switch signal.Type.Role
                case "query"
                    try
                        out = OceanAgents.Minds.Signals.Signal.ok(obj.answer(signal.Value));
                    catch err
                        if ~startsWith(err.identifier, 'OceanAgents:Memory:'), rethrow(err); end
                        out = OceanAgents.Minds.Signals.Signal.refusal(err.message);
                    end
                case "command"
                    if signal.Sender ~= obj.WriterID || strlength(obj.WriterID) == 0 || obj.RouterID == 0 || ...
                            signal.Via ~= obj.RouterID || signal.To ~= obj.AttachedAs_ || signal.Seq <= obj.LastSeq_
                        error('OceanAgents:Memory:NotWriter', ...
                            'Memory "%s" accepts writes only from "%s"; "%s" tried to write.', obj.ID, obj.WriterID, ...
                            signal.Sender);
                    end
                    obj.store(signal.Value);
                    obj.LastSeq_ = signal.Seq;
                    out = OceanAgents.Minds.Signals.Signal.ok();
                otherwise
                    out = OceanAgents.Minds.Signals.Signal.refusal("a memory answers queries and accepts writes only");
            end
        end
    end

    methods (Access = {?OceanAgents.Minds.Signals.Router})
        function joinRouter_(obj, routerID, owner, id)
            % JOINROUTER_  A memory belongs to one agent (critic AB1 rounds 2-3). An owned memory joins only routers
            % of its own mind (a body swap); an unowned one joins one plain router, or is claimed by a mind.
            if ~isempty(obj.Owner_)
                if isempty(owner) || obj.Owner_ ~= owner
                    error('OceanAgents:Memory:WriterBound', 'Memory "%s" already belongs to another agent.', obj.ID);
                end
            elseif ~isempty(owner)
                obj.Owner_ = owner;
            elseif obj.RouterID ~= 0 && obj.RouterID ~= routerID
                error('OceanAgents:Memory:WriterBound', 'Memory "%s" is already attached to another router.', obj.ID);
            end
            obj.RouterID = routerID;
            obj.AttachedAs_ = string(id);
            obj.LastSeq_ = 0;
        end
    end

    methods (Access = {?OceanAgents.Minds.SignalMind})
        function claimOwner_(obj, mind)
            % CLAIMOWNER_  A mind takes this memory as its own (addMemory); another mind's memory is refused.
            if ~isempty(obj.Owner_) && obj.Owner_ ~= mind
                error('OceanAgents:Memory:WriterBound', 'Memory "%s" belongs to another mind.', obj.ID);
            end
            obj.Owner_ = mind;
        end
    end

    methods (Access = protected)
        function badQuery_(obj, fmt, varargin)
            error('OceanAgents:Memory:BadQuery', ['Memory "%s": ' fmt], obj.ID, varargin{:});
        end
    end
end
