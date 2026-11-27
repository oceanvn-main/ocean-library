classdef EvidenceRecord
    % EVIDENCERECORD  One observable event (or a claim about one) that a mind may use to revise a belief.
    % Evidence reaches an agent only inside its ActorView (view.Evidence), so an agent never learns about an event it
    % was not entitled to observe.
    %
    %   e = OceanAgents.Core.EvidenceRecord('EvidenceID', "E1-3-alice", 'SubjectID', "p1", 'Context', "cheap", ...
    %           'Outcome', "success", 'SourceKind', "direct", 'SourceID', "alice", 'EpisodeID', 1, 'Tick', 3);
    %
    %   Outcome     "success" | "failure" | "unknown"  (unknown = unattributable, e.g. obstructed; never a failure)
    %   SourceKind  "direct" (the event was observed) | "report" (someone's claim about an event)
    %   Quantity    what is assessed; A2 uses "completion" (default)
    % The same event delivered to several observers keeps the same EvidenceID; canonicalKey() detects an ID reused with
    % different content. See Planning/System/Game Theory/A2_Beliefs_Contract.md section 2.

    properties (SetAccess = immutable)
        EvidenceID string = ""
        SubjectID string = ""
        Context string = ""
        Quantity string = "completion"
        Outcome string = "unknown"
        SourceKind string = "direct"
        SourceID string = ""
        EpisodeID double = 0
        Tick double = 0
    end

    properties (Constant)
        Outcomes = ["success" "failure" "unknown"]
        SourceKinds = ["direct" "report"]
    end

    methods
        function obj = EvidenceRecord(options)
            arguments
                options.EvidenceID
                options.SubjectID
                options.Context
                options.Quantity = "completion"
                options.Outcome
                options.SourceKind = "direct"
                options.SourceID
                options.EpisodeID
                options.Tick = 0
            end
            requireFields_(options, {'EvidenceID', 'SubjectID', 'Context', 'Outcome', 'SourceID', 'EpisodeID'}, ...
                'EvidenceRecord');
            obj.EvidenceID = asString_(options.EvidenceID, 'EvidenceID');
            obj.SubjectID = asString_(options.SubjectID, 'SubjectID');
            obj.Context = asString_(options.Context, 'Context');
            obj.Quantity = asString_(options.Quantity, 'Quantity');
            obj.Outcome = asString_(options.Outcome, 'Outcome');
            if ~any(obj.Outcome == obj.Outcomes)
                fail_('InvalidRecord', 'Outcome must be one of: %s.', strjoin(obj.Outcomes, ', '));
            end
            obj.SourceKind = asString_(options.SourceKind, 'SourceKind');
            if ~any(obj.SourceKind == obj.SourceKinds)
                fail_('InvalidRecord', 'SourceKind must be one of: %s.', strjoin(obj.SourceKinds, ', '));
            end
            obj.SourceID = asString_(options.SourceID, 'SourceID');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.Tick = asId_(options.Tick, 'Tick');
            % identifiers are used to build lookup keys; control characters could make two keys collide
            ids = [obj.EvidenceID obj.SubjectID obj.Context obj.Quantity obj.SourceID];
            if any(cellfun(@(c) any(double(c) < 32 | double(c) == 127), cellstr(ids)))
                fail_('InvalidRecord', 'Evidence identifiers must not contain control characters.');
            end
        end

        function key = canonicalKey(obj)
            % CANONICALKEY  One string of every field, so an EvidenceID reused with different content is detectable.
            key = strjoin([obj.EvidenceID, obj.SubjectID, obj.Context, obj.Quantity, obj.Outcome, ...
                obj.SourceKind, obj.SourceID, string(obj.EpisodeID), string(obj.Tick)], char(31));
        end
    end
end
