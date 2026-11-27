classdef TimeContext
    % TIMECONTEXT  Immutable record of "when" for one time domain (TIME0_Time_Domains_Contract.md section 4).
    % Engine, history and views share it; a multi-domain frame carries one TimeContext per domain and never one implied
    % common time. Time alone is not an identity (reset and the first step share a tick): Commit is.
    %
    %   ctx = OceanAgents.Time.TimeContext('DomainID', "arm", 'Time', 12, 'Commit', 7);
    %   ctx.withGeneration(3)       % copy with another cursor generation
    properties (SetAccess = immutable)
        RunID string = "run"
        BranchID string = "main"
        Episode double = 1
        DomainID string = "world"
        Time double = 0
        Unit string = "tick"
        Phase string = ""
        Commit double = 0
        MappingRevision double = 0
        CursorGeneration double = 0
        SourceVersion double = 0
        ActiveSetRevision double = 0   % revision of the registered active set (kernel-backed domains)
    end
    methods
        function obj = TimeContext(opts)
            arguments
                opts.RunID (1,1) string = "run"
                opts.BranchID (1,1) string = "main"
                opts.Episode (1,1) double = 1
                opts.DomainID (1,1) string = "world"
                opts.Time (1,1) double = 0
                opts.Unit (1,1) string = "tick"
                opts.Phase (1,1) string = ""
                opts.Commit (1,1) double = 0
                opts.MappingRevision (1,1) double = 0
                opts.CursorGeneration (1,1) double = 0
                opts.SourceVersion (1,1) double = 0
                opts.ActiveSetRevision (1,1) double = 0
            end
            for f = string(fieldnames(opts))'
                obj.(f) = opts.(f);
            end
        end

        function obj = withGeneration(obj, g)
            o = obj.toStruct();
            o.CursorGeneration = g;
            c = namedargs2cell(o);
            obj = OceanAgents.Time.TimeContext(c{:});
        end

        function obj = withTime(obj, t)
            o = obj.toStruct();
            o.Time = t;
            c = namedargs2cell(o);
            obj = OceanAgents.Time.TimeContext(c{:});
        end

        function s = toStruct(obj)
            s = struct();
            for f = ["RunID" "BranchID" "Episode" "DomainID" "Time" "Unit" "Phase" "Commit" ...
                    "MappingRevision" "CursorGeneration" "SourceVersion" "ActiveSetRevision"]
                s.(f) = obj.(f);
            end
        end

        function tf = sameCommit(a, b)
            % SAMECOMMIT  Same stored state: run, branch, episode, domain and commit all equal (time alone is not enough).
            tf = a.RunID == b.RunID && a.BranchID == b.BranchID && a.Episode == b.Episode ...
                && a.DomainID == b.DomainID && a.Commit == b.Commit;
        end
    end
end
