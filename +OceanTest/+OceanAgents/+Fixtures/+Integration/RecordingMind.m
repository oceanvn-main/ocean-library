classdef RecordingMind < OceanAgents.Core.IDecisionProcess
    % RECORDINGMIND  Test decorator: forwards every call to an inner mind and keeps value copies of the evidence each
    % call delivered and of every decision returned, so a test can rebuild exactly what the agent was shown.
    properties (SetAccess = private)
        Inner
        Delivered = {}          % one cell per delivery (beginEpisode view / observe NextView): 1xK EvidenceRecord
        Decisions = {}          % Decision objects, in order
        Contexts = strings(1, 0)
    end
    methods
        function obj = RecordingMind(inner)
            obj.Inner = inner;
        end
        function d = describe(obj), d = obj.Inner.describe(); end
        function checkCompatible(obj, o, a, r), obj.Inner.checkCompatible(o, a, r); end
        function beginEpisode(obj, view)
            obj.Delivered{end + 1} = view.Evidence;
            obj.Inner.beginEpisode(view);
        end
        function dec = decide(obj, view, stream)
            dec = obj.Inner.decide(view, stream);
            obj.Decisions{end + 1} = dec;
            obj.Contexts(end + 1) = view.Info.context;
        end
        function observe(obj, fb)
            obj.Delivered{end + 1} = fb.NextView.Evidence;
            obj.Inner.observe(fb);
        end
        function endEpisode(obj), obj.Inner.endEpisode(); end
        function s = exportState(obj), s = obj.Inner.exportState(); end
        function importState(obj, s), obj.Inner.importState(s); end
        function ev = allEvidence(obj)
            % ALLEVIDENCE  Every delivered EvidenceRecord, in delivery order (1 x K).
            ev = [obj.Delivered{:}];
            if isempty(ev), ev = OceanAgents.Core.EvidenceRecord.empty(1, 0); end
        end
    end
end
