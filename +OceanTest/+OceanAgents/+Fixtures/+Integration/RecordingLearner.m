classdef RecordingLearner < OceanAgents.Core.ILearner
    % RECORDINGLEARNER  Test decorator: forwards everything to an inner learner and keeps a copy of every Transition
    % it consumed, in order, so a test can replay the exact experience through an independent reference.
    properties (SetAccess = private)
        Inner
        Seen = {}
    end
    methods
        function obj = RecordingLearner(inner)
            obj.Inner = inner;
        end
        function d = describe(obj), d = obj.Inner.describe(); end
        function checkCompatible(obj, pd), obj.Inner.checkCompatible(pd); end
        function consume(obj, t)
            obj.Seen{end + 1} = t;
            obj.Inner.consume(t);
        end
        function n = update(obj, boundary), n = obj.Inner.update(boundary); end
        function s = exportState(obj), s = obj.Inner.exportState(); end
        function importState(obj, s), obj.Inner.importState(s); end
    end
end
