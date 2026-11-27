classdef FictitiousPlayLearner < OceanAgents.Core.ILearner
    % FICTITIOUSPLAYLEARNER  Learns about the other players by fictitious play (Brown 1951), using the Game Theory
    % component OceanStats.GameTheory.Learning.FictitiousPlay (PG1 contract). After each committed step it reads the
    % opponents' revealed actions from the NEXT observation (one entry per opponent, in the component's opponent
    % order; 0 = not revealed) and updates the empirical beliefs. A step whose next view was not sensed is skipped. Rewards are not needed: it models the others.
    %
    %   fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1);       % my guess vs the opponent's put
    %   L  = OceanAgents.Learners.FictitiousPlayLearner(fp);

    properties (SetAccess = private)
        FP
        NumUpdates double = 0
    end

    properties (Access = private)
        Pending_ = {}
    end

    methods
        function obj = FictitiousPlayLearner(fp)
            arguments
                fp (1,1) OceanStats.GameTheory.Learning.FictitiousPlay
            end
            obj.FP = fp;
        end

        function d = describe(obj)
            d = struct('Name', "FictitiousPlayLearner", 'Version', "1", 'Gamma', [], 'Boundary', "step", ...
                'Opponents', obj.FP.Opponents);
        end

        function consume(obj, t)
            obj.Pending_{end + 1} = t;
        end

        function n = update(obj, boundary)
            n = 0;
            if boundary ~= "step", return; end
            for k = 1:numel(obj.Pending_)
                if ~obj.Pending_{k}.NextSensed, continue; end   % nothing new was seen: do not count the old view again
                seen = double(reshape(obj.Pending_{k}.NextObservation, 1, []));
                if numel(seen) ~= numel(obj.FP.Opponents)
                    error('OceanAgents:Learners:Incompatible', ...
                        'FictitiousPlayLearner expects %d revealed opponent actions in the observation, got %d.', ...
                        numel(obj.FP.Opponents), numel(seen));
                end
                seen(seen == 0) = NaN;
                if all(isnan(seen)), continue; end
                profile = NaN(1, numel(obj.FP.Opponents) + 1);
                profile(obj.FP.Opponents) = seen;
                obj.FP.observe(profile);
                n = n + 1;
            end
            obj.Pending_ = {};
            obj.NumUpdates = obj.NumUpdates + n;
        end

        function s = exportState(obj)
            s = struct('Belief', obj.FP.Belief.exportState(), 'NumUpdates', obj.NumUpdates);
        end

        function importState(obj, s)
            obj.FP.Belief.importState(s.Belief);
            obj.NumUpdates = s.NumUpdates;
        end
    end
end
