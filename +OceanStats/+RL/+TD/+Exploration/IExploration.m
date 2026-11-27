classdef (Abstract) IExploration < OceanStats.RL.TD.ITDComponent
    % IEXPLORATION  Strategy: the behaviour policy as a function of the action values.
    %   p = rule.probabilities(q, valid, progress)   (1 x A) probabilities, 0 for invalid actions
    %   a = rule.select(q, valid, progress, stream)  index of a sampled action
    % q is a (1 x A) row, valid a (1 x A) logical mask, progress in [0,1] is the fraction of training done
    % (lets a rule decay), stream the RandStream that owns ALL randomness (no hidden global rng).
    methods (Abstract)
        p = probabilities(obj, q, valid, progress)
    end
    methods
        function a = select(obj, q, valid, progress, stream)
            p = obj.probabilities(q, valid, progress);
            c = cumsum(p);
            a = find(rand(stream) * c(end) <= c, 1);
        end
    end
end
