classdef NeuralV < OceanStats.RL.TD.Values.IValueFunction
    % NEURALV  State-value function V(s; theta) backed by an OceanDL.nn.Module neural network.
    properties (Constant)
        Latex = "V(s; \theta) = \text{NeuralNet}(s; \theta)"
        Reference = "Sutton & Barto (2018) Ch. 9, On-policy Prediction with Approximation"
    end
    properties (SetAccess = protected)
        Network           % OceanDL.nn.Module neural network backbone
        StateDim (1,1) double {mustBeInteger, mustBePositive} = 1
        Counts double = []
    end
    methods
        function obj = NeuralV(netModule, stateDim)
            arguments
                netModule
                stateDim (1,1) double = 1
            end
            obj.Network = netModule;
            obj.StateDim = stateDim;
        end
        function s = name(~), s = "neuralv"; end
        function k = kind(~), k = "state"; end

        function vVal = forward(obj, sState)
            vVal = obj.Network.forward(sState);
        end

        function x = estimates(obj, si)
            if isscalar(si) && si <= 10000
                sVec = zeros(1, obj.StateDim);
                if si <= obj.StateDim, sVec(si) = 1; else, sVec(1) = si; end
                vVal = obj.forward(sVec);
            else
                vVal = obj.forward(si);
            end
            x = vVal;
        end

        function x = behaviourEstimates(obj, si), x = obj.estimates(si); end
        function x = get(obj, si, ~), x = obj.estimates(si); end

        function n = touch(obj, si, ~)
            if isscalar(si)
                if isempty(obj.Counts), obj.Counts = zeros(1000, 1); end
                if si <= size(obj.Counts, 1)
                    obj.Counts(si) = obj.Counts(si) + 1;
                    n = obj.Counts(si);
                    return;
                end
            end
            n = 1;
        end

        function update(obj, ~, ~, ~, ~), end
        function T = zerosLike(obj), T = []; end
        function applyIncrement(obj, ~), end

        function coin = oneStep(obj, target, stepSize, si, ai, r, s2i, a2i, term, gamma, probsNext, valid2, stream)
            coin = [];
        end

        function v = stateValues(obj, idx)
            v = obj.estimates(idx);
        end
    end
end
