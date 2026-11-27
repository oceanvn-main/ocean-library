classdef NeuralQ < OceanStats.RL.TD.Values.IValueFunction
    % NEURALQ  Action-value function Q(s, a; theta) backed by an OceanDL.nn.Module neural network.
    properties (Constant)
        Latex = "Q(s, a; \theta) = \text{NeuralNet}(s; \theta)"
        Reference = "Mnih et al. (2015) Human-level control through deep reinforcement learning"
    end
    properties (SetAccess = protected)
        Network           % OceanDL.nn.Module neural network backbone
        NumActions (1,1) double {mustBeInteger, mustBePositive} = 1
        StateDim (1,1) double {mustBeInteger, mustBePositive} = 1
        Counts double = []
    end
    methods
        function obj = NeuralQ(netModule, numActions, stateDim)
            arguments
                netModule
                numActions (1,1) double {mustBeInteger, mustBePositive}
                stateDim (1,1) double = 1
            end
            obj.Network = netModule;
            obj.NumActions = numActions;
            obj.StateDim = stateDim;
        end
        function s = name(~), s = "neuralq"; end
        function k = kind(~), k = "action"; end
        
        function qVals = forward(obj, sState)
            % Forward pass through the neural network backbone
            if isnumeric(sState) && ismatrix(sState)
                qVals = obj.Network.forward(sState);
            else
                qVals = obj.Network.forward(sState);
            end
        end

        function x = estimates(obj, si)
            % Returns Q(s, :) values for a given state.
            %   si: scalar integer index -> one-hot vector of length StateDim
            %   si: numeric row vector    -> used directly as feature vector
            if isscalar(si) && isnumeric(si) && si == fix(si) && si >= 1 && si <= obj.StateDim
                % Integer state index: build one-hot feature vector
                sVec = zeros(1, obj.StateDim);
                sVec(si) = 1;
                x = obj.forward(sVec);
            else
                % Feature vector passed directly (e.g. for continuous states)
                x = obj.forward(si);
            end
        end

        function x = behaviourEstimates(obj, si), x = obj.estimates(si); end
        function x = get(obj, si, ai)
            qAll = obj.estimates(si);
            x = qAll(ai);
        end

        function n = touch(obj, si, ai)
            if isscalar(si)
                if isempty(obj.Counts), obj.Counts = zeros(1000, obj.NumActions); end
                if si <= size(obj.Counts, 1)
                    obj.Counts(si, ai) = obj.Counts(si, ai) + 1;
                    n = obj.Counts(si, ai);
                    return;
                end
            end
            n = 1;
        end

        function update(obj, ~, ~, ~, ~)
            % Neural update is performed via backprop in NeuralSemiGradientBackup or DQNBackup
        end

        function T = zerosLike(obj), T = []; end
        function applyIncrement(obj, ~), end

        function coin = oneStep(obj, target, stepSize, si, ai, r, s2i, a2i, term, gamma, probsNext, valid2, stream)
            % Standard 1-step sample evaluation
            coin = [];
        end

        function q = actionValues(obj, idx)
            q = obj.estimates(idx);
        end
    end
end
