classdef Trainer < handle
    % TRAINER  Minibatch training loop: VectorizedEngine → BackpropEngine → optimizer.
    %
    %   Orchestrates a full supervised training run on an NNGraph.
    %   Each epoch: data is shuffled, split into minibatches, and each batch
    %   performs a forward pass, backward pass, and parameter update.
    %
    %   USAGE:
    %     % Build graph
    %     spec  = OceanDL.Graph.NNTopologySpec.make( ...
    %                 [2, 8, 1], {'linear','relu','sigmoid'});
    %     graph = OceanDL.Graph.NNGraph.fromSpec(spec);
    %
    %     % Choose loss and optimizer factory
    %     loss    = OceanStats.Loss.Classification.BinaryBCELoss();
    %     optFn   = @() OceanStats.Optim.AdamOptimizer('LR', 0.01);
    %
    %     % Train
    %     trainer = OceanDL.Simulation.Trainer(graph, loss, optFn);
    %     history = trainer.fit(X_train, Y_train, 'Epochs', 500, 'BatchSize', 32);
    %     % history.Loss    — (1 x nEpochs) mean loss per epoch
    %
    %     % Inference
    %     eng = OceanDL.Simulation.VectorizedEngine();
    %     [Y_pred, ~] = eng.forward(graph, X_test);
    %
    %   See also: OceanDL.Simulation.VectorizedEngine
    %             OceanDL.Simulation.BackpropEngine
    %             OceanStats.Optim.IOptimizer

    properties (Access = private)
        Graph_    % NNGraph — mutated in-place during training
        Loss_     % ILoss   — loss function strategy
        OptFn_    % function_handle — called to create a fresh IOptimizer per param
        FwdEng_   % VectorizedEngine
        BwdEng_   % BackpropEngine
        OptsW_    % cell{nL} of IOptimizer for weight matrices
        OptsB_    % cell{nL} of IOptimizer for bias vectors
    end

    methods
        function obj = Trainer(graph, loss, optimizerFn)
            % TRAINER  Construct and initialise optimizer instances.
            %
            %   trainer = Trainer(graph, loss, optimizerFn)
            %
            %   optimizerFn — zero-arg function handle that returns an IOptimizer.
            %                 e.g. @() OceanStats.Optim.AdamOptimizer('LR', 0.001)
            %                 Called once per trainable parameter tensor.
            arguments
                graph       OceanDL.Graph.NNGraph
                loss        OceanStats.Loss.ILoss
                optimizerFn function_handle
            end
            obj.Graph_  = graph;
            obj.Loss_   = loss;
            obj.OptFn_  = optimizerFn;
            obj.FwdEng_ = OceanDL.Simulation.VectorizedEngine();
            obj.BwdEng_ = OceanDL.Simulation.BackpropEngine();

            % Create one optimizer per parameter tensor (W and b per layer)
            nL = graph.numLayers();
            obj.OptsW_ = cell(nL, 1);
            obj.OptsB_ = cell(nL, 1);
            for L = 2:nL
                obj.OptsW_{L} = optimizerFn();
                obj.OptsB_{L} = optimizerFn();
            end
        end

        function history = fit(obj, X, Y, opts)
            % FIT  Run the training loop and return loss history.
            %
            %   history = trainer.fit(X, Y)
            %   history = trainer.fit(X, Y, 'Epochs', 1000, 'BatchSize', 32, 'Verbose', true)
            %
            %   INPUTS:
            %     X — (N x d_in) feature matrix
            %     Y — (N x d_out) target matrix
            %
            %   OUTPUT:
            %     history.Loss   — (1 x Epochs) mean loss per epoch
            %     history.Epochs — number of epochs run
            arguments
                obj
                X       (:,:) double
                Y       (:,:) double
                opts.Epochs    (1,1) double {mustBePositive} = 500
                opts.BatchSize (1,1) double {mustBePositive} = 32
                opts.Verbose   (1,1) logical                 = false
            end

            N           = size(X, 1);
            nEpochs     = opts.Epochs;
            batchSize   = min(opts.BatchSize, N);
            lossHistory = zeros(1, nEpochs);

            for ep = 1:nEpochs
                % Shuffle data each epoch
                idx = randperm(N);
                X_s = X(idx, :);
                Y_s = Y(idx, :);

                epochLoss = 0;
                nBatches  = 0;

                for b = 1 : batchSize : N
                    bEnd = min(b + batchSize - 1, N);
                    Xb   = X_s(b:bEnd, :);
                    Yb   = Y_s(b:bEnd, :);

                    % Forward
                    [Yhat, cache] = obj.FwdEng_.forward(obj.Graph_, Xb);

                    % Loss value
                    epochLoss = epochLoss + obj.Loss_.forward(Yb, Yhat);
                    nBatches  = nBatches + 1;

                    % Backward
                    grads = obj.BwdEng_.backward(obj.Graph_, cache, Yhat, Yb, obj.Loss_);

                    % Parameter update
                    obj.applyGradients_(grads);
                end

                lossHistory(ep) = epochLoss / nBatches;

                if opts.Verbose && (mod(ep, max(1, floor(nEpochs/10))) == 0)
                    fprintf('  Epoch %4d / %d  loss = %.6f\n', ep, nEpochs, lossHistory(ep));
                end
            end

            history.Loss   = lossHistory;
            history.Epochs = nEpochs;
        end
    end

    methods (Access = private)
        function applyGradients_(obj, grads)
            % APPLYGRADIENTS_  Update all W and b in the graph using stored optimizers.
            nL = obj.Graph_.numLayers();
            for L = 2:nL
                W_old = obj.Graph_.getWeightMatrix(L-1);
                b_old = obj.Graph_.getBiasVector(L);

                W_new = obj.OptsW_{L}.step(W_old, grads{L}.dW);
                b_new = obj.OptsB_{L}.step(b_old, grads{L}.db);

                obj.Graph_.setWeightMatrix(L-1, W_new);
                obj.setBiasVector_(L, b_new);
            end
        end

        function setBiasVector_(obj, L, b)
            % SETBIASVECTOR_  Write b back into NNNode.Bias_ for layer L.
            layer = obj.Graph_.getLayer(L);
            nodes = layer.getNodes();
            for ni = 1:numel(nodes)
                nodes{ni}.Bias_ = b(ni);
            end
        end
    end
end
