classdef BackpropEngine
    % BACKPROPENGINE  Chain-rule backpropagation using ForwardCache from VectorizedEngine.
    %
    %   CONVENTION (matches VectorizedEngine):
    %     Z, A — (d x N) column-major
    %     W    — (d_out x d_in), b — (d_out x 1)
    %     dW   — (d_out x d_in), db — (d_out x 1)
    %
    %   CHAIN RULE (layer L):
    %     delta_L  = dLoss/dZ_L = (dLoss/dA_L) .* act'(Z_L)
    %     dW_L     = delta_L * A_{L-1}' / N
    %     db_L     = mean(delta_L, 2)
    %     dA_{L-1} = W_L' * delta_L
    %
    %   RESIDUAL SKIP (when spec.ResidualStart(L)):
    %     delta flows through BOTH the main W' path AND the skip path.
    %
    %   SOFTMAX + CE FUSION:
    %     When output activation is 'softmax', loss.gradient() returns (p-y) directly.
    %
    %   BACKEND (NN-6):
    %     An optional ComputeBackend routes weight matrices to the device.
    %     CPUBackend (default): identity — zero overhead.
    %     GPUBackend: W moved via gpuArray; dW gathered back to CPU before storage.
    %
    %   See also: OceanDL.Simulation.VectorizedEngine
    %             OceanDL.ComputeBackend

    methods
        function grads = backward(obj, graph, cache, Y, Y_true, loss, backend) %#ok<INUSL>
            % BACKWARD  Compute dW and db for every layer.
            %
            %   grads = back.backward(graph, cache, Y, Y_true, loss)
            %   grads = back.backward(graph, cache, Y, Y_true, loss, GPUBackend())

            if nargin < 7 || isempty(backend)
                backend = OceanDL.CPUBackend();
            end

            import OceanDL.Activation.ActivationFactory

            nL   = graph.numLayers();
            spec = graph.toSpec();
            N    = size(Y, 1);

            grads = cell(nL, 1);
            for L = 1:nL
                grads{L} = struct('dW', [], 'db', []);
            end

            % Accumulated upstream gradient per layer (for residual skip merging)
            dA_acc = cell(nL, 1);

            % ---- Output layer delta ---------------------------------- %
            dA      = loss.gradient(Y_true, Y)';   % (d_out x N)
            actName = spec.Activations{nL};
            if strcmp(actName, 'softmax')
                delta = dA;
            else
                act   = ActivationFactory.create(actName);
                delta = dA .* act.grad(cache{nL}.Z);
            end
            dA_acc{nL} = delta;

            % ---- Backprop through layers L = nL down to 2 ----------- %
            for L = nL : -1 : 2
                if ~isempty(dA_acc{L})
                    delta = dA_acc{L};
                end

                A_prev = cache{L-1}.A;                             % (d_{L-1} x N) on device or CPU
                W_L    = backend.send(graph.getWeightMatrix(L-1)); % (d_L x d_{L-1}) on device

                % Compute dW, db — then immediately gather to CPU for optimizer
                dW_dev = (delta * A_prev') / N;
                grads{L}.dW = backend.recv(dW_dev);
                grads{L}.db = backend.recv(mean(delta, 2));

                % Main path: propagate delta to previous layer
                dA_prev = W_L' * delta;
                if L > 2
                    actPrev    = ActivationFactory.create(spec.Activations{L-1});
                    delta_prev = dA_prev .* actPrev.grad(cache{L-1}.Z);
                    if isempty(dA_acc{L-1})
                        dA_acc{L-1} = delta_prev;
                    else
                        dA_acc{L-1} = dA_acc{L-1} + delta_prev;
                    end
                end

                % Residual skip path: delta flows to layer L-2
                if spec.ResidualStart(L) && L >= 3
                    W_skip = spec.SkipProjections{L};
                    if isempty(W_skip)
                        dA_skip = delta;
                    else
                        W_skip_dev = backend.send(W_skip);
                        dA_skip    = W_skip_dev' * delta;
                    end
                    if L-2 > 1
                        actSkip = ActivationFactory.create(spec.Activations{L-2});
                        dA_skip = dA_skip .* actSkip.grad(cache{L-2}.Z);
                    end
                    if isempty(dA_acc{L-2})
                        dA_acc{L-2} = dA_skip;
                    else
                        dA_acc{L-2} = dA_acc{L-2} + dA_skip;
                    end
                end
            end
        end
    end
end
