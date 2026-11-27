classdef VectorizedEngine
    % VECTORIZEDENGINE  Batch matrix forward pass for NNGraph.
    %
    %   Replaces the per-sample loop in ForwardPassEngine with a single
    %   vectorized matrix multiplication per layer. Much faster for
    %   large N. Returns a ForwardCache used by BackpropEngine (NN-2).
    %
    %   CONVENTION:
    %     X    — (N x d_in)  input (N samples, row = one sample)
    %     W    — (d_out x d_in)  weight matrix for layer L → L+1
    %     b    — (d_out x 1)  bias vector for layer L+1
    %     Z    — (d_out x N)  pre-activation  (internal column-major)
    %     A    — (d_out x N)  post-activation (internal column-major)
    %     cache — (nL x 1) cell, cache{L} = struct('Z','Z_raw','A','A_skip')
    %
    %   BACKEND (NN-6):
    %     An optional ComputeBackend controls device routing.
    %     CPUBackend (default): identity — zero overhead.
    %     GPUBackend: W, b, A moved via gpuArray; Y gathered back to CPU.
    %     Existing callers that omit the Backend arg are unaffected.
    %
    %   USAGE:
    %     engine = OceanDL.Simulation.VectorizedEngine();
    %     [Y, cache] = engine.forward(graph, X);
    %     [Y, cache] = engine.forward(graph, X, 'train', OceanDL.GPUBackend());
    %
    %   See also: OceanDL.Graph.NNGraph
    %             OceanDL.ComputeBackend

    methods
        function [Y, cache] = forward(obj, graph, X, mode, backend) %#ok<INUSL>
            % FORWARD  Vectorized forward pass through all layers.
            %
            %   [Y, cache] = engine.forward(graph, X)
            %   [Y, cache] = engine.forward(graph, X, 'infer')
            %   [Y, cache] = engine.forward(graph, X, 'train', GPUBackend())
            %
            %   INPUTS:
            %     graph   — NNGraph instance
            %     X       — (N x d_in) input matrix
            %     mode    — 'train' (default) or 'infer'
            %     backend — ComputeBackend (default: CPUBackend)

            if nargin < 4 || isempty(mode),    mode    = 'train'; end
            if nargin < 5 || isempty(backend), backend = OceanDL.CPUBackend(); end

            import OceanDL.Activation.ActivationFactory

            nL   = graph.numLayers();
            spec = graph.toSpec();
            N    = size(X, 1);

            % Send input to device
            A_col = backend.send(X');       % (d_in x N) — on device

            cache    = cell(nL, 1);
            cache{1} = struct('Z', [], 'Z_raw', [], 'A', A_col, 'A_skip', []);

            for L = 2:nL
                % Send weights + bias to device
                W = backend.send(graph.getWeightMatrix(L - 1));   % (d_L x d_{L-1})
                b = backend.send(graph.getBiasVector(L));          % (d_L x 1)

                Z_raw = W * A_col + repmat(b, 1, N);               % (d_L x N)

                % Normalization (hidden layers only; norm runs on whatever device Z_raw is on)
                normLayer = spec.NormLayers{L};
                if ~isempty(normLayer) && L < nL
                    Z = normLayer.normalize(Z_raw, mode);
                else
                    Z = Z_raw;
                end

                act   = ActivationFactory.create(spec.Activations{L});
                A_col = act.forward(Z);                             % (d_L x N)

                % Residual skip connection (hidden layers only)
                A_skip = [];
                if spec.ResidualStart(L) && L < nL
                    A_skip = cache{L-2}.A;
                    W_skip = spec.SkipProjections{L};
                    if ~isempty(W_skip)
                        W_skip = backend.send(W_skip);
                    end
                    A_col  = OceanDL.ResidualBlock.applySkip(A_col, A_skip, W_skip);
                end

                cache{L} = struct('Z', Z, 'Z_raw', Z_raw, 'A', A_col, 'A_skip', A_skip);
            end

            Y = backend.recv(A_col)';       % gather from device → (N x d_out) CPU double
        end
    end
end
