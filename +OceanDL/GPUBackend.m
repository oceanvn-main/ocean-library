classdef GPUBackend < OceanDL.ComputeBackend
    % GPUBACKEND  GPU compute backend using MATLAB's Parallel Computing Toolbox.
    %
    %   Routes all weight matrices and activation tensors to the GPU via
    %   gpuArray(). Results are gathered back to CPU doubles via gather().
    %
    %   SAFE FALLBACK:
    %     If the Parallel Computing Toolbox is absent, or no compatible GPU
    %     is found, GPUBackend degrades to CPU-mode SILENTLY ON CONSTRUCTION
    %     (emitting a one-time warning). All downstream ops continue on CPU.
    %     This means code written for GPU runs correctly on CPU-only machines.
    %
    %   PERFORMANCE NOTE:
    %     GPU acceleration is only beneficial for large batch sizes (N >> 512)
    %     or very wide networks (d >> 256). For small nets, PCIe transfer
    %     overhead dominates. See Tests/test_nn6.m for benchmark results.
    %
    %   USAGE:
    %     backend = OceanDL.GPUBackend();
    %     [Y, cache] = engine.forward(graph, X, 'train', backend);
    %     % Y is a CPU double regardless — recv() handles the gather().
    %
    %   See also: OceanDL.CPUBackend
    %             OceanDL.ComputeBackend

    properties (SetAccess = private)
        UseGPU_ (1,1) logical = false   % false if fallback was triggered
    end

    methods
        function obj = GPUBackend()
            % GPUBACKEND  Construct and probe GPU availability.
            %   Emits a warning and falls back to CPU if toolbox is missing.
            try
                if ~exist('gpuArray', 'file') && ~exist('gpuArray', 'builtin')
                    error('no_toolbox');
                end
                info = gpuDevice();   % throws if no compatible GPU
                obj.UseGPU_ = true;
                fprintf('[GPUBackend] Using GPU: %s  (%.0f MB free)\n', ...
                    info.Name, info.AvailableMemory / 1e6);
            catch
                warning('OceanDL:GPUBackend:Fallback', ...
                    'No compatible GPU or Parallel Computing Toolbox found. ' ...
                    'Falling back to CPU. Results will be identical to CPUBackend.');
                obj.UseGPU_ = false;
            end
        end

        function X_dev = send(obj, X)
            if obj.UseGPU_
                X_dev = gpuArray(X);
            else
                X_dev = X;
            end
        end

        function X_cpu = recv(obj, X_dev)
            if obj.UseGPU_
                X_cpu = gather(X_dev);
            else
                X_cpu = X_dev;
            end
        end

        function tf = isGPU(obj)
            tf = obj.UseGPU_;
        end

        function s = name(obj)
            if obj.UseGPU_
                s = 'gpu';
            else
                s = 'gpu(fallback:cpu)';
            end
        end
    end
end
