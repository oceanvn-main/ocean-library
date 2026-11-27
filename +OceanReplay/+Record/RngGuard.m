classdef RngGuard
    % RNGGUARD  Run a function without disturbing any random-number state the
    %   computation uses: the global CPU generator (rng) and, when a GPU is
    %   available, the GPU generator. Views run inside it, so recording them
    %   cannot change the recorded run (invariance, Master plan Sec. 5.3).
    %
    %   On a machine with a GPU, the first call touches the GPU generator; MATLAB may then print
    %   its own one-time notice about the GPU model (e.g. compute capability 6.1). That notice is
    %   raised inside MATLAB and cannot be silenced with warning('off', ...); it is harmless.
    %
    %   Example:
    %     v = OceanReplay.Record.RngGuard.run(@() randn(3));   % rng state unchanged afterwards

    methods (Static)
        function varargout = run(fn)
            arguments
                fn (1,1) function_handle
            end
            cpu = rng;
            gpu = [];
            if OceanReplay.Record.RngGuard.hasGPU()
                gpu = parallel.gpu.rng;
            end
            restore = onCleanup(@() OceanReplay.Record.RngGuard.restore(cpu, gpu));
            [varargout{1:nargout}] = fn();
        end

        function tf = hasGPU()
            % Checked once per session (gpuDeviceCount is slow on its first call).
            persistent cached
            if isempty(cached)
                cached = false;
                try
                    cached = exist('gpuDeviceCount', 'file') > 0 && gpuDeviceCount("available") > 0;
                catch
                end
            end
            tf = cached;
        end
    end

    methods (Static, Access = private)
        function restore(cpu, gpu)
            rng(cpu);
            if ~isempty(gpu), parallel.gpu.rng(gpu); end
        end
    end
end
