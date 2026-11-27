classdef Device
    % DEVICE  Hardware-neutral precision / placement helpers (PyTorch
    %   tensor.to(dtype, device)). The library never branches on a GPU model:
    %   callers choose a DataType ("single" | "double") and a Device
    %   ("cpu" | "gpu"); everything downstream follows the arrays ('like').
    %
    %   X = OceanDL.Device.cast(X, 'DataType', "single", 'Device', "gpu");
    %   Y = OceanDL.Device.cast(Y, 'Device', "cpu");          % gather back
    %   OceanDL.Device.name(X)        -> "cpu" | "gpu"
    %   OceanDL.Device.dataType(X)    -> "single" | "double" | ...
    %   OceanDL.Device.gpuAvailable() -> true only if a usable GPU is present
    %
    %   Why "single" is usually the right choice for training: consumer GPUs
    %   run FP64 at 1/32 - 1/64 of the FP32 rate, and on CPU single halves
    %   memory traffic. Keep "double" for gradient checks (finite differences).

    methods (Static)
        function X = cast(X, opts)
            arguments
                X
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["", "single", "double"])} = ""
                opts.Device (1,1) string {mustBeMember(opts.Device, ["", "cpu", "gpu"])} = ""
            end
            if isa(X, 'OceanDL.Autograd.Tensor')
                error('OceanDL:Device:Tensor', 'Cast the array (Tensor.Data) or use Module.to().');
            end
            if ~isnumeric(X) && ~islogical(X), return; end
            if opts.Device == "gpu" && ~isa(X, 'gpuArray')
                OceanDL.Device.requireGpu();
                X = gpuArray(X);
            elseif opts.Device == "cpu" && isa(X, 'gpuArray')
                X = gather(X);
            end
            if opts.DataType ~= "" && isfloat(X)                  % never cast integer / logical data (labels, masks)
                X = cast(X, opts.DataType);
            end
        end

        function d = name(X)
            if isa(X, 'OceanDL.Autograd.Tensor'), X = X.Data; end
            if isa(X, 'gpuArray'), d = "gpu"; else, d = "cpu"; end
        end

        function t = dataType(X)
            if isa(X, 'OceanDL.Autograd.Tensor'), X = X.Data; end
            t = string(underlyingType(X));
        end

        function tf = gpuAvailable()
            tf = false;
            try
                tf = canUseGPU();
            catch
            end
        end
    end

    methods (Static, Access = private)
        function requireGpu()
            if OceanDL.Device.gpuAvailable(), return; end
            reason = 'Parallel Computing Toolbox or a supported GPU is not available.';
            try
                gpuDevice();
            catch e
                reason = e.message;
            end
            error('OceanDL:Device:NoGPU', 'Device "gpu" requested but no usable GPU: %s', reason);
        end
    end
end
