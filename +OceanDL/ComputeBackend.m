classdef (Abstract) ComputeBackend < handle
    % COMPUTEBACKEND  Abstract interface for compute device routing.
    %
    %   Provides a thin wrapper over data movement so that VectorizedEngine
    %   and BackpropEngine can run on CPU or GPU without any internal changes.
    %   All matrix operations (W*A, W+b, etc.) are unchanged — only the DATA
    %   lives on the device selected by the backend.
    %
    %   CONTRACT:
    %     X   = backend.send(X)        move X to the compute device
    %     out = backend.recv(out)      move result back to CPU double
    %     tf  = backend.isGPU()        true iff running on GPU
    %     s   = backend.name()         human-readable string
    %
    %   DEFAULT BACKEND:
    %     Both engines default to CPUBackend when no Backend arg is provided.
    %     This means existing code is 100% backward-compatible.
    %
    %   USAGE:
    %     % CPU (default — identical to not passing Backend at all):
    %     backend = OceanDL.CPUBackend();
    %     [Y, cache] = engine.forward(graph, X, 'train', backend);
    %
    %     % GPU (with automatic safe fallback):
    %     backend = OceanDL.GPUBackend();
    %     [Y, cache] = engine.forward(graph, X, 'train', backend);
    %
    %   See also: OceanDL.CPUBackend
    %             OceanDL.GPUBackend
    %             OceanDL.Simulation.VectorizedEngine

    methods (Abstract)
        X_dev = send(obj, X)
        % SEND  Move array X to the compute device.
        %   CPU: identity. GPU: gpuArray(X).

        X_cpu = recv(obj, X_dev)
        % RECV  Move result back to CPU (double).
        %   CPU: identity. GPU: gather(X_dev).

        tf = isGPU(obj)
        % ISGPU  Return true iff this backend routes to GPU.

        s = name(obj)
        % NAME  Return string: 'cpu' or 'gpu'.
    end
end
