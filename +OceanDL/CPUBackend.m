classdef CPUBackend < OceanDL.ComputeBackend
    % CPUBACKEND  Identity compute backend — all ops stay on the CPU.
    %
    %   This is the default backend for VectorizedEngine and BackpropEngine.
    %   send() and recv() are pure identity — zero overhead, zero copies.
    %   Results are bit-identical to direct matrix operations.
    %
    %   The primary purpose of CPUBackend is to allow existing code to use
    %   the Backend interface without any performance penalty, and to serve
    %   as a correctness reference when comparing with GPUBackend results.
    %
    %   Usage:
    %     backend = OceanDL.CPUBackend();
    %     [Y, cache] = engine.forward(graph, X, 'train', backend);
    %
    %   See also: OceanDL.GPUBackend
    %             OceanDL.ComputeBackend

    methods
        function X_dev = send(~, X)
            X_dev = X;          % identity — already on CPU
        end

        function X_cpu = recv(~, X_dev)
            X_cpu = X_dev;      % identity — already CPU double
        end

        function tf = isGPU(~)
            tf = false;
        end

        function s = name(~)
            s = 'cpu';
        end
    end
end
