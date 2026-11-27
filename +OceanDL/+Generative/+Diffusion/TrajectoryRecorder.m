classdef TrajectoryRecorder < handle
    % TRAJECTORYRECORDER  Watch the reverse (denoising) process from inside,
    %   without changing any sampler: it wraps the predictor that every
    %   ISampler calls, and stores what goes in and what comes out.
    %
    %   rec = TrajectoryRecorder(predict);                  % any predictor handle
    %   X   = sampler.sample(rec.handle(), schedule, shape, ...);
    %   Per call i (one network evaluation):
    %     rec.T(i)       time t given to the model
    %     rec.X{i}       INPUT: the current noisy sample x_t
    %     rec.X0{i}      OUTPUT: the model's guess of the clean image E[x0 | x_t]
    %     rec.Eps{i}     OUTPUT: the model's guess of the noise    E[eps | x_t]
    %     rec.Extra{i}   whatever 'OnCall' returns (e.g. denoiser diagnostics)

    properties (SetAccess = private)
        T = []
        X = {}
        X0 = {}
        Eps = {}
        Extra = {}
    end

    properties (Access = private)
        Predict
        OnCall
    end

    methods
        function obj = TrajectoryRecorder(predict, opts)
            arguments
                predict function_handle
                opts.OnCall = []            % @() value stored after each call
            end
            obj.Predict = predict;
            obj.OnCall = opts.OnCall;
        end

        function [x0, eps] = predict(obj, x, t)
            [x0, eps] = obj.Predict(x, t);
            obj.T(end+1) = t;
            obj.X{end+1} = x; obj.X0{end+1} = x0; obj.Eps{end+1} = eps;
            if ~isempty(obj.OnCall), obj.Extra{end+1} = obj.OnCall(); end
        end

        function fn = handle(obj)
            fn = @(x, t) obj.predict(x, t);
        end

        function n = numCalls(obj)
            n = numel(obj.T);
        end
    end
end
