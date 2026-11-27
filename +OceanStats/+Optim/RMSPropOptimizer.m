classdef RMSPropOptimizer < OceanStats.Optim.IOptimizer
    % RMSPROPOPTIMIZER  Per-parameter adaptive learning rate optimizer.
    %
    %   Update rule (Hinton, 2012 Coursera lecture):
    %     s = rho * s + (1 - rho) * grad.^2
    %     param = param - lr * grad ./ sqrt(s + eps)
    %
    %   Adapts the learning rate per-parameter by normalizing by the
    %   running RMS of recent gradients. Works well for non-stationary
    %   objectives (e.g. recurrent nets, RL). No bias correction.
    %
    %   Usage:
    %     opt = OceanStats.Optim.RMSPropOptimizer('LR', 0.001, 'Rho', 0.9);
    %     W   = opt.step(W, dW);

    properties
        LR      (1,1) double = 0.001   % Learning rate
        Rho     (1,1) double = 0.9     % Decay factor for squared gradient EMA
        Epsilon (1,1) double = 1e-8    % Numerical stability constant
    end

    properties (Access = private)
        S_   % Running mean of squared gradients (lazy-init)
    end

    methods
        function obj = RMSPropOptimizer(opts)
            arguments
                opts.LR      (1,1) double = 0.001
                opts.Rho     (1,1) double = 0.9
                opts.Epsilon (1,1) double = 1e-8
            end
            obj.LR      = opts.LR;
            obj.Rho     = opts.Rho;
            obj.Epsilon = opts.Epsilon;
            obj.S_      = [];
        end

        function param = step(obj, param, grad)
            if isempty(obj.S_)
                obj.S_ = zeros(size(grad), 'like', grad);   % state follows the gradient's precision and device
            end
            obj.S_ = obj.Rho * obj.S_ + (1 - obj.Rho) * grad.^2;
            param  = param - obj.LR * grad ./ sqrt(obj.S_ + obj.Epsilon);
        end

        function reset(obj)
            obj.S_ = [];
        end

        function s = describe(obj)
            s = sprintf('RMSProp(lr=%.4f, rho=%.3f, eps=%.0e)', ...
                obj.LR, obj.Rho, obj.Epsilon);
        end
    end
end
