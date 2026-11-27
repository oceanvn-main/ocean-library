classdef SGDMomentumOptimizer < OceanStats.Optim.IOptimizer
    % SGDMOMENTUMOPTIMIZER  SGD with exponential moving average of gradients.
    %
    %   Update rule (heavy-ball / classical momentum):
    %     v = beta * v + (1 - beta) * grad
    %     param = param - lr * v
    %
    %   The EMA velocity v accumulates gradient history, allowing the
    %   optimizer to accelerate in consistent gradient directions and
    %   dampen oscillations in noisy directions.
    %
    %   Usage:
    %     opt = OceanStats.Optim.SGDMomentumOptimizer('LR', 0.01, 'Beta', 0.9);
    %     W   = opt.step(W, dW);

    properties
        LR   (1,1) double = 0.01   % Learning rate
        Beta (1,1) double = 0.9    % Momentum coefficient (0 = no momentum, → SGD)
    end

    properties (Access = private)
        V_   % Velocity buffer (lazy-init on first step)
    end

    methods
        function obj = SGDMomentumOptimizer(opts)
            arguments
                opts.LR   (1,1) double = 0.01
                opts.Beta (1,1) double = 0.9
            end
            obj.LR   = opts.LR;
            obj.Beta = opts.Beta;
            obj.V_   = [];
        end

        function param = step(obj, param, grad)
            if isempty(obj.V_)
                obj.V_ = zeros(size(grad), 'like', grad);   % state follows the gradient's precision and device
            end
            obj.V_ = obj.Beta * obj.V_ + (1 - obj.Beta) * grad;
            param  = param - obj.LR * obj.V_;
        end

        function reset(obj)
            obj.V_ = [];
        end

        function s = describe(obj)
            s = sprintf('SGDMomentum(lr=%.4f, beta=%.3f)', obj.LR, obj.Beta);
        end
    end
end
