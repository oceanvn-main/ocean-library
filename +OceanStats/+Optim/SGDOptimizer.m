classdef SGDOptimizer < OceanStats.Optim.IOptimizer
    % SGDOPTIMIZER  Vanilla stochastic gradient descent.
    %
    %   Update rule:
    %     param = param - lr * grad
    %
    %   Stateless — no moment buffers. One instance shared across steps.
    %
    %   Usage:
    %     opt = OceanStats.Optim.SGDOptimizer('LR', 0.01);
    %     W   = opt.step(W, dW);

    properties
        LR (1,1) double = 0.01   % Learning rate
    end

    methods
        function obj = SGDOptimizer(opts)
            arguments
                opts.LR (1,1) double = 0.01
            end
            obj.LR = opts.LR;
        end

        function param = step(obj, param, grad)
            param = param - obj.LR * grad;
        end

        function reset(~)
            % No state to clear — SGD is stateless.
        end

        function s = describe(obj)
            s = sprintf('SGD(lr=%.4f)', obj.LR);
        end
    end
end
