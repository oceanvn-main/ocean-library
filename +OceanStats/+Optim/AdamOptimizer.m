classdef AdamOptimizer < OceanStats.Optim.IOptimizer
    % ADAMOPTIMIZER  Per-parameter Adam optimizer (Kingma & Ba, 2015).
    %
    %   Each instance manages the moment estimates (m, v) for ONE trainable
    %   parameter tensor. Create one optimizer per parameter.
    %
    %   Algorithm (from the original paper):
    %     m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (1st moment)
    %     v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (2nd moment)
    %     m_hat = m_t / (1 - beta1^t)                         (bias-corrected)
    %     v_hat = v_t / (1 - beta2^t)                         (bias-corrected)
    %     theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    %
    %   Reference:
    %     Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
    %     optimization. ICLR 2015. https://arxiv.org/abs/1412.6980
    %
    %   USAGE — one optimizer per parameter:
    %     optW1 = OceanStats.Optim.AdamOptimizer('LR', 0.001);
    %     optb1 = OceanStats.Optim.AdamOptimizer('LR', 0.001);
    %     % In training loop:
    %     W1 = optW1.step(W1, dW1);
    %     b1 = optb1.step(b1, db1);
    %
    %   USAGE — shared hyperparams using a factory:
    %     opt = @() OceanStats.Optim.AdamOptimizer('LR', lr, 'Beta1', 0.9);
    %     [oW1, ob1, oW2, ob2] = deal(opt(), opt(), opt(), opt());
    %
    % See also: OceanStats.Optim

    properties
        LR      (1,1) double = 0.001   % Learning rate
        Beta1   (1,1) double = 0.9     % 1st moment decay (default: 0.9)
        Beta2   (1,1) double = 0.999   % 2nd moment decay (default: 0.999)
        Epsilon (1,1) double = 1e-8    % Numerical stability constant
    end

    properties (SetAccess = private)
        StepCount (1,1) double = 0     % Current iteration t (auto-incremented)
    end

    properties (Access = private)
        M_   % 1st moment estimate (same shape as parameter)
        V_   % 2nd moment estimate (same shape as parameter)
    end

    methods
        function obj = AdamOptimizer(opts)
            % Constructor. All arguments are optional name-value pairs.
            %
            %   opt = OceanStats.Optim.AdamOptimizer()
            %   opt = OceanStats.Optim.AdamOptimizer('LR', 0.001, 'Beta1', 0.9)
            arguments
                opts.LR      (1,1) double = 0.001
                opts.Beta1   (1,1) double = 0.9
                opts.Beta2   (1,1) double = 0.999
                opts.Epsilon (1,1) double = 1e-8
            end
            obj.LR      = opts.LR;
            obj.Beta1   = opts.Beta1;
            obj.Beta2   = opts.Beta2;
            obj.Epsilon = opts.Epsilon;
        end

        function param = step(obj, param, grad)
            % STEP  Apply one Adam update step, returns updated parameter.
            %
            %   param_new = opt.step(param, grad)
            %
            %   param - current parameter value (any numeric array shape)
            %   grad  - gradient w.r.t. param (same shape as param)
            %
            %   The step count is auto-incremented internally.
            %   On the first call, moment buffers are zero-initialised.
            obj.StepCount = obj.StepCount + 1;

            if obj.StepCount == 1
                % Lazy init — allocates moment buffers on first use
                obj.M_ = zeros(size(grad), 'like', grad);   % state follows the gradient's precision and device
                obj.V_ = zeros(size(grad), 'like', grad);   % state follows the gradient's precision and device
            end

            % Moment updates
            obj.M_ = obj.Beta1 * obj.M_ + (1 - obj.Beta1) * grad;
            obj.V_ = obj.Beta2 * obj.V_ + (1 - obj.Beta2) * grad.^2;

            % Bias correction
            t     = obj.StepCount;
            m_hat = obj.M_ / (1 - obj.Beta1^t);
            v_hat = obj.V_ / (1 - obj.Beta2^t);

            % Parameter update
            param = param - obj.LR * m_hat ./ (sqrt(v_hat) + obj.Epsilon);
        end

        function reset(obj)
            % RESET  Clear moment buffers and step count (re-use for new run).
            obj.M_        = [];
            obj.V_        = [];
            obj.StepCount = 0;
        end

        function s = describe(obj)
            % DESCRIBE  Human-readable summary of hyperparameters.
            s = sprintf('Adam(lr=%.4f, b1=%.3f, b2=%.4f, eps=%.0e)', ...
                obj.LR, obj.Beta1, obj.Beta2, obj.Epsilon);
        end
    end

    methods (Static)
        function optimizers = forParams(paramCount, opts)
            % FORPARAMS  Create an array of AdamOptimizers with shared hyperparams.
            %
            %   opts = OceanStats.Optim.AdamOptimizer.forParams(6, 'LR', 0.001);
            %   % Returns a 1x6 cell array of AdamOptimizer instances.
            %   % Use: opts{1}.step(W1, dW1), opts{2}.step(b1, db1), ...
            arguments
                paramCount  (1,1) double
                opts.LR      (1,1) double = 0.001
                opts.Beta1   (1,1) double = 0.9
                opts.Beta2   (1,1) double = 0.999
                opts.Epsilon (1,1) double = 1e-8
            end
            optimizers = cell(1, paramCount);
            for i = 1:paramCount
                optimizers{i} = OceanStats.Optim.AdamOptimizer( ...
                    'LR', opts.LR, 'Beta1', opts.Beta1, ...
                    'Beta2', opts.Beta2, 'Epsilon', opts.Epsilon);
            end
        end
    end
end
