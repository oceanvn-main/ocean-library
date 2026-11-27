classdef AdamWOptimizer < OceanStats.Optim.IOptimizer
    % ADAMWOPTIMIZER  Adam with DECOUPLED weight decay (Loshchilov & Hutter 2019).
    %
    %   Adam moments as in AdamOptimizer, then
    %     theta_t = theta_{t-1} - lr * ( m_hat / (sqrt(v_hat) + eps)  +  lambda * theta_{t-1} )
    %
    %   Why "decoupled": adding lambda*theta to the GRADIENT (classic L2) would
    %   pass the decay through Adam's per-parameter scaling 1/sqrt(v_hat), so
    %   parameters with large gradients would be decayed LESS. AdamW applies
    %   the decay directly to the weights, uniformly -- the standard optimiser
    %   for Transformers (BERT, ViT, Swin).
    %
    %   One instance per parameter. Parameters that should not be decayed
    %   (biases, LayerNorm gains, position embeddings, [CLS]) get WeightDecay 0 --
    %   see OceanDL.optim.ModuleOptimizer.forModule.
    %
    %   opt = OceanStats.Optim.AdamWOptimizer('LR', 1e-3, 'WeightDecay', 0.05);
    %   W   = opt.step(W, dW);

    properties
        LR          (1,1) double = 0.001
        Beta1       (1,1) double = 0.9
        Beta2       (1,1) double = 0.999
        Epsilon     (1,1) double = 1e-8
        WeightDecay (1,1) double = 0.01
    end

    properties (SetAccess = private)
        StepCount (1,1) double = 0
    end

    properties (Access = private)
        M_
        V_
    end

    methods
        function obj = AdamWOptimizer(opts)
            arguments
                opts.LR          (1,1) double = 0.001
                opts.Beta1       (1,1) double = 0.9
                opts.Beta2       (1,1) double = 0.999
                opts.Epsilon     (1,1) double = 1e-8
                opts.WeightDecay (1,1) double {mustBeNonnegative} = 0.01
            end
            obj.LR = opts.LR; obj.Beta1 = opts.Beta1; obj.Beta2 = opts.Beta2;
            obj.Epsilon = opts.Epsilon; obj.WeightDecay = opts.WeightDecay;
        end

        function param = step(obj, param, grad)
            obj.StepCount = obj.StepCount + 1;
            if obj.StepCount == 1
                obj.M_ = zeros(size(grad), 'like', grad);   % state follows the gradient's precision and device
                obj.V_ = zeros(size(grad), 'like', grad);   % state follows the gradient's precision and device
            end
            obj.M_ = obj.Beta1 * obj.M_ + (1 - obj.Beta1) * grad;
            obj.V_ = obj.Beta2 * obj.V_ + (1 - obj.Beta2) * grad.^2;
            t = obj.StepCount;
            mHat = obj.M_ / (1 - obj.Beta1^t);
            vHat = obj.V_ / (1 - obj.Beta2^t);
            param = param - obj.LR * (mHat ./ (sqrt(vHat) + obj.Epsilon) + obj.WeightDecay * param);
        end

        function reset(obj)
            obj.M_ = []; obj.V_ = []; obj.StepCount = 0;
        end

        function s = describe(obj)
            s = sprintf('AdamW(lr=%.4g, b1=%.3f, b2=%.4f, eps=%.0e, wd=%.3g)', ...
                obj.LR, obj.Beta1, obj.Beta2, obj.Epsilon, obj.WeightDecay);
        end
    end
end
