classdef WarmupCosine
    % WARMUPCOSINE  Learning-rate schedule used to train Transformers
    %   (ViT, DeiT, Swin): linear warmup, then cosine decay.
    %
    %     step <= W :  lr = BaseLR * step / W
    %     step >  W :  lr = MinLR + (BaseLR - MinLR) * (1 + cos(pi * (step - W) / (S - W))) / 2
    %   S = TotalSteps, W = WarmupSteps.
    %   Warmup lets Adam's second-moment estimates settle before large steps
    %   (critical for Post-LN stacks); cosine decay anneals to MinLR.
    %
    %   sched = OceanDL.optim.WarmupCosine(1e-3, 'WarmupSteps', 200, 'TotalSteps', 5000);
    %   opt.setLR(sched.at(step));

    properties (SetAccess = private)
        BaseLR (1,1) double
        MinLR (1,1) double
        WarmupSteps (1,1) double
        TotalSteps (1,1) double
    end

    methods
        function obj = WarmupCosine(baseLR, opts)
            arguments
                baseLR (1,1) double {mustBePositive}
                opts.WarmupSteps (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.TotalSteps (1,1) double {mustBePositive, mustBeInteger} = 1000
                opts.MinLR (1,1) double {mustBeNonnegative} = 0
            end
            obj.BaseLR = baseLR; obj.MinLR = opts.MinLR;
            obj.WarmupSteps = opts.WarmupSteps; obj.TotalSteps = opts.TotalSteps;
        end

        function lr = at(obj, step)
            W = obj.WarmupSteps; S = obj.TotalSteps;
            if step <= W && W > 0
                lr = obj.BaseLR * step / W;
            else
                progress = min(1, (step - W) / max(1, S - W));
                lr = obj.MinLR + (obj.BaseLR - obj.MinLR) * (1 + cos(pi * progress)) / 2;
            end
        end
    end
end
