function r = fit(model, lossFn, steps, opts)
% FIT  One training loop for every model in the library (AdamW + warmup-
%   cosine learning rate + gradient-norm clipping + optional weight EMA).
%
%   r = OceanDL.Training.fit(model, @(s) model.forward(batch(s)), 3000, 'LR', 1e-3)
%     lossFn(s)   returns the loss Tensor of step s (the caller draws the batch)
%     'OnStep'    @(s) ... called before each step (e.g. KL warm-up: model.Beta = ...)
%     'Record'    @() row  -- numbers logged after each step (e.g. LastStats fields)
%     'EMA'       an OceanDL.optim.ModelEMA, updated after each step
%   r.Curve (loss per step), r.Records (steps x numel(row)), r.Seconds,
%   r.StepSeconds (median wall-clock per step).

arguments
    model OceanDL.nn.Module
    lossFn function_handle
    steps (1,1) double {mustBePositive, mustBeInteger}
    opts.LR (1,1) double = 1e-3
    opts.WeightDecay (1,1) double = 0
    opts.WarmupFraction (1,1) double = 0.02
    opts.Clip (1,1) double = 1
    opts.OnStep = []
    opts.Record = []
    opts.EMA = []
    opts.Verbose (1,1) double = 0          % print every k steps (0 = silent)
end
opt = OceanDL.optim.ModuleOptimizer.forModule(model, ...
    @(wd) OceanStats.Optim.AdamWOptimizer('LR', opts.LR, 'WeightDecay', wd), 'WeightDecay', opts.WeightDecay);
sched = OceanDL.optim.WarmupCosine(opts.LR, 'WarmupSteps', max(1, round(opts.WarmupFraction * steps)), ...
    'TotalSteps', steps, 'MinLR', opts.LR / 20);
params = model.parameters();
r.Curve = zeros(1, steps); r.Records = [];
t = zeros(1, steps);
model.train();
for s = 1:steps
    t0 = tic;
    if ~isempty(opts.OnStep), opts.OnStep(s); end
    opt.setLR(sched.at(s));
    opt.zeroGrad();
    L = lossFn(s);
    L.backward();
    if opts.Clip > 0, OceanDL.nn.clipGradNorm(params, opts.Clip); end
    opt.step();
    if ~isempty(opts.EMA), opts.EMA.update(); end
    r.Curve(s) = double(gather(L.Data));
    if ~isempty(opts.Record)
        row = opts.Record();
        if s == 1, r.Records = zeros(steps, numel(row)); end
        r.Records(s, :) = row;
    end
    t(s) = toc(t0);
    if opts.Verbose > 0 && mod(s, opts.Verbose) == 0
        fprintf('    step %5d / %d   loss %.4f   (%.2f s/step)\n', s, steps, mean(r.Curve(max(1, s - opts.Verbose + 1):s)), median(t(1:s)));
    end
end
r.Seconds = sum(t);
r.StepSeconds = median(t);
end
