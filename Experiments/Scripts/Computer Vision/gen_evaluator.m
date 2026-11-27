function ev = gen_evaluator(D, steps, ckptDir)
% GEN_EVALUATOR  The sample-quality classifier for dataset D, trained once on
%   the REAL training images and cached (state dict) via Checkpoint.
K = numel(D.ClassNames);
key = sprintf('evaluator_%s_%d_s%d', D.Name, numel(D.ytr), steps);
out = OceanDL.Training.Checkpoint.run(ckptDir, key, @() train());
net = OceanDL.Generative.SampleEvaluator.classifier(1, D.Size, K, 0);
net.loadStateDict(out.State);
ev = OceanDL.Generative.SampleEvaluator(D.Xtr, D.ytr, 'NumClasses', K, 'Net', net);
ev.testAccuracy(double(D.Xte), D.yte);
fprintf('  evaluator classifier: test accuracy %.2f%% (%s)\n', 100 * ev.Accuracy, D.Name);

    function out = train()
        e = OceanDL.Generative.SampleEvaluator(double(D.Xtr), D.ytr, 'NumClasses', K, 'Steps', steps, 'Seed', 0);
        out = struct('State', e.Net.stateDict());
    end
end
