function result = benchmarkDyna()
% BENCHMARKDYNA Measured Dyna throughput for reproducible performance review.
budgets = [0 5 50]; elapsed = zeros(size(budgets)); realSteps = elapsed; planningSteps = elapsed;
for k = 1:numel(budgets)
    env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 99);
    learner = OceanStats.RL.Dyna.Factory.dynaQ(env, 'PlanningSteps', budgets(k), 'Seed', 17);
    ticId = tic;
    output = learner.run('NumEpisodes', 5, 'MaxStepsPerEpisode', 1000);
    elapsed(k) = toc(ticId); realSteps(k) = output.RealSteps; planningSteps(k) = output.PlanningSteps;
end
result = table(budgets(:), elapsed(:), realSteps(:), planningSteps(:), ...
    'VariableNames', {'PlanningBudget','ElapsedSeconds','RealSteps','PlanningSteps'});
end
