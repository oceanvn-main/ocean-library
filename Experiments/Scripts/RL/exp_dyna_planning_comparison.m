%% Dyna-Q planning-budget comparison (Sutton & Barto, Chapter 8)
% Compares learning per real environment interaction and reports computation
% separately, so larger planning budgets are not presented as free speedups.
budgets = [0 5 20]; seeds = 1:10; numEpisodes = 40;
returns = zeros(numEpisodes, numel(budgets), numel(seeds));
backups = zeros(numel(budgets), numel(seeds));
for b = 1:numel(budgets)
    for k = 1:numel(seeds)
        env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19);
        learner = OceanStats.RL.Dyna.Factory.dynaQ(env, 'Gamma', 1, ...
            'PlanningSteps', budgets(b), 'Seed', seeds(k));
        result = learner.run('NumEpisodes', numEpisodes, 'MaxStepsPerEpisode', 2000);
        returns(:, b, k) = result.Returns;
        backups(b, k) = sum(result.Counts(:));
    end
end
meanReturn = mean(returns, 3);
viz = OceanViz.Core('Theme', 'Dark');
figure('Name', 'Dyna planning comparison');
ax = axes(); viz.styleAxes(ax); hold(ax, 'on');
colors = lines(numel(budgets));
for b = 1:numel(budgets)
    plot(ax, 1:numEpisodes, meanReturn(:, b), 'LineWidth', 1.8, ...
        'Color', colors(b, :), 'DisplayName', sprintf('%d planning updates', budgets(b)));
end
xlabel(ax, 'Episode'); ylabel(ax, 'Mean return'); title(ax, 'Dyna-Q: real experience versus planning');
lgd = legend(ax, 'Location', 'best'); lgd.Color = ax.Color; lgd.TextColor = ax.XColor;
results = struct('PlanningBudgets', budgets, 'Seeds', seeds, 'Returns', returns, ...
    'MeanReturn', meanReturn, 'TotalBackups', backups); %#ok<NASGU>
