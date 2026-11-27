%% Prioritized sweeping versus uniform Dyna planning
% Comparison uses the same stationary maze and reports computation in total
% value backups (real plus simulated), rather than treating planning as free.
if ~exist('QuickMode','var'), QuickMode = false; end
scriptDirectory = fileparts(mfilename('fullpath')); addpath(scriptDirectory);
if QuickMode, seeds = 1:2; numEpisodes = 12;
else, seeds = 1:30; numEpisodes = 50; end
planningSteps = 5; algorithms = ["uniform","prioritized"];
episodeLength = zeros(numEpisodes,2,numel(seeds));
cumulativeBackups = zeros(numEpisodes,2,numel(seeds));
for a = 1:2
    for k = 1:numel(seeds)
        env = DynaExperiments.ChangingMazeEnv('ChangeStep',intmax('int32'), ...
            'InitialBlockedRC',[3*ones(8,1) (2:9)'], ...
            'ChangedBlockedRC',[3*ones(8,1) (2:9)']);
        mask = OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
        model = OceanStats.RL.Dyna.Models.DeterministicTabularModel(size(mask,1),size(mask,2),mask);
        if algorithms(a) == "uniform"
            planner = OceanStats.RL.Dyna.Planning.UniformPlanning();
        else
            planner = OceanStats.RL.Dyna.Planning.PrioritizedSweeping(1e-8);
        end
        observer = OceanStats.RL.Dyna.Observers.Recorder();
        learner = OceanStats.RL.Dyna.DynaLearner(env,'Model',model,'Planning',planner, ...
            'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(0.1), ...
            'Exploration',OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0.1), ...
            'Gamma',0.95,'PlanningSteps',planningSteps,'Seed',seeds(k),'Observer',observer);
        run = learner.run('NumEpisodes',numEpisodes,'MaxStepsPerEpisode',3000);
        episodeLength(:,a,k) = run.Lengths;
        % Count actual committed backups. Prioritized sweeping may exhaust
        % its queue before using the requested planning budget.
        episodeEndStep = cumsum(run.Lengths);
        updateRealStep = zeros(numel(observer.UpdateEvents),1);
        for u = 1:numel(observer.UpdateEvents)
            updateRealStep(u) = observer.UpdateEvents{u}.RealStep;
        end
        for ep = 1:numEpisodes
            cumulativeBackups(ep,a,k) = nnz(updateRealStep <= episodeEndStep(ep));
        end
    end
end
results = struct('Experiment',"prioritized-versus-uniform", ...
    'Algorithms',algorithms,'Seeds',seeds,'NumEpisodes',numEpisodes, ...
    'PlanningSteps',planningSteps,'EpisodeLength',episodeLength, ...
    'CumulativeBackups',cumulativeBackups,'MeanEpisodeLength',mean(episodeLength,3), ...
    'MeanCumulativeBackups',mean(cumulativeBackups,3)); %#ok<NASGU>

viz = OceanViz.Core('Theme','Dark'); figure('Name','Dyna planning efficiency');
ax = axes(); viz.styleAxes(ax); hold(ax,'on'); colors = lines(2);
for a = 1:2
    plot(ax,1:numEpisodes,results.MeanCumulativeBackups(:,a),'LineWidth',1.8, ...
        'Color',colors(a,:),'DisplayName',algorithms(a));
end
xlabel(ax,'Completed episodes'); ylabel(ax,'Cumulative value backups');
title(ax,'Planning efficiency: uniform versus prioritized sweeping');
lgd = legend(ax,'Location','northwest'); lgd.Color=ax.Color; lgd.TextColor=ax.XColor;
