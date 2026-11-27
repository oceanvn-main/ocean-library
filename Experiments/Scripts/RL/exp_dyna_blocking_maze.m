%% Dyna-Q+ on the Chapter 8 blocking maze
% Set QuickMode=true before running for a reduced smoke execution.
if ~exist('QuickMode','var'), QuickMode = false; end
scriptDirectory = fileparts(mfilename('fullpath')); addpath(scriptDirectory);
if QuickMode, seeds = 1:2; horizon = 350; changeStep = 120; episodes = 30;
else, seeds = 1:30; horizon = 3000; changeStep = 1000; episodes = 250; end

configuration = struct('ChangeStep',changeStep, ...
    'InitialBlockedRC',[3*ones(8,1) (2:9)'], ...
    'ChangedBlockedRC',[3*ones(8,1) (1:8)'], ...
    'Alpha',0.1,'Epsilon',0.1,'Gamma',0.95,'PlanningSteps',50, ...
    'Kappa',1e-3,'Horizon',horizon,'NumEpisodes',episodes, ...
    'MaxStepsPerEpisode',horizon,'Seed',0);
algorithms = ["dyna-q","dyna-q+"];
cumulativeReward = zeros(horizon,numel(algorithms),numel(seeds));
observed = zeros(numel(algorithms),numel(seeds));
for a = 1:numel(algorithms)
    for k = 1:numel(seeds)
        configuration.Seed = seeds(k);
        run = DynaExperiments.runChangingMaze(algorithms(a),configuration);
        cumulativeReward(:,a,k) = run.CumulativeReward;
        observed(a,k) = run.InteractionsObserved;
    end
end
results = struct('Experiment',"blocking-maze",'Algorithms',algorithms, ...
    'Seeds',seeds,'Configuration',configuration,'CumulativeReward',cumulativeReward, ...
    'MeanCumulativeReward',mean(cumulativeReward,3),'InteractionsObserved',observed); %#ok<NASGU>

viz = OceanViz.Core('Theme','Dark'); figure('Name','Dyna blocking maze');
ax = axes(); viz.styleAxes(ax); hold(ax,'on'); colors = lines(2);
for a = 1:2
    plot(ax,1:horizon,results.MeanCumulativeReward(:,a),'LineWidth',1.8, ...
        'Color',colors(a,:),'DisplayName',algorithms(a));
end
xline(ax,changeStep,'--','Environment changed','Color',ax.XColor);
xlabel(ax,'Real environment interactions'); ylabel(ax,'Cumulative reward');
title(ax,'Blocking maze: adaptation after the route closes');
lgd = legend(ax,'Location','northwest'); lgd.Color=ax.Color; lgd.TextColor=ax.XColor;
