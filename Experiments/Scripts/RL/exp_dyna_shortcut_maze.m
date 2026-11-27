%% Dyna-Q+ on the Chapter 8 shortcut maze
% Set QuickMode=true before running for a reduced smoke execution.
if ~exist('QuickMode','var'), QuickMode = false; end
scriptDirectory = fileparts(mfilename('fullpath')); addpath(scriptDirectory);
if QuickMode, seeds = 1:2; horizon = 500; changeStep = 200; episodes = 40;
else, seeds = 1:30; horizon = 6000; changeStep = 3000; episodes = 400; end

configuration = struct('ChangeStep',changeStep, ...
    'InitialBlockedRC',[3*ones(8,1) (1:8)'], ...
    'ChangedBlockedRC',[3*ones(7,1) (1:7)'], ...
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
results = struct('Experiment',"shortcut-maze",'Algorithms',algorithms, ...
    'Seeds',seeds,'Configuration',configuration,'CumulativeReward',cumulativeReward, ...
    'MeanCumulativeReward',mean(cumulativeReward,3),'InteractionsObserved',observed); %#ok<NASGU>

viz = OceanViz.Core('Theme','Dark'); figure('Name','Dyna shortcut maze');
ax = axes(); viz.styleAxes(ax); hold(ax,'on'); colors = lines(2);
for a = 1:2
    plot(ax,1:horizon,results.MeanCumulativeReward(:,a),'LineWidth',1.8, ...
        'Color',colors(a,:),'DisplayName',algorithms(a));
end
xline(ax,changeStep,'--','Shortcut opened','Color',ax.XColor);
xlabel(ax,'Real environment interactions'); ylabel(ax,'Cumulative reward');
title(ax,'Shortcut maze: continued exploration discovers improvement');
lgd = legend(ax,'Location','northwest'); lgd.Color=ax.Color; lgd.TextColor=ax.XColor;
