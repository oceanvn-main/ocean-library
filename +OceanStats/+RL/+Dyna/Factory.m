classdef Factory
    % FACTORY Transparent constructors for supported tabular Dyna compositions.
    methods (Static)
        function learner=dynaQ(environment,options)
            arguments
                environment (1,1) OceanStats.RL.Env.IEnvironment
                options.Alpha (1,1) double {mustBePositive,mustBeLessThanOrEqual(options.Alpha,1)}=0.1
                options.Epsilon (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(options.Epsilon,1)}=0.1
                options.Gamma (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(options.Gamma,1)}=0.95
                options.PlanningSteps (1,1) double {mustBeInteger,mustBeNonnegative}=5
                options.Seed (1,1) double {mustBeInteger,mustBeNonnegative}=1
                options.Observer=OceanStats.RL.Dyna.Observers.NullObserver()
            end
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(environment);
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(size(mask,1),size(mask,2),mask);
            learner=OceanStats.RL.Dyna.DynaLearner(environment,'Model',model, ...
                'Planning',OceanStats.RL.Dyna.Planning.UniformPlanning(), ...
                'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(options.Alpha), ...
                'Exploration',OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(options.Epsilon), ...
                'Gamma',options.Gamma,'PlanningSteps',options.PlanningSteps,'Seed',options.Seed,'Observer',options.Observer);
        end
        function learner=dynaQPlus(environment,options)
            arguments
                environment (1,1) OceanStats.RL.Env.IEnvironment
                options.Kappa (1,1) double {mustBeNonnegative}=1e-3
                options.Alpha (1,1) double {mustBePositive,mustBeLessThanOrEqual(options.Alpha,1)}=0.1
                options.Epsilon (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(options.Epsilon,1)}=0.1
                options.Gamma (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(options.Gamma,1)}=0.95
                options.PlanningSteps (1,1) double {mustBeInteger,mustBeNonnegative}=5
                options.Seed (1,1) double {mustBeInteger,mustBeNonnegative}=1
                options.Observer=OceanStats.RL.Dyna.Observers.NullObserver()
            end
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(environment);
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(size(mask,1),size(mask,2),mask,'IncludeUnseenActions',true);
            learner=OceanStats.RL.Dyna.DynaLearner(environment,'Model',model,'Planning',OceanStats.RL.Dyna.Planning.UniformPlanning(), ...
                'Bonus',OceanStats.RL.Dyna.Bonuses.RecencyBonus(options.Kappa,mask), ...
                'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(options.Alpha), ...
                'Exploration',OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(options.Epsilon), ...
                'Gamma',options.Gamma,'PlanningSteps',options.PlanningSteps,'Seed',options.Seed,'Observer',options.Observer);
        end
    end
end
