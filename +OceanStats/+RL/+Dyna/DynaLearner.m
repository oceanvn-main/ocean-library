classdef DynaLearner < handle
    % DYNALEARNER Tabular Dyna-Q orchestration with explicit model and planner.
    properties (SetAccess = immutable)
        Environment
        Model
        Planning
        Bonus
        Values OceanStats.RL.TD.Values.TabularQ
        Target
        StepSize
        Exploration
        Observer
        Gamma (1,1) double
        PlanningSteps (1,1) double
        Seed (1,1) double
        ValidActionMask logical
        ActionValues double
    end
    properties (Access=private)
        InitialQ double
        InitialCounts double
        BehaviourStream
        ModelStream
        PlanningStream
        RealStep (1,1) double = 0
        PlanningStep (1,1) double = 0
    end
    methods
        function obj=DynaLearner(environment,options)
            arguments
                environment (1,1) OceanStats.RL.Env.IEnvironment
                options.Model (1,1) OceanStats.RL.Dyna.Models.IWorldModel
                options.Planning (1,1) OceanStats.RL.Dyna.Planning.IPlanningStrategy = OceanStats.RL.Dyna.Planning.UniformPlanning()
                options.Bonus (1,1) OceanStats.RL.Dyna.Bonuses.IPlanningBonus = OceanStats.RL.Dyna.Bonuses.NoBonus()
                options.Values = []
                options.Target (1,1) OceanStats.RL.TD.Targets.ITarget = OceanStats.RL.TD.Targets.QLearningTarget()
                options.StepSize (1,1) OceanStats.RL.TD.StepSize.IStepSize = OceanStats.RL.TD.StepSize.ConstantStepSize(0.1)
                options.Exploration (1,1) OceanStats.RL.TD.Exploration.IExploration = OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0.1)
                options.Observer (1,1) OceanStats.RL.Dyna.Observers.IObserver = OceanStats.RL.Dyna.Observers.NullObserver()
                options.Gamma (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(options.Gamma,1)} = 0.95
                options.PlanningSteps (1,1) double {mustBeInteger,mustBeNonnegative} = 5
                options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 1
            end
            if ~isa(environment,'OceanStats.RL.Env.IEnumerableEnvironment')
                error('OceanStats:RL:Dyna:NonEnumerable','Dyna requires an enumerable finite environment.');
            end
            [mask,actions]=obj.describeEnvironment_(environment);
            if ~isequal(mask,options.Model.ValidActionMask)
                error('OceanStats:RL:Dyna:MaskMismatch','Environment and model action masks differ.');
            end
            values=options.Values;
            if isempty(values), values=OceanStats.RL.TD.Values.TabularQ(mask,0); end
            if ~isa(values,'OceanStats.RL.TD.Values.TabularQ') || ~isequal(size(values.finalQ()),size(mask))
                error('OceanStats:RL:Dyna:ValueMismatch','Values must be a matching TabularQ.');
            end
            obj.Environment=environment; obj.Model=options.Model; obj.Planning=options.Planning;
            obj.Bonus=options.Bonus; obj.Values=values; obj.Target=options.Target; obj.StepSize=options.StepSize;
            obj.Exploration=options.Exploration; obj.Observer=options.Observer; obj.Gamma=options.Gamma;
            obj.PlanningSteps=options.PlanningSteps; obj.Seed=options.Seed; obj.ValidActionMask=mask; obj.ActionValues=actions;
            obj.InitialQ=values.finalQ(); obj.InitialCounts=values.Counts;
        end
        function result=run(obj,options)
            arguments
                obj
                options.NumEpisodes (1,1) double {mustBeInteger,mustBePositive}=1
                options.MaxStepsPerEpisode (1,1) double {mustBeInteger,mustBePositive}=1000
            end
            obj.restart_(); E=options.NumEpisodes; returns=zeros(E,1); lengths=zeros(E,1);
            terminated=false(E,1); truncated=false(E,1); obj.Observer.startRun();
            for ep=1:E
                [obs,info]=obj.Environment.reset('Seed',obj.Seed+ep-1); si=obj.index_(obs,false); ret=0;
                for t=1:options.MaxStepsPerEpisode
                    valid=obj.mask_(si,info); ai=obj.Exploration.select(obj.Values.behaviourEstimates(si),valid,(ep-1)/E,obj.BehaviourStream);
                    [next,r,term,trunc,info2]=obj.Environment.step(obj.ActionValues(ai));
                    if ~term && ~trunc && t==options.MaxStepsPerEpisode, trunc=true; end
                    s2i=obj.index_(next,term); obj.RealStep=obj.RealStep+1;
                    tr=OceanStats.RL.Dyna.Core.Transition('StateIndex',si,'ActionIndex',ai,'Reward',r, ...
                        'NextStateIndex',s2i,'Terminated',logical(term),'Truncated',logical(trunc), ...
                        'Origin',"real",'RealStep',obj.RealStep,'PlanningStep',obj.PlanningStep);
                    obj.Observer.realTransition(tr);
                    if ~obj.Model.isReadOnly()
                        me=obj.Model.update(tr); obj.Observer.modelUpdated(me);
                    end
                    obj.Planning.observeReal(tr,obj.Model,obj.Values,obj.Gamma);
                    obj.update_(tr,0); obj.Bonus.onRealVisit(si,ai,obj.RealStep);
                    for p=1:obj.PlanningSteps
                        if ~obj.Planning.canPlan(obj.Model), break; end
                        c=obj.Planning.next(obj.Model,obj.Values,obj.Gamma,obj.PlanningStream);
                        obj.PlanningStep=obj.PlanningStep+1;
                        if obj.Model.planningBackupMode()=="expectation" && ...
                                isa(obj.Planning,'OceanStats.RL.Dyna.Planning.PrioritizedSweeping')
                            ev=obj.updateExpected_(c);
                        else
                            o=obj.Model.sample(c.StateIndex,c.ActionIndex,obj.ModelStream);
                            reward=o.Reward+obj.Bonus.evaluate(c.StateIndex,c.ActionIndex,obj.RealStep,[]);
                            pt=OceanStats.RL.Dyna.Core.Transition('StateIndex',c.StateIndex,'ActionIndex',c.ActionIndex, ...
                                'Reward',reward,'NextStateIndex',o.NextStateIndex,'Terminated',logical(o.Terminated), ...
                                'Truncated',false,'Origin',"planning",'RealStep',obj.RealStep,'PlanningStep',obj.PlanningStep);
                            ev=obj.update_(pt,c.Priority);
                        end
                        obj.Planning.afterUpdate(ev,obj.Model,obj.Values,obj.Gamma);
                    end
                    ret=ret+r; lengths(ep)=t; si=s2i; info=info2;
                    if term||trunc, terminated(ep)=term; truncated(ep)=trunc; break; end
                end
                returns(ep)=ret; obj.Observer.finishEpisode(struct('Return',ret,'Length',lengths(ep), ...
                    'Terminated',terminated(ep),'Truncated',truncated(ep)));
            end
            result=struct('Q',obj.Values.finalQ(),'Counts',obj.Values.Counts,'Returns',returns,'Lengths',lengths, ...
                'Terminated',terminated,'Truncated',truncated,'RealSteps',obj.RealStep,'PlanningSteps',obj.PlanningStep, ...
                'Model',obj.Model,'Planning',obj.Planning.inspect(),'Seed',obj.Seed);
            obj.Observer.finishRun(result);
        end
    end
    methods (Access=private)
        function ev=update_(obj,tr,priority)
            s=tr.StateIndex; a=tr.ActionIndex; old=obj.Values.get(s,a);
            if tr.Terminated, qNext=zeros(1,size(obj.ValidActionMask,2)); valid=false(size(qNext));
            else, qNext=obj.Values.estimates(tr.NextStateIndex); valid=obj.ValidActionMask(tr.NextStateIndex,:); end
            y=obj.Target.compute(tr.Reward,qNext,0,tr.Terminated,obj.Gamma,[],valid);
            ev=obj.commitTarget_(tr.Origin,s,a,old,y,priority);
        end
        function ev=updateExpected_(obj,c)
            outcomes=obj.Model.distribution(c.StateIndex,c.ActionIndex); bonus=obj.Bonus.evaluate(c.StateIndex,c.ActionIndex,obj.RealStep,[]);
            y=0;
            for k=1:numel(outcomes)
                o=outcomes(k); target=o.Reward+bonus;
                if ~o.Terminated
                    q=obj.Values.estimates(o.NextStateIndex); valid=obj.ValidActionMask(o.NextStateIndex,:);
                    target=target+obj.Gamma*max(q(valid));
                end
                y=y+o.Probability*target;
            end
            old=obj.Values.get(c.StateIndex,c.ActionIndex);
            ev=obj.commitTarget_("planning",c.StateIndex,c.ActionIndex,old,y,c.Priority);
        end
        function ev=commitTarget_(obj,origin,s,a,old,y,priority)
            n=obj.Values.touch(s,a); alpha=obj.StepSize.alpha(n); obj.Values.update(s,a,alpha,y); new=obj.Values.get(s,a);
            ev=OceanStats.RL.Dyna.Core.UpdateEvent('Origin',origin,'StateIndex',s,'ActionIndex',a, ...
                'OldQ',old,'Target',y,'TDError',y-old,'Alpha',alpha,'NewQ',new,'PlanningPriority',priority, ...
                'RealStep',obj.RealStep,'PlanningStep',obj.PlanningStep,'ModelVersion',double(obj.Model.Version));
            obj.Observer.valueUpdated(ev);
        end
        function restart_(obj)
            obj.Values.setTables(obj.InitialQ,obj.InitialCounts); obj.Model.reset(); obj.Planning.reset(); obj.Bonus.reset();
            obj.RealStep=0; obj.PlanningStep=0;
            obj.BehaviourStream=RandStream('mrg32k3a','Seed',obj.Seed+1);
            obj.ModelStream=RandStream('mrg32k3a','Seed',obj.Seed+2);
            obj.PlanningStream=RandStream('mrg32k3a','Seed',obj.Seed+3);
        end
        function idx=index_(obj,obs,terminal)
            idx=obj.Environment.state_to_index(obs);
            if isempty(idx)||idx==0
                if terminal, idx=0; else, error('OceanStats:RL:Dyna:StateOutsideTable','Nonterminal observation is outside the table.'); end
            end
        end
        function valid=mask_(obj,si,info)
            valid=obj.ValidActionMask(si,:);
            if isstruct(info)&&isfield(info,'action_mask')&&~isequal(logical(info.action_mask(:)'),valid)
                error('OceanStats:RL:Dyna:DynamicLegality','v1 requires stationary action legality.');
            end
        end
    end
    methods (Static,Access=private)
        function [mask,labels]=describeEnvironment_(env)
            states=env.get_all_states(); S=size(states,1); A=env.ActionSpace.N; mask=false(S,A); labels=zeros(1,A); seen=false(1,A);
            for s=1:S
                acts=env.get_valid_actions(states(s,:));
                for action=acts(:)'
                    k=env.action_to_index(action); mask(s,k)=true; labels(k)=action; seen(k)=true;
                end
            end
            if ~all(seen), labels(~seen)=env.ActionSpace.Start-1+find(~seen); end
        end
    end
end
