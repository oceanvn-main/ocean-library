classdef ExactModelAdapter < OceanStats.RL.Dyna.Models.IWorldModel
    % EXACTMODELADAPTER Read-only indexed view of an IModelEnvironment.

    properties (SetAccess = immutable)
        Environment
    end

    methods
        function tf = isReadOnly(~), tf = true; end
        function obj = ExactModelAdapter(environment, validActionMask)
            arguments
                environment (1,1) OceanStats.RL.Env.IModelEnvironment
                validActionMask (:,:) logical = OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(environment)
            end
            states = environment.get_all_states();
            obj@OceanStats.RL.Dyna.Models.IWorldModel(size(states,1), size(validActionMask,2), validActionMask);
            obj.Environment = environment;
        end
        function event = update(~, ~) %#ok<STOUT>
            error('OceanStats:RL:Dyna:Models:ExactModelReadOnly','ExactModelAdapter is read-only.');
        end
        function tf = contains(obj, s, a), obj.validatePair(s,a,false); tf = obj.ValidActionMask(s,a); end
        function pairs = observedPairs(obj), pairs = find(obj.ValidActionMask); end
        function outcome = sample(obj, s, a, stream)
            arguments
                obj
                s (1,1) double {mustBeInteger,mustBePositive}
                a (1,1) double {mustBeInteger,mustBePositive}
                stream (1,1) RandStream
            end
            d = obj.distribution(s,a); c = cumsum([d.Probability]);
            outcome = d(find(rand(stream) <= c,1));
        end
        function outcomes = distribution(obj, s, a)
            obj.validatePair(s,a,true);
            state = obj.Environment.index_to_state(s); action = obj.actionLabel_(state,a);
            [next,p,reward] = obj.Environment.get_transitions(state,action);
            p = p(:); reward = reward(:);
            if numel(p) ~= size(next,1) || numel(reward) ~= numel(p) || any(p < 0) || abs(sum(p)-1) > 1e-9
                error('OceanStats:RL:Dyna:Models:BadExactDistribution','Environment returned an invalid transition distribution.');
            end
            outcomes = repmat(struct('NextStateIndex',0,'Reward',0,'Terminated',false,'Probability',0),numel(p),1);
            for k=1:numel(p)
                nextRow = next(k,:); nextIndex = obj.Environment.state_to_index(nextRow);
                terminated = obj.Environment.is_terminal(nextRow);
                if nextIndex == 0 && ~terminated
                    error('OceanStats:RL:Dyna:Models:UnindexableSuccessor','Nonterminal successor is outside the state table.');
                end
                outcomes(k)=struct('NextStateIndex',nextIndex,'Reward',reward(k), ...
                    'Terminated',logical(terminated),'Probability',p(k));
            end
        end
        function pred = predecessors(obj, nextStateIndex)
            if nextStateIndex < 1 || nextStateIndex > obj.NumStates || nextStateIndex ~= fix(nextStateIndex)
                error('OceanStats:RL:Dyna:Models:StateOutOfRange','Next-state index is out of range.');
            end
            pred = repmat(struct('StateIndex',0,'ActionIndex',0,'PairID',0,'Probability',0,'MeanReward',0),0,1);
            ids = obj.observedPairs();
            for j=1:numel(ids)
                [s,a]=ind2sub([obj.NumStates,obj.NumActions],ids(j)); d=obj.distribution(s,a);
                m=[d.NextStateIndex]==nextStateIndex;
                if any(m)
                    probability=sum([d(m).Probability]); meanReward=sum([d(m).Probability].*[d(m).Reward])/probability;
                    pred(end+1,1)=struct('StateIndex',s,'ActionIndex',a,'PairID',ids(j), ...
                        'Probability',probability,'MeanReward',meanReward); %#ok<AGROW>
                end
            end
        end
        function snapshot = inspect(obj,s,a)
            obj.validatePair(s,a,false); id=sub2ind([obj.NumStates,obj.NumActions],s,a);
            outcomes=repmat(struct('NextStateIndex',0,'Reward',0,'Terminated',false,'Probability',0),0,1);
            if obj.ValidActionMask(s,a), outcomes=obj.distribution(s,a); end
            snapshot=struct('StateIndex',s,'ActionIndex',a,'PairID',id,'Observed',obj.ValidActionMask(s,a), ...
                'ObservationCount',Inf,'Outcomes',outcomes,'Version',obj.Version);
        end
        function reset(~), end
    end
    methods (Access=private)
        function action=actionLabel_(obj,state,index)
            actions=obj.Environment.get_valid_actions(state);
            hits=arrayfun(@(x)obj.Environment.action_to_index(x),actions)==index;
            if nnz(hits)~=1, error('OceanStats:RL:Dyna:Models:ActionMapping','Cannot uniquely map action index to an environment label.'); end
            action=actions(find(hits,1));
        end
    end
    methods (Static)
        function mask=deriveMask(environment)
            states=environment.get_all_states(); labels=cell(size(states,1),1); maxIndex=0;
            for s=1:size(states,1)
                labels{s}=environment.get_valid_actions(states(s,:));
                for action=labels{s}(:)', maxIndex=max(maxIndex,environment.action_to_index(action)); end
            end
            if maxIndex==0, error('OceanStats:RL:Dyna:Models:NoActions','Environment exposes no valid actions.'); end
            mask=false(size(states,1),maxIndex);
            for s=1:size(states,1)
                for action=labels{s}(:)', mask(s,environment.action_to_index(action))=true; end
            end
        end
    end
end
