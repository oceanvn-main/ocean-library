classdef PrioritizedSweeping < OceanStats.RL.Dyna.Planning.IPlanningStrategy
    % PRIORITIZEDSWEEPING Max-priority scheduler with predecessor propagation.
    properties (SetAccess = immutable)
        PriorityThreshold (1,1) double
    end
    properties (Access = private)
        Queue OceanStats.RL.Dyna.Planning.MaxPriorityQueue
        EnqueueCount (1,1) uint64 = uint64(0)
        SelectionCount (1,1) uint64 = uint64(0)
    end
    methods
        function obj = PrioritizedSweeping(priorityThreshold)
            arguments
                priorityThreshold (1,1) double {mustBeFinite,mustBeNonnegative} = 1e-8
            end
            obj.PriorityThreshold = priorityThreshold;
            obj.Queue = OceanStats.RL.Dyna.Planning.MaxPriorityQueue();
        end
        function observeReal(obj, transition, model, values, gamma)
            [s,a] = obj.transitionPair_(transition);
            obj.enqueueIfNeeded_(s, a, model, values, gamma);
        end
        function tf = canPlan(obj, ~), tf = ~obj.Queue.isEmpty(); end
        function candidate = next(obj, model, ~, ~, stream) %#ok<INUSD>
            [id, priority] = obj.Queue.pop();
            [s,a] = ind2sub([model.NumStates, model.NumActions], id);
            obj.SelectionCount = obj.SelectionCount + 1;
            candidate = struct('StateIndex', s, 'ActionIndex', a, 'PairID', id, ...
                'Priority', priority, 'ModelVersion', model.Version);
        end
        function afterUpdate(obj, event, model, values, gamma)
            stateIndex = obj.field_(event, 'StateIndex');
            pred = model.predecessors(stateIndex);
            for k = 1:numel(pred)
                obj.enqueueIfNeeded_(pred(k).StateIndex, pred(k).ActionIndex, model, values, gamma);
            end
        end
        function reset(obj)
            obj.Queue.reset(); obj.EnqueueCount=uint64(0); obj.SelectionCount=uint64(0);
        end
        function stats = inspect(obj)
            stats = struct('Name', "prioritized-sweeping", 'PriorityThreshold', obj.PriorityThreshold, ...
                'EnqueueCount', double(obj.EnqueueCount), 'SelectionCount', double(obj.SelectionCount), ...
                'Queue', obj.Queue.inspect());
        end
    end
    methods (Access = private)
        function enqueueIfNeeded_(obj,s,a,model,values,gamma)
            if ~model.contains(s,a), return; end
            priority = obj.expectedPriority_(s,a,model,values,gamma);
            if priority > obj.PriorityThreshold
                id = sub2ind([model.NumStates,model.NumActions],s,a);
                obj.Queue.push(id,priority); obj.EnqueueCount=obj.EnqueueCount+1;
            end
        end
        function p = expectedPriority_(~,s,a,model,values,gamma)
            outcomes=model.distribution(s,a); expectedTarget=0;
            for j=1:numel(outcomes)
                o=outcomes(j); target=o.Reward;
                if ~o.Terminated
                    qNext=values.estimates(o.NextStateIndex);
                    valid=model.ValidActionMask(o.NextStateIndex,:);
                    if ~any(valid)
                        error('OceanStats:RL:Dyna:Planning:NoValidNextAction', ...
                            'A nonterminal modeled successor must have a valid action.');
                    end
                    target=target+gamma*max(qNext(valid));
                end
                expectedTarget=expectedTarget+o.Probability*target;
            end
            p=abs(expectedTarget-values.get(s,a));
        end
    end
    methods (Static, Access = private)
        function [s,a]=transitionPair_(t)
            s=OceanStats.RL.Dyna.Planning.PrioritizedSweeping.field_(t,'StateIndex');
            a=OceanStats.RL.Dyna.Planning.PrioritizedSweeping.field_(t,'ActionIndex');
        end
        function value=field_(record,name)
            if (isstruct(record)&&isfield(record,name)) || (isobject(record)&&isprop(record,name))
                value=record.(name); return
            end
            error('OceanStats:RL:Dyna:Planning:BadRecord','Record lacks %s.',name);
        end
    end
end
