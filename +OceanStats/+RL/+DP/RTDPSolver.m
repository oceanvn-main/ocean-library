classdef RTDPSolver < handle
    % RTDPSOLVER Bounded real-time dynamic programming on an exact model.
    properties (SetAccess = private)
        Model (1,1) OceanStats.RL.Dyna.Models.IWorldModel
        ValueVector (:,1) double
        PolicyVector (:,1) double
        VisitCount (:,1) double
        DiscountFactor (1,1) double
        MaxDepth (1,1) double
        TieTolerance (1,1) double
        Stream (1,1) RandStream
        LastTrajectory struct
    end
    methods
        function obj = RTDPSolver(model, opts)
            arguments
                model
                opts.DiscountFactor (1,1) double {mustBeInRange(opts.DiscountFactor,0,1)} = 1
                opts.InitialValue (:,1) double {mustBeFinite} = []
                opts.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 1
                opts.MaxDepth (1,1) double {mustBeInteger,mustBePositive} = 1000
                opts.TieTolerance (1,1) double {mustBeNonnegative} = 1e-9
            end
            if isa(model,'OceanStats.RL.Env.IModelEnvironment')
                model = OceanStats.RL.Dyna.Models.ExactModelAdapter(model);
            elseif ~isa(model,'OceanStats.RL.Dyna.Models.IWorldModel') || ~isscalar(model)
                error('OceanStats:RL:DP:BadModel','Model must be IModelEnvironment or IWorldModel.');
            end
            if isempty(opts.InitialValue), opts.InitialValue = zeros(model.NumStates,1); end
            if numel(opts.InitialValue) ~= model.NumStates
                error('OceanStats:RL:DP:BadInitialValue','InitialValue must have one element per model state.');
            end
            obj.Model = model; obj.ValueVector = opts.InitialValue;
            obj.PolicyVector = zeros(model.NumStates,1);
            obj.VisitCount = zeros(model.NumStates,1);
            obj.DiscountFactor = opts.DiscountFactor;
            obj.MaxDepth = opts.MaxDepth; obj.TieTolerance = opts.TieTolerance;
            obj.Stream = RandStream('mrg32k3a','Seed',opts.Seed);
            obj.LastTrajectory = obj.emptyTrace_();
        end

        function trace = runTrial(obj,startState)
            obj.checkState_(startState);
            trace = obj.emptyTrace_(); state = startState;
            trace.Steps = repmat(struct('StateIndex',0,'ActionIndex',0,'ActionValues',[], ...
                'NextStateIndex',0,'Reward',0,'Terminated',false, ...
                'ValueBefore',0,'ValueAfter',0,'Residual',0),obj.MaxDepth,1);
            used = 0;
            for depth = 1:obj.MaxDepth
                [newValue,action,q,d] = OceanStats.RL.DP.BellmanOptimality.backup( ...
                    obj.Model,state,obj.ValueVector,obj.DiscountFactor,obj.TieTolerance);
                if action == 0
                    trace.TerminationReason = "terminal_state"; break
                end
                oldValue = obj.ValueVector(state);
                obj.ValueVector(state) = newValue;
                obj.PolicyVector(state) = action;
                obj.VisitCount(state) = obj.VisitCount(state)+1;
                outcomes = d{action}; cumulative = cumsum([outcomes.Probability]);
                k = find(rand(obj.Stream) <= cumulative,1,'first');
                if isempty(k), k = numel(outcomes); end
                sampled = outcomes(k);
                item = struct('StateIndex',state,'ActionIndex',action, ...
                    'ActionValues',q,'NextStateIndex',sampled.NextStateIndex, ...
                    'Reward',sampled.Reward,'Terminated',logical(sampled.Terminated), ...
                    'ValueBefore',oldValue,'ValueAfter',newValue, ...
                    'Residual',abs(newValue-oldValue));
                used = used+1;
                trace.Steps(used) = item;
                if sampled.Terminated
                    trace.TerminationReason = "terminal_transition"; break
                end
                state = sampled.NextStateIndex;
            end
            trace.Steps = trace.Steps(1:used);
            if trace.TerminationReason == "", trace.TerminationReason = "depth_limit"; end
            trace.Backups = used;
            obj.LastTrajectory = trace;
        end

        function [values,policy,result] = solve(obj,startState,opts)
            arguments
                obj
                startState (1,1) double {mustBeInteger,mustBePositive}
                opts.MaxTrials (1,1) double {mustBeInteger,mustBePositive} = 1000
                opts.MaxBackups (1,1) double {mustBeInteger,mustBePositive} = 100000
                opts.Tolerance (1,1) double {mustBeNonnegative} = 1e-9
            end
            obj.checkState_(startState);
            trials = 0; backups = 0; reason = "trial_budget";
            while trials < opts.MaxTrials && backups < opts.MaxBackups
                % Bound each trial by the remaining backup budget.
                savedDepth = obj.MaxDepth;
                obj.MaxDepth = min(savedDepth,opts.MaxBackups-backups);
                try
                    trace = obj.runTrial(startState);
                catch err
                    obj.MaxDepth = savedDepth; rethrow(err)
                end
                obj.MaxDepth = savedDepth;
                trials = trials+1; backups = backups+trace.Backups;
                if trace.Backups == 0 && trace.TerminationReason == "terminal_state"
                    reason = "terminal_start"; break
                end
                if backups >= opts.MaxBackups
                    reason = "backup_budget"; break
                end
            end
            explored = find(obj.VisitCount>0);
            for k = 1:numel(explored)
                s = explored(k);
                [~,action] = OceanStats.RL.DP.BellmanOptimality.backup( ...
                    obj.Model,s,obj.ValueVector,obj.DiscountFactor,obj.TieTolerance);
                obj.PolicyVector(s) = action;
            end
            values = obj.ValueVector; policy = obj.PolicyVector;
            result = struct('Trials',trials,'Backups',backups, ...
                'TerminationReason',reason,'Converged',false, ...
                'Tolerance',opts.Tolerance,'ExploredStates',explored, ...
                'LastTrajectory',obj.LastTrajectory);
        end

        function value = getValue(obj,stateIndex)
            obj.checkState_(stateIndex); value = obj.ValueVector(stateIndex);
        end
        function action = getPolicy(obj,stateIndex)
            obj.checkState_(stateIndex);
            if obj.VisitCount(stateIndex)==0, action = 0; return; end
            [~,action] = OceanStats.RL.DP.BellmanOptimality.backup( ...
                obj.Model,stateIndex,obj.ValueVector,obj.DiscountFactor,obj.TieTolerance);
            obj.PolicyVector(stateIndex) = action;
        end
        function trace = getLastTrajectory(obj), trace = obj.LastTrajectory; end
        function metadata = getMetadata(obj)
            metadata = struct('Algorithm',"RTDP",'NumStates',obj.Model.NumStates, ...
                'NumActions',obj.Model.NumActions,'DiscountFactor',obj.DiscountFactor, ...
                'MaxDepth',obj.MaxDepth,'TieTolerance',obj.TieTolerance, ...
                'VisitedStates',nnz(obj.VisitCount));
        end
    end
    methods (Access = private)
        function checkState_(obj,s)
            if ~(isnumeric(s) && isscalar(s) && isfinite(s) && s==fix(s) && s>=1 && s<=obj.Model.NumStates)
                error('OceanStats:RL:DP:BadState','State must be an indexed model state.');
            end
        end
    end
    methods (Static, Access = private)
        function trace = emptyTrace_()
            steps = repmat(struct('StateIndex',0,'ActionIndex',0,'ActionValues',[], ...
                'NextStateIndex',0,'Reward',0,'Terminated',false, ...
                'ValueBefore',0,'ValueAfter',0,'Residual',0),0,1);
            trace = struct('Steps',steps,'Backups',0,'TerminationReason',"");
        end
    end
end
