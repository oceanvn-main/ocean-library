classdef TinyRTDPModel < OceanStats.RL.Dyna.Models.IWorldModel
    %TINYRTDPMODEL Exact three-state stochastic MDP for independent DP tests.
    % State 1: action 1 branches to state 2 or terminal state 3;
    % action 2 enters state 3. State 2 enters state 3. State 3 is terminal.
    properties (SetAccess=private)
        FaultMode (1,1) string
    end
    methods
        function obj = TinyRTDPModel(faultMode)
            arguments
                faultMode (1,1) string {mustBeMember(faultMode,["none","bad_probability","unindexed_successor"])} = "none"
            end
            obj@OceanStats.RL.Dyna.Models.IWorldModel(3,2,logical([1 1;1 0;0 0]));
            obj.FaultMode = faultMode;
        end
        function tf = isReadOnly(~), tf = true; end
        function event = update(~,~) %#ok<STOUT>
            error('OceanTest:RL:RTDP:ReadOnly','Exact fixture is read only.');
        end
        function tf = contains(obj,s,a), tf = obj.ValidActionMask(s,a); end
        function pairs = observedPairs(obj), pairs = find(obj.ValidActionMask); end
        function outcome = sample(obj,s,a,stream)
            d = obj.distribution(s,a);
            k = find(rand(stream) <= cumsum([d.Probability]),1);
            outcome = d(k);
        end
        function d = distribution(obj,s,a)
            obj.validatePair(s,a,true);
            if s == 1 && a == 1
                d = [obj.row_(2,2,false,0.25),obj.row_(3,4,true,0.75)];
                if obj.FaultMode == "bad_probability", d(2).Probability = 0.5; end
                if obj.FaultMode == "unindexed_successor", d(1).NextStateIndex = 0; end
            elseif s == 1 && a == 2
                d = obj.row_(3,3,true,1);
            else
                d = obj.row_(3,1,true,1);
            end
        end
        function pred = predecessors(~,~), pred = []; end
        function snapshot = inspect(obj,s,a)
            snapshot = struct('StateIndex',s,'ActionIndex',a,'Outcomes',obj.distribution(s,a));
        end
        function reset(~), end
    end
    methods (Static, Access=private)
        function d = row_(next,reward,terminal,probability)
            d = struct('NextStateIndex',next,'Reward',reward, ...
                'Terminated',terminal,'Probability',probability);
        end
    end
end
