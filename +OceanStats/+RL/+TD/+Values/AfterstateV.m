classdef AfterstateV < OceanStats.RL.TD.Values.TabularV
    % AFTERSTATEV  Tabular afterstate state-value function V(s').
    %   Sutton & Barto (2018) Sec 6.8: Afterstates.
    properties (Constant)
        AfterstateLatex = "V(s') \leftarrow V(s') + \alpha \left[ R + \gamma V(s'') - V(s') \right]"
        AfterstateReference = "Sutton & Barto (2018) Sec 6.8, Afterstates"
    end
    methods
        function obj = AfterstateV(numAfterstates, initialValue)
            if nargin < 2, initialValue = 0; end
            obj@OceanStats.RL.TD.Values.TabularV(numAfterstates, initialValue);
        end
        function s = name(~), s = "afterstatev"; end
    end
end
