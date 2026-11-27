classdef SetBiasCommand < OceanDL.Commands.INodeCommand
    % SETBIASCOMMAND  Command to set a single NNNode bias.
    %
    %   OldBias_ is captured during execute() so that stacking multiple
    %   SetBiasCommand on the same node undoes correctly.
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Commands.SetWeightCommand

    properties (Access = private)
        NodeID_  (1,1) string   % target node ID
        NewBias_ (1,1) double   % desired bias after execute()
        OldBias_ (1,1) double   % previous bias, captured in execute()
    end

    methods
        function obj = SetBiasCommand(nodeID, newBias)
            % SETBIASCOMMAND  Construct for the given node.
            %
            % INPUTS:
            %   nodeID  - (1,1) string, e.g. "L2_N3"
            %   newBias - (1,1) double, bias value to assign
            arguments
                nodeID  (1,1) string
                newBias (1,1) double
            end
            obj.NodeID_  = nodeID;
            obj.NewBias_ = newBias;
            obj.OldBias_ = NaN;     % filled in during execute()
        end

        function execute(obj, graph)
            node         = graph.getNodeByID(obj.NodeID_);
            obj.OldBias_ = node.Bias_;          % capture before mutating
            node.Bias_   = obj.NewBias_;
        end

        function undo(obj, graph)
            node       = graph.getNodeByID(obj.NodeID_);
            node.Bias_ = obj.OldBias_;
        end

        function s = describe(obj)
            s = sprintf("Set bias %s :  %.4f → %.4f", ...
                obj.NodeID_, obj.OldBias_, obj.NewBias_);
        end
    end
end
