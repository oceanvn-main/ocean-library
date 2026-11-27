classdef DisableNodeCommand < OceanDL.Commands.INodeCommand
    % DISABLENODECOMMAND  Command to disable a single NNNode.
    %
    %   Captures the node's previous state before disabling so that
    %   undo() can restore it exactly (Active → back to Active,
    %   Highlighted → back to Highlighted).
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Commands.EnableNodeCommand

    properties (Access = private)
        NodeID_        (1,1) string   % ID of the target node
        PreviousState_ (1,1) string   % captured during execute(), used by undo()
    end

    methods
        function obj = DisableNodeCommand(nodeID)
            % DISABLENODECOMMAND  Construct with the target node ID.
            %
            % INPUTS:
            %   nodeID - (1,1) string, e.g. "L2_N3"
            arguments
                nodeID (1,1) string
            end
            obj.NodeID_        = nodeID;
            obj.PreviousState_ = "";    % filled in during execute()
        end

        function execute(obj, graph)
            % Capture state BEFORE mutating — required for correct undo.
            node = graph.getNodeByID(obj.NodeID_);
            obj.PreviousState_ = node.getState();
            node.disable();
        end

        function undo(obj, graph)
            node = graph.getNodeByID(obj.NodeID_);
            switch obj.PreviousState_
                case 'Active'
                    node.enable();
                case 'Highlighted'
                    node.enable();
                    node.highlight();
                otherwise
                    % Was already disabled — do nothing
            end
        end

        function s = describe(obj)
            s = sprintf("Disable %s", obj.NodeID_);
        end
    end
end
