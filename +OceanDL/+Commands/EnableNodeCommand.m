classdef EnableNodeCommand < OceanDL.Commands.INodeCommand
    % ENABLENODECOMMAND  Command to enable (re-activate) a disabled NNNode.
    %
    %   Undo restores the node to the Disabled state.
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Commands.DisableNodeCommand

    properties (Access = private)
        NodeID_        (1,1) string   % ID of the target node
        PreviousState_ (1,1) string   % captured during execute()
    end

    methods
        function obj = EnableNodeCommand(nodeID)
            % ENABLENODECOMMAND  Construct with the target node ID.
            %
            % INPUTS:
            %   nodeID - (1,1) string, e.g. "L2_N3"
            arguments
                nodeID (1,1) string
            end
            obj.NodeID_        = nodeID;
            obj.PreviousState_ = "";
        end

        function execute(obj, graph)
            node = graph.getNodeByID(obj.NodeID_);
            obj.PreviousState_ = node.getState();   % should be 'Disabled'
            node.enable();
        end

        function undo(obj, graph)
            node = graph.getNodeByID(obj.NodeID_);
            if strcmp(obj.PreviousState_, 'Disabled')
                node.disable();
            end
            % If it was already Active / Highlighted, nothing to undo
        end

        function s = describe(obj)
            s = sprintf("Enable %s", obj.NodeID_);
        end
    end
end
