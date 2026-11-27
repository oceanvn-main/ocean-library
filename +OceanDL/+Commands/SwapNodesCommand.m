classdef SwapNodesCommand < OceanDL.Commands.INodeCommand
    % SWAPNODESCOMMAND  Command to swap two NNNodes within the same layer.
    %
    %   KEY INSIGHT: swap(A,B) is its own inverse — swapping the same two
    %   nodes twice restores the original state. Therefore:
    %
    %     undo() == execute()   (same code, not a mistake)
    %
    %   This property is guaranteed by the permutation-symmetry design of
    %   NNGraph.swapNodes_(), which was verified in the Phase 1 test suite
    %   (max output difference = 0.00e+00 after double-swap).
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Graph.NNGraph.swapNodes_

    properties (Access = private)
        IdA_ (1,1) string   % first node ID
        IdB_ (1,1) string   % second node ID
    end

    methods
        function obj = SwapNodesCommand(idA, idB)
            % SWAPNODESCOMMAND  Construct with the two node IDs to swap.
            %
            % INPUTS:
            %   idA - (1,1) string, e.g. "L2_N1"
            %   idB - (1,1) string, e.g. "L2_N3"
            arguments
                idA (1,1) string
                idB (1,1) string
            end
            obj.IdA_ = idA;
            obj.IdB_ = idB;
        end

        function execute(obj, graph)
            graph.swapNodes_(obj.IdA_, obj.IdB_);
        end

        function undo(obj, graph)
            % Swap is its own inverse — exact same operation.
            graph.swapNodes_(obj.IdA_, obj.IdB_);
        end

        function s = describe(obj)
            s = sprintf("Swap %s ↔ %s", obj.IdA_, obj.IdB_);
        end
    end
end
