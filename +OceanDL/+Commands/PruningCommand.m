classdef PruningCommand < OceanDL.Commands.INodeCommand
    % PRUNINGCOMMAND  Zero-out small weights (|w| < epsilon).
    %
    %   This command iterates through all active edges in a graph and
    %   sets their weights to exactly 0 if their magnitude is below
    %    a user-defined threshold.
    %
    %   Undoable: Stores a full graph snapshot before pruning to allow
    %   bit-perfect restoration of all weights.
    %
    %   Usage:
    %     cmd = OceanDL.Commands.PruningCommand(0.05);
    %     graph.execute(cmd);
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Graph.NNGraphMemento

    properties (Access = private)
        Threshold (1,1) double
        Memento_  % OceanDL.Graph.NNGraphMemento
    end

    methods
        function obj = PruningCommand(threshold)
            obj.Threshold = threshold;
        end

        function execute(obj, graph)
            % 1. Capture state for undo
            obj.Memento_ = graph.saveSnapshot();

            % 2. Prune small weights
            edges = graph.getActiveEdges();
            count = 0;
            for i = 1:numel(edges)
                e = edges{i};
                if abs(e.Weight_) < obj.Threshold
                    e.Weight_ = 0;
                    count     = count + 1;
                end
            end
            
            % Note: We don't notify here; the NNGraph.execute() wrapper 
            % usually handles notification if implemented that way, 
            % but in our set-up the graph calls notify after command execution.
        end

        function undo(obj, graph)
            if ~isempty(obj.Memento_)
                graph.loadSnapshot(obj.Memento_);
            end
        end

        function str = describe(obj)
            str = sprintf('Pruned weights with |w| < %.3f', obj.Threshold);
        end
    end
end
