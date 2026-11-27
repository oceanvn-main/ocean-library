classdef (Abstract) INodeCommand < handle
    % INODECOMMAND  Abstract base class for all NNGraph mutation commands.
    %
    %   Every mutation on NNGraph (disable, enable, swap, setWeight, setBias)
    %   is wrapped in a concrete subclass of this interface. This enables:
    %
    %     · Undo/redo via CommandHistory
    %     · Human-readable audit log via describe()
    %     · Future: serialisation, macro-commands (Phase 4+)
    %
    %   Design Pattern: Command (GoF)
    %
    %   Concrete implementations:
    %     DisableNodeCommand  — disable a single NNNode
    %     EnableNodeCommand   — enable a single NNNode
    %     SwapNodesCommand    — swap two NNNodes in the same layer
    %     SetWeightCommand    — set an NNEdge weight scalar
    %     SetBiasCommand      — set an NNNode bias scalar
    %
    %   Usage (by CommandHistory only — do not call execute/undo directly):
    %     history.execute(cmd, graph);   % executes + pushes to undo stack
    %     history.undo(graph);           % pops + calls cmd.undo(graph)
    %     history.redo(graph);           % pops + calls cmd.execute(graph)
    %
    % See also: OceanDL.Commands.CommandHistory

    methods (Abstract)

        execute(obj, graph)
        % EXECUTE  Apply the mutation to the graph.
        %
        %   Must capture any state needed for undo BEFORE mutating.
        %   Called by CommandHistory.execute() and CommandHistory.redo().
        %
        %   INPUTS:
        %     graph - OceanDL.Graph.NNGraph handle

        undo(obj, graph)
        % UNDO  Reverse the mutation exactly.
        %
        %   Must restore the graph to the state it was in before
        %   execute() was called.
        %
        %   INPUTS:
        %     graph - OceanDL.Graph.NNGraph handle

        s = describe(obj)
        % DESCRIBE  Return a human-readable string describing this command.
        %
        %   OUTPUTS:
        %     s - (1,1) string, e.g. "Disable L2_N3"

    end
end
