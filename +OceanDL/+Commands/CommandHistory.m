classdef CommandHistory < handle
    % COMMANDHISTORY  Undo/redo stack manager for NNGraph mutations.
    %   COMMAND PATTERN: Each command í self-contained invertible unit.
    %   Fixed depth double stack (default 50 steps)
    %   Holds two stacks of INodeCommand objects:
    %     · UndoStack_ — commands that have been executed (can be undone)
    %     · RedoStack_ — commands that have been undone (can be re-executed)
    %
    %   Rules enforced automatically:
    %     · Executing a new command clears the redo stack (no branching history)
    %     · Stack depth is capped at MaxDepth_ (default: 100) to bound memory
    %
    %   Usage (called by NNGraph — do not use directly from experiment scripts):
    %     history = CommandHistory();
    %     history.execute(cmd, graph);   % run + push to undo stack
    %     history.undo(graph);           % pop + reverse
    %     history.redo(graph);           % pop from redo + re-run
    %     history.getLog();              % cell array of describe() strings
    %
    %   Design Pattern: Memento (light) — holds a linear stack of Commands
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Graph.NNGraph

    properties (Access = private)
        UndoStack_ cell   % cell{end} = most-recently executed command (LIFO)
        RedoStack_ cell   % cell{end} = most-recently undone command
        MaxDepth_  (1,1) double = 100
    end

    % ====================================================================
    methods

        function obj = CommandHistory(maxDepth)
            % COMMANDHISTORY  Construct with optional max stack depth.
            %   maxDepth - positive integer, default: 100
            if nargin < 1
                maxDepth = 100;
            end
            obj.MaxDepth_  = maxDepth;
            obj.UndoStack_ = {};
            obj.RedoStack_ = {};
        end

        % ----------------------------------------------------------------
        function execute(obj, cmd, graph)
            % EXECUTE  Run a command and push it onto the undo stack.
            %
            %   Any existing redo stack is cleared — new actions invalidate
            %   the "redo branch."
            %
            % INPUTS:
            %   cmd   - INodeCommand subclass instance
            %   graph - NNGraph handle (passed to cmd.execute)

            cmd.execute(graph);

            % Push to undo stack
            obj.UndoStack_{end+1} = cmd;

            % Cap depth — drop oldest entry if over limit
            if numel(obj.UndoStack_) > obj.MaxDepth_
                obj.UndoStack_(1) = [];
            end

            % New action invalidates redo branch
            obj.RedoStack_ = {};
        end

        % ----------------------------------------------------------------
        function undo(obj, graph)
            % UNDO  Reverse the most recent command.
            %
            %   Pops from UndoStack_, calls cmd.undo(graph), pushes to
            %   RedoStack_. No-op if stack is empty.
            %
            % INPUTS:
            %   graph - NNGraph handle
            if ~obj.canUndo()
                return;
            end
            cmd = obj.UndoStack_{end};
            obj.UndoStack_(end) = [];
            cmd.undo(graph);
            obj.RedoStack_{end+1} = cmd;
        end

        % ----------------------------------------------------------------
        function redo(obj, graph)
            % REDO  Re-execute the most recently undone command.
            %
            %   Pops from RedoStack_, calls cmd.execute(graph), pushes back
            %   to UndoStack_. No-op if redo stack is empty.
            %
            % INPUTS:
            %   graph - NNGraph handle
            if ~obj.canRedo()
                return;
            end
            cmd = obj.RedoStack_{end};
            obj.RedoStack_(end) = [];
            cmd.execute(graph);
            obj.UndoStack_{end+1} = cmd;
        end

        % ----------------------------------------------------------------
        function tf = canUndo(obj)
            % CANUNDO  True iff there is at least one command to undo.
            tf = ~isempty(obj.UndoStack_);
        end

        % ----------------------------------------------------------------
        function tf = canRedo(obj)
            % CANREDO  True iff there is at least one command to redo.
            tf = ~isempty(obj.RedoStack_);
        end

        % ----------------------------------------------------------------
        function clear(obj)
            % CLEAR  Empty both stacks (use when loading a new graph).
            obj.UndoStack_ = {};
            obj.RedoStack_ = {};
        end

        % ----------------------------------------------------------------
        function log = getLog(obj)
            % GETLOG  Return ordered log of executed commands as strings.
            %
            %   log{1} is the oldest, log{end} is the most recent.
            %
            % OUTPUTS:
            %   log - (1,N) cell of strings, one per command in undo stack
            n   = numel(obj.UndoStack_);
            log = cell(1, n);
            for i = 1:n
                log{i} = obj.UndoStack_{i}.describe();
            end
        end

        % ----------------------------------------------------------------
        function n = undoDepth(obj)
            % UNDODEPTH  Number of commands currently in the undo stack.
            n = numel(obj.UndoStack_);
        end

        % ----------------------------------------------------------------
        function n = redoDepth(obj)
            % REDODEPTH  Number of commands currently in the redo stack.
            n = numel(obj.RedoStack_);
        end

        % ----------------------------------------------------------------
        function disp(obj)
            fprintf('CommandHistory  [undo=%d, redo=%d, maxDepth=%d]\n', ...
                obj.undoDepth(), obj.redoDepth(), obj.MaxDepth_);
            if obj.canUndo()
                fprintf('  Last action : %s\n', obj.UndoStack_{end}.describe());
            end
        end

    end
end
