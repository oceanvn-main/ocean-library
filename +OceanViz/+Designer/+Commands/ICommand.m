classdef (Abstract) ICommand < handle
    % ICOMMAND  Abstract Command Pattern Interface for Local Undo events.
    % 
    %   All local undo/redo actions (AddStation, DeleteEdge, MovePoint)
    %   must strictly implement this interface to participate in the 
    %   TrackDesigner TimelineController's Event Bus pipeline.
    
    methods (Abstract)
        % EXECUTE  Execute the forward change mathematically and visually.
        execute(obj)
        
        % UNDO  Surgically reverse the specific delta change.
        undo(obj)
        
        % GETNAME  Identify the command.
        name = getName(obj)
    end
end
