classdef TimelineController < handle
    % UNDOMANAGER  Centralized Event Bus tracking Local Undo/Redo deltas.
    %
    %   Replaces the TrackDesigner's archaic full-canvas snapshot stack. 
    %   It holds an array of specialized ICommand objects and gracefully 
    %   navigates forwards and backwards via a local index pointer.
    
    properties (SetAccess = private)
        HistoryStack_     cell = {}        % Ordered array of ICommands
        CurrentIndex_  (1,1) double = 0 % Where we are in the history
        MaxHistory_    (1,1) double = 40 % Increased delta tracking capacity
    end
    
    methods
        function execute(obj, cmd)
            % EXECUTE  Runs a new operation and pushes it to history.
            %          If we're in the middle of the stack, futures are purged.
            cmd.execute();
            
            % Discard branching futures
            if obj.CurrentIndex_ < numel(obj.HistoryStack_)
                obj.HistoryStack_ = obj.HistoryStack_(1:obj.CurrentIndex_);
            end
            
            % Archive Command
            obj.HistoryStack_{end+1} = cmd;
            obj.CurrentIndex_ = numel(obj.HistoryStack_);
            
            % Clip deep memory
            if obj.CurrentIndex_ > obj.MaxHistory_
                obj.HistoryStack_(1) = [];
                obj.CurrentIndex_ = obj.MaxHistory_;
            end
        end
        
        function undo(obj)
            % UNDO  Steps timeline backwards 1 step surgically.
            if obj.CurrentIndex_ > 0
                cmd = obj.HistoryStack_{obj.CurrentIndex_};
                cmd.undo();
                obj.CurrentIndex_ = obj.CurrentIndex_ - 1;
            end
        end
        
        function redo(obj)
            % REDO  Steps timeline forwards 1 step surgically.
            if obj.CurrentIndex_ < numel(obj.HistoryStack_)
                obj.CurrentIndex_ = obj.CurrentIndex_ + 1;
                cmd = obj.HistoryStack_{obj.CurrentIndex_};
                cmd.execute();
            end
        end
        
        function clear(obj)
            % CLEAR  Permanently wipes history stack. 
            %        Used primarily for non-reversible map-wipe actions.
            obj.HistoryStack_ = {};
            obj.CurrentIndex_ = 0;
        end
    end
end
