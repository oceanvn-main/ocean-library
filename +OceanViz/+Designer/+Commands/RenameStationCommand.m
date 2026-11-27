classdef RenameStationCommand < OceanViz.Designer.Commands.ICommand
    % RENAMESTATIONCOMMAND  Delta object for renaming a station safely using the Undo Pipeline.
    
    properties (SetAccess = private)
        Designer_
        StationId_
        OldName_
        NewName_
    end
    
    methods
        function obj = RenameStationCommand(designer, stationId, oldName, newName)
            obj.Designer_ = designer;
            obj.StationId_ = stationId;
            obj.OldName_ = oldName;
            obj.NewName_ = newName;
        end
        
        function execute(obj)
            % Forward Action: Change the text to the New Name.
            obj.Designer_.Transit_.updateStationName(obj.StationId_, obj.NewName_);
            if ~isempty(obj.Designer_.Renderer_)
                obj.Designer_.Renderer_.updateLabelText(obj.StationId_, obj.NewName_);
            end
            
            % Update UI if the user still has it selected
            if obj.Designer_.SelectedStation_ == obj.StationId_
                set(obj.Designer_.PropPanel_.selectedLabel, 'String', ...
                    sprintf('Selected: %s (ID %d)', obj.NewName_, obj.StationId_));
                set(obj.Designer_.PropPanel_.nameField, 'String', obj.NewName_);
            end
        end
        
        function undo(obj)
            % Reverse Action: Roll back the text to the Old Name!
            obj.Designer_.Transit_.updateStationName(obj.StationId_, obj.OldName_);
            if ~isempty(obj.Designer_.Renderer_)
                obj.Designer_.Renderer_.updateLabelText(obj.StationId_, obj.OldName_);
            end
            
            % Update UI if the user still has it selected
            if obj.Designer_.SelectedStation_ == obj.StationId_
                set(obj.Designer_.PropPanel_.selectedLabel, 'String', ...
                    sprintf('Selected: %s (ID %d)', obj.OldName_, obj.StationId_));
                set(obj.Designer_.PropPanel_.nameField, 'String', obj.OldName_);
            end
        end
        
        function name = getName(obj)
            name = sprintf('Rename Station %d to %s', obj.StationId_, obj.NewName_);
        end
    end
end
