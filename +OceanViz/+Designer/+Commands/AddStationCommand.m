classdef AddStationCommand < OceanViz.Designer.Commands.ICommand
    % ADDSTATIONCOMMAND  Delta object for placing a new station.
    % 
    %   State tracked:
    %   - Name
    %   - Position
    %   - StationId_ (assigned upon forward execution)
    
    properties (SetAccess = private)
        Designer_
        Name_
        Position_
        StationId_ = []
    end
    
    methods
        function obj = AddStationCommand(designer, name, position)
            obj.Designer_ = designer;
            obj.Name_ = name;
            obj.Position_ = position;
        end
        
        function execute(obj)
            % Forward Action
            obj.StationId_ = obj.Designer_.addStation(obj.Name_, obj.Position_);
        end
        
        function undo(obj)
            % Reverse Action
            if ~isempty(obj.StationId_)
                obj.Designer_.deleteStation_(obj.StationId_, true);
                % Once undone, the mathematical ID is lost forever.
                % Redoing implies calling addStation again which gets a new ID.
                obj.StationId_ = [];
            end
        end
        
        function name = getName(obj)
            name = sprintf('Add Station: %s', obj.Name_);
        end
    end
end
