classdef WorkspaceManager < handle
    % WORKSPACEMANAGER  Controller for file I/O operations and workspace lifecycles.
    
    properties (SetAccess = private)
        Designer_
    end
    
    methods
        function obj = WorkspaceManager(designer)
            obj.Designer_ = designer;
        end
        
        function onExport(obj)
            choice = questdlg('Export format:', 'Export', ...
                'Workspace', 'JSON File', 'Cancel', 'JSON File');
            switch choice
                case 'Workspace'
                    g = obj.Designer_.exportGraph();
                    assignin('base', 'trackGraph', g); %Assign g to var name trackGraph
                    g.describe();
                    msgbox('Graph exported to workspace as "trackGraph"', 'Export');
                case 'JSON File'
                    obj.exportJSON();
            end
        end
        
        function exportJSON(obj)
            td = obj.Designer_;
            [file, path] = uiputfile('*.json', 'Export Track as JSON');
            if file == 0, return; end
            filepath = fullfile(path, file);
            
            overlayState = struct( ...
                'legendPos1', td.OverlayRenderer_.LegendPos1_, ...
                'legendPos2', td.OverlayRenderer_.LegendPos2_, ...
                'titlePos',   td.OverlayRenderer_.TitlePos_, ...
                'mapTitle',   {td.OverlayRenderer_.MapTitle_});
            
            OceanViz.Designer.Persistence.DesignerSerializer.exportJSON( ...
                filepath, td.Transit_, td.Renderer_, ...
                td.StationCount_, td.EdgeCount_, overlayState);
        end
        
        function importJSON(obj)
            td = obj.Designer_;
            [file, path] = uigetfile('*.json', 'Import Track from JSON');
            if file == 0, return; end
            filepath = fullfile(path, file);
            OceanViz.Designer.Persistence.DesignerSerializer.importJSON(filepath, td);
        end
        
        function onSave(obj)
            td = obj.Designer_;
            [file, path] = uiputfile('*.mat', 'Save Track Design');
            if file == 0, return; end
            td.saveDesign(fullfile(path, file));
        end
        
        function onLoad(obj)
            td = obj.Designer_;
            choice = questdlg('Load format:', 'Load Design', ...
                'Designer File (.mat)', 'JSON File (.json)', 'Cancel', 'Designer File (.mat)');
            switch choice
                case 'Designer File (.mat)'
                    [file, path] = uigetfile('*.mat', 'Load Track Design');
                    if file ~= 0
                        td.loadDesign(fullfile(path, file));
                    end
                case 'JSON File (.json)'
                    obj.importJSON();
            end
        end
        
        function onNew(obj)
            td = obj.Designer_;
            choice = questdlg('Clear entire workspace? Unsaved changes will be lost.', ...
                'Confirm New', 'Yes', 'Cancel', 'Cancel');
            if strcmp(choice, 'Yes')
                td.clearUndo_();
                td.Transit_ = OceanMath.Transit.TransitNetwork();
                td.StationCount_ = 0;
                td.EdgeCount_ = 0;
                td.WorkingPos_ = [0 0 0];
                td.SelectedStation_ = 0;
                td.ActiveLineCode_ = '';
                td.ActiveEditCurve_ = 0;
                if ~isempty(td.Renderer_)
                    td.Renderer_.clearAll();
                end
                td.OverlayRenderer_.clearWeightLabels_();
                
                % Clear legend
                for k = 1:numel(td.OverlayRenderer_.LegendHandles_)
                    if isvalid(td.OverlayRenderer_.LegendHandles_{k}), delete(td.OverlayRenderer_.LegendHandles_{k}); end
                end
                td.OverlayRenderer_.LegendHandles_ = {};
                td.OverlayRenderer_.LegendPos1_ = [NaN NaN];
                td.OverlayRenderer_.LegendPos2_ = [NaN NaN];
                td.OverlayRenderer_.TitlePos_ = [NaN NaN];
                if ~isempty(td.OverlayRenderer_.TitleHandle_) && isvalid(td.OverlayRenderer_.TitleHandle_)
                    delete(td.OverlayRenderer_.TitleHandle_);
                end
                td.OverlayRenderer_.ShowWeights_ = false;
                if isfield(td.ToolButtons_, 'weights')
                    set(td.ToolButtons_.weights, 'Value', 0, 'BackgroundColor', [0.2 0.2 0.25]);
                end
                td.updateStatus_();
                td.refreshLineDropdown_();
                set(td.LinePanel_.colorIndicator, ...
                    'BackgroundColor', [0.15, 0.15, 0.18], 'String', '');
                set(td.PropPanel_.drawOrderLabel, 'String', 'Active Curve: (none)');
                set(td.PropPanel_.selectedLabel, 'String', 'No station selected');
            end
        end
    end
end
