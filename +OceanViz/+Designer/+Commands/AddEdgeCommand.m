classdef AddEdgeCommand < OceanViz.Designer.Commands.ICommand
    % ADDEDGECOMMAND  Delta object for routing a new curve between stations.
    
    properties (SetAccess = private)
        Designer_
        SourceId_
        TargetId_
        Spline_
        Options_
        EdgeIdx_ = []
    end
    
    methods
        function obj = AddEdgeCommand(designer, sourceId, targetId, spline, options)
            obj.Designer_ = designer;
            obj.SourceId_ = sourceId;
            obj.TargetId_ = targetId;
            obj.Spline_ = spline;
            obj.Options_ = options;
        end
        
        function execute(obj)
            % Forward Action
            obj.Designer_.Transit_.addRoute(obj.SourceId_, obj.TargetId_, obj.Spline_);
            obj.Designer_.EdgeCount_ = obj.Designer_.EdgeCount_ + 1;
            obj.EdgeIdx_ = obj.Designer_.EdgeCount_;
            
            if ~isempty(obj.Designer_.Renderer_)
                obj.Designer_.Renderer_.drawCurve(obj.EdgeIdx_, obj.Spline_, 'Color', obj.Options_.Color);
                
                % Provide interactive handlers
                [grabCb, moveCb, relCb] = obj.Designer_.InteractionHandler_.makeCallbacks_();
                obj.Designer_.Renderer_.drawControlPoints(obj.EdgeIdx_, obj.Spline_, grabCb, moveCb, relCb);
            end
        end
        
        function undo(obj)
            % Reverse Action
            if ~isempty(obj.EdgeIdx_)
                obj.Designer_.Transit_.removeRoute(obj.SourceId_, obj.TargetId_);
                if ~isempty(obj.Designer_.Renderer_)
                    obj.Designer_.Renderer_.deleteCurve(obj.EdgeIdx_);
                end
                
                % Re-sync logic: If this was the last edge added, retract the counter
                if obj.Designer_.EdgeCount_ == obj.EdgeIdx_
                    obj.Designer_.EdgeCount_ = obj.Designer_.EdgeCount_ - 1;
                end
                obj.EdgeIdx_ = [];
            end
        end
        
        function name = getName(obj)
            name = sprintf('Add Edge (%d -> %d)', obj.SourceId_, obj.TargetId_);
        end
    end
end
