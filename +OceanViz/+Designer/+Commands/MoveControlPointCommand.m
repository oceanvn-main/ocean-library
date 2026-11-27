classdef MoveControlPointCommand < OceanViz.Designer.Commands.ICommand
    % MOVECONTROLPOINTCOMMAND  Delta object for adjusting a spline curve point.
    
    properties (SetAccess = private)
        Designer_
        EdgeIdx_
        CpIdx_
        StartPos_
        EndPos_
    end
    
    methods
        function obj = MoveControlPointCommand(designer, edgeIdx, cpIdx, startPos, endPos)
            obj.Designer_ = designer;
            obj.EdgeIdx_ = edgeIdx;
            obj.CpIdx_ = cpIdx;
            obj.StartPos_ = startPos;
            obj.EndPos_ = endPos;
        end
        
        function execute(obj)
            % Forward Action
            spline = obj.Designer_.Transit_.getSpline(obj.EdgeIdx_);
            spline.setControlPoint(obj.CpIdx_, obj.EndPos_);
            obj.Designer_.Transit_.updateRouteSpline(obj.EdgeIdx_, spline); %Update the network also
            if ~isempty(obj.Designer_.Renderer_)
                obj.Designer_.Renderer_.updateCurve(obj.EdgeIdx_, spline, 'Samples', 200);
            end
        end
        
        function undo(obj)
            % Reverse Action
            spline = obj.Designer_.Transit_.getSpline(obj.EdgeIdx_);
            spline.setControlPoint(obj.CpIdx_, obj.StartPos_);
            obj.Designer_.Transit_.updateRouteSpline(obj.EdgeIdx_, spline);
            if ~isempty(obj.Designer_.Renderer_)
                obj.Designer_.Renderer_.updateCurve(obj.EdgeIdx_, spline, 'Samples', 200);
            end
        end
        
        function name = getName(obj)
            name = sprintf('Move Point (Edge %d, Cp %d)', obj.EdgeIdx_, obj.CpIdx_);
        end
    end
end
