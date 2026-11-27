classdef SetWeightCommand < OceanDL.Commands.INodeCommand
    % SETWEIGHTCOMMAND  Command to set a single NNEdge weight.
    %
    %   OldWeight_ is captured during execute() (not at construction) so
    %   that stacking multiple SetWeightCommand on the same edge works
    %   correctly: each one remembers the value IT replaced.
    %
    % See also: OceanDL.Commands.INodeCommand
    %           OceanDL.Commands.SetBiasCommand

    properties (Access = private)
        FromID_    (1,1) string   % source node ID
        ToID_      (1,1) string   % destination node ID
        NewWeight_ (1,1) double   % desired weight after execute()
        OldWeight_ (1,1) double   % previous weight, captured in execute()
    end

    methods
        function obj = SetWeightCommand(fromID, toID, newWeight)
            % SETWEIGHTCOMMAND  Construct for the edge fromID → toID.
            %
            % INPUTS:
            %   fromID    - (1,1) string, source node ID
            %   toID      - (1,1) string, destination node ID
            %   newWeight - (1,1) double, weight value to assign
            arguments
                fromID    (1,1) string
                toID      (1,1) string
                newWeight (1,1) double
            end
            obj.FromID_    = fromID;
            obj.ToID_      = toID;
            obj.NewWeight_ = newWeight;
            obj.OldWeight_ = NaN;    % filled in during execute()
        end

        function execute(obj, graph)
            edge           = graph.getEdge_(obj.FromID_, obj.ToID_);
            obj.OldWeight_ = edge.Weight_;          % capture before mutating
            edge.Weight_   = obj.NewWeight_;
        end

        function undo(obj, graph)
            edge         = graph.getEdge_(obj.FromID_, obj.ToID_);
            edge.Weight_ = obj.OldWeight_;
        end

        function s = describe(obj)
            s = sprintf("Set weight %s → %s :  %.4f → %.4f", ...
                obj.FromID_, obj.ToID_, obj.OldWeight_, obj.NewWeight_);
        end
    end
end
