classdef NNEdge < handle
    % NNEDGE  A directed, weighted connection between two NNNode objects.
    %
    %   OWNS:
    %     - One scalar weight parameter (Weight_)
    %     - References to the source (FromNode) and destination (ToNode) NNNode
    %
    %   ACTIVE SEMANTICS:
    %     An edge is only active when BOTH its endpoints are enabled.
    %     isActive() is derived — it queries the nodes directly. Nothing is cached.
    %     This means: disabling a node automatically deactivates all its edges.
    %
    %   VISUAL ENCODING (used by NetworkGraphStrategy in Phase 3):
    %     Edge thickness  ∝  |Weight_|     (magnitude)
    %     Edge color      =  blue if Weight_ > 0, red if Weight_ < 0
    %
    %   See also: OceanDL.Graph.NNNode
    %             OceanDL.Graph.NNGraph

    % ------------------------------------------------------------------ %
    properties (Access = public)
        FromNode    % NNNode — source neuron  (set at construction, not typed to avoid circular ref)
        ToNode      % NNNode — destination neuron
        Weight_  (1,1) double = 0   % learnable weight w_{ij}
    end

    % ================================================================== %
    methods
        function obj = NNEdge(fromNode, toNode, weight)
            % NNEDGE  Construct a directed edge between two nodes.
            %
            %   edge = NNEdge(fromNode, toNode)           % weight = 0
            %   edge = NNEdge(fromNode, toNode, weight)
            arguments
                fromNode   % NNNode
                toNode     % NNNode
                weight  (1,1) double = 0
            end
            obj.FromNode = fromNode;
            obj.ToNode   = toNode;
            obj.Weight_  = weight;
        end

        % ---- State query -------------------------------------------- %

        function tf = isActive(obj)
            % ISACTIVE  True when both endpoints are enabled.
            %   This is O(1) — just reads two state flags.
            tf = obj.FromNode.isEnabled() && obj.ToNode.isEnabled();
        end

        % ---- Serialization ------------------------------------------ %

        function s = toStruct(obj)
            % TOSTRUCT  Export to a plain struct for save/load.
            s.fromID  = obj.FromNode.ID;
            s.toID    = obj.ToNode.ID;
            s.weight  = obj.Weight_;
            s.active  = obj.isActive();
        end

        % ---- Display ------------------------------------------------- %

        function disp(obj)
            fprintf('NNEdge  %s → %s  w=%.4f  active=%d\n', ...
                obj.FromNode.ID, obj.ToNode.ID, obj.Weight_, obj.isActive());
        end
    end
end
