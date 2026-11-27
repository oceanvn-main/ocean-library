classdef GraphStyle < handle
    % GRAPHSTYLE  Mutable visual attribute container for graph rendering.
    %
    % Stores per-node and per-edge visual properties (color, size, icon,
    % dash pattern) plus path highlights and text annotations.
    %
    % The renderer checks GraphStyle first, then falls back to theme
    % defaults for any unset property.
    %
    % Usage:
    %   style = OceanViz.Adapters.GraphStyle(numNodes, numEdges);
    %   style.setNodeColor(1, [1 0 0]);
    %   style.setNodeIcon(2, 'interchange');
    %   style.setLineColor('NSL', [0.8 0.1 0.1], edgeLabels);
    %   style.highlightPath([1 2 3], [0 1 0.5], 4);
    %
    % See also: OceanViz.Strategies.TransitGraphStrategy

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        NumNodes_  (1,1) double = 0
        NumEdges_  (1,1) double = 0

        % Per-node visuals (NaN = "use default")
        NodeColors_     % [V x 3] RGB or NaN
        NodeSizes_      % [V x 1] marker size or NaN
        NodeIcons_      % {V x 1} cell of char: 'circle','square','diamond','interchange'
        NodeLabelShow_  % [V x 1] logical

        % Per-edge visuals
        EdgeColors_     % [E x 3] RGB or NaN
        EdgeWidths_     % [E x 1] line width or NaN
        EdgeDashes_     % {E x 1} cell of char: '-','--',':','-.'

        % Highlighted paths
        Paths_          % cell array of structs: {nodeSeq, color, width}

        % Text annotations
        Annotations_    % cell array of structs: {position, text, color, fontSize}
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = GraphStyle(numNodes, numEdges)
            % GRAPHSTYLE  Construct for a graph with V nodes and E edges.
            arguments
                numNodes (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                numEdges (1,1) double {mustBeNonnegative, mustBeInteger} = 0
            end
            obj.NumNodes_ = numNodes;
            obj.NumEdges_ = numEdges;

            % Initialize all to NaN (= "use default")
            obj.NodeColors_    = nan(numNodes, 3);
            obj.NodeSizes_     = nan(numNodes, 1);
            obj.NodeIcons_     = repmat({''}, numNodes, 1);
            obj.NodeLabelShow_ = true(numNodes, 1);

            obj.EdgeColors_    = nan(numEdges, 3);
            obj.EdgeWidths_    = nan(numEdges, 1);
            obj.EdgeDashes_    = repmat({'-'}, numEdges, 1);

            obj.Paths_       = {};
            obj.Annotations_ = {};
        end

        % ============================================================== %
        %  Node Styling
        % ============================================================== %

        function setNodeColor(obj, nodeId, rgb)
            % SETNODECOLOR  Set the fill color of a node.
            arguments
                obj
                nodeId (1,1) double {mustBePositive, mustBeInteger}
                rgb    (1,3) double
            end
            obj.NodeColors_(nodeId, :) = rgb;
        end

        function setNodeSize(obj, nodeId, sz)
            % SETNODESIZE  Set the marker size of a node.
            arguments
                obj
                nodeId (1,1) double {mustBePositive, mustBeInteger}
                sz     (1,1) double {mustBePositive}
            end
            obj.NodeSizes_(nodeId) = sz;
        end

        function setNodeIcon(obj, nodeId, icon)
            % SETNODEICON  Set the marker shape of a node.
            %   Options: 'circle', 'square', 'diamond', 'interchange'
            arguments
                obj
                nodeId (1,1) double {mustBePositive, mustBeInteger}
                icon   (1,:) char
            end
            obj.NodeIcons_{nodeId} = icon;
        end

        function setNodeLabelVisible(obj, nodeId, tf)
            % SETNODELABELVISIBLE  Show/hide the station name label.
            arguments
                obj
                nodeId (1,1) double {mustBePositive, mustBeInteger}
                tf     (1,1) logical
            end
            obj.NodeLabelShow_(nodeId) = tf;
        end

        function rgb = getNodeColor(obj, nodeId)
            rgb = obj.NodeColors_(nodeId, :);
        end

        function sz = getNodeSize(obj, nodeId)
            sz = obj.NodeSizes_(nodeId);
        end

        function icon = getNodeIcon(obj, nodeId)
            icon = obj.NodeIcons_{nodeId};
        end

        function tf = getNodeLabelVisible(obj, nodeId)
            tf = obj.NodeLabelShow_(nodeId);
        end

        % ============================================================== %
        %  Edge Styling
        % ============================================================== %

        function setEdgeColor(obj, edgeIdx, rgb)
            % SETEDGECOLOR  Set the line color of an edge.
            arguments
                obj
                edgeIdx (1,1) double {mustBePositive, mustBeInteger}
                rgb     (1,3) double
            end
            obj.EdgeColors_(edgeIdx, :) = rgb;
        end

        function setEdgeWidth(obj, edgeIdx, w)
            % SETEDGEWIDTH  Set the line width of an edge.
            arguments
                obj
                edgeIdx (1,1) double {mustBePositive, mustBeInteger}
                w       (1,1) double {mustBePositive}
            end
            obj.EdgeWidths_(edgeIdx) = w;
        end

        function setEdgeDash(obj, edgeIdx, dash)
            % SETEDGEDASH  Set the line dash pattern of an edge.
            %   Options: '-', '--', ':', '-.'
            arguments
                obj
                edgeIdx (1,1) double {mustBePositive, mustBeInteger}
                dash    (1,:) char
            end
            obj.EdgeDashes_{edgeIdx} = dash;
        end

        function rgb = getEdgeColor(obj, edgeIdx)
            rgb = obj.EdgeColors_(edgeIdx, :);
        end

        function w = getEdgeWidth(obj, edgeIdx)
            w = obj.EdgeWidths_(edgeIdx);
        end

        function dash = getEdgeDash(obj, edgeIdx)
            dash = obj.EdgeDashes_{edgeIdx};
        end

        % ============================================================== %
        %  Line Group Coloring (by edge label)
        % ============================================================== %

        function setLineColor(obj, label, rgb, edgeLabels)
            % SETLINECOLOR  Color all edges with a matching label.
            %
            %   style.setLineColor('NSL', [0.8 0.1 0.1], edgeLabels)
            %
            %   edgeLabels is a {1 x E} cell array of edge label strings.
            arguments
                obj
                label      (1,:) char
                rgb        (1,3) double
                edgeLabels (1,:) cell
            end
            for i = 1:numel(edgeLabels)
                if strcmp(edgeLabels{i}, label)
                    obj.EdgeColors_(i, :) = rgb;
                end
            end
        end

        % ============================================================== %
        %  Path Highlighting
        % ============================================================== %

        %Store records of paths to be highlighted
        function highlightPath(obj, nodeSequence, color, width)
            % HIGHLIGHTPATH  Mark a path for overlay rendering.
            %
            %   style.highlightPath([1, 2, 3], [0 1 0.5], 4)
            arguments
                obj
                nodeSequence (1,:) double {mustBePositive, mustBeInteger}
                color        (1,3) double = [0.2, 1.0, 0.6]
                width        (1,1) double {mustBePositive} = 4
            end
            p.NodeSequence = nodeSequence;
            p.Color = color;
            p.Width = width;
            obj.Paths_{end+1} = p;
        end

        function clearHighlights(obj)
            % CLEARHIGHLIGHTS  Remove all path highlights.
            obj.Paths_ = {};
        end

        function paths = getPaths(obj)
            paths = obj.Paths_;
        end

        % ============================================================== %
        %  Text Annotations
        % ============================================================== %

        function addAnnotation(obj, position, txt, options)
            % ADDANNOTATION  Place free text on the map.
            %
            %   style.addAnnotation([5, 8], 'North-South Line', 'Color', [1 0 0])
            arguments
                obj
                position (1,2) double
                txt      (1,:) char
                options.Color    (1,3) double = [0.9, 0.9, 0.9]
                options.FontSize (1,1) double {mustBePositive} = 11
            end
            a.Position = position;
            a.Text     = txt;
            a.Color    = options.Color;
            a.FontSize = options.FontSize;
            obj.Annotations_{end+1} = a;
        end

        function clearAnnotations(obj)
            obj.Annotations_ = {};
        end

        function anns = getAnnotations(obj)
            anns = obj.Annotations_;
        end

        % ============================================================== %
        %  Utility
        % ============================================================== %

        function tf = hasNodeColor(obj, nodeId)
            tf = ~any(isnan(obj.NodeColors_(nodeId, :)));
        end

        function tf = hasNodeSize(obj, nodeId)
            tf = ~isnan(obj.NodeSizes_(nodeId));
        end

        function tf = hasNodeIcon(obj, nodeId)
            tf = ~isempty(obj.NodeIcons_{nodeId});
        end

        function tf = hasEdgeColor(obj, edgeIdx)
            tf = ~any(isnan(obj.EdgeColors_(edgeIdx, :)));
        end

        function tf = hasEdgeWidth(obj, edgeIdx)
            tf = ~isnan(obj.EdgeWidths_(edgeIdx));
        end
    end
end
