classdef HuffmanTreeStrategy < OceanViz.Strategies.BaseStrategy
    % HUFFMANTREESTRATEGY  OceanViz rendering strategy for Huffman trees.
    %
    % Renders a HuffmanTree as a rooted binary tree diagram using a
    % Reingold-Tilford style layout. Theme-aware: adapts colors to the
    % Dark or Scientific theme loaded by OceanViz.Core.
    %
    % Registration: 'HuffmanTree' in OceanViz.Core.plot() switch block.
    %
    % Input data struct fields:
    %   data.tree           (HuffmanTree, required)  built Huffman tree
    %   data.showWeights    (logical, default true)   node weight labels
    %   data.showCodewords  (logical, default true)   leaf codeword labels
    %   data.showEdgeLabels (logical, default true)   '0'/'1' on edges
    %   data.title          (string,  default '')     axes title
    %
    % Usage:
    %   coder = OceanMath.Information.Coding.HuffmanCoder();
    %   tree  = coder.buildTree({'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
    %   viz = OceanViz.Core('Theme', 'Dark');
    %   viz.plot(ax, 'HuffmanTree', struct('tree', tree));
    %
    % See also: OceanMath.Information.Coding.HuffmanTree
    %           OceanMath.Information.Coding.HuffmanCoder
    %           OceanViz.Strategies.BaseStrategy

    % ------------------------------------------------------------------ %
    properties (Constant)
        NODE_RADIUS      = 0.30    % radius of node circles
        LEVEL_HEIGHT     = 1.20    % vertical spacing between tree levels
        MIN_H_SPACING    = 0.85    % minimum horizontal spacing between leaves
    end

    % ------------------------------------------------------------------ %
    methods
        function plot(obj, ax, data, options)
            % PLOT  Render the Huffman tree on axes ax.
            %
            % INPUTS:
            %   ax      - MATLAB axes handle
            %   data    - struct with at least data.tree (HuffmanTree)
            %   options - struct (from OceanViz.Core, contains Title etc.)

            % ---- Extract theme colors --------------------------------- %
            colors = obj.getColors_();

            % ---- Extract inputs --------------------------------------- %
            tree = data.tree;
            if ~tree.Built_
                error('OceanViz:HuffmanTreeStrategy:NotBuilt', ...
                    'data.tree must be a built HuffmanTree. Call tree.build() first.');
            end

            showWeights    = obj.getField_(data, 'showWeights',    true);
            showCodewords  = obj.getField_(data, 'showCodewords',  true);
            showEdgeLabels = obj.getField_(data, 'showEdgeLabels', true);
            titleStr       = obj.getField_(data, 'title',          '');
            if isfield(options, 'Title') && ~isempty(char(options.Title))
                titleStr = char(options.Title);
            end

            % ---- Compute layout --------------------------------------- %
            g      = tree.toDigraph();
            rootId = tree.getRootId();
            [xMap, yMap] = obj.layoutTree_(g, rootId);

            % ---- Draw edges ------------------------------------------ %
            hold(ax, 'on');
            nodeIds = 1:g.getNodeCount();
            for nid = nodeIds
                node = g.getNode(nid);
                if isempty(node), continue; end
                children = g.neighbors(nid);
                for ci = 1:numel(children)
                    cid = children(ci);
                    x1 = xMap(nid); y1 = yMap(nid);
                    x2 = xMap(cid); y2 = yMap(cid);
                    % Draw edge line
                    line(ax, [x1, x2], [y1, y2], ...
                        'Color', colors.edge, 'LineWidth', 1.4, ...
                        'HitTest', 'off');
                    % Edge label ('0' or '1')
                    if showEdgeLabels
                        eidx = g.getEdgeId(nid, cid);
                        if eidx > 0
                            lbl = g.getEdge(eidx).Label;
                            mx  = (x1 + x2) / 2;
                            my  = (y1 + y2) / 2 + 0.08;
                            text(ax, mx, my, lbl, ...
                                'Color', colors.edgeLabel, ...
                                'FontSize', 8, 'FontWeight', 'bold', ...
                                'HorizontalAlignment', 'center', ...
                                'VerticalAlignment', 'bottom', ...
                                'Interpreter', 'none');
                        end
                    end
                end
            end

            % ---- Draw nodes ------------------------------------------ %
            for nid = nodeIds
                node = g.getNode(nid);
                if isempty(node), continue; end
                meta = node.Metadata;
                x = xMap(nid); y = yMap(nid);

                if meta.isLeaf
                    nodeColor = colors.leaf;
                    textColor = colors.leafText;
                    shape = 's';   % square marker for leaves
                else
                    nodeColor = colors.internal;
                    textColor = colors.internalText;
                    shape = 'o';   % circle for internal
                end

                % Draw node marker
                r = obj.NODE_RADIUS;
                if strcmp(shape, 'o')
                    th = linspace(0, 2*pi, 48);
                    fill(ax, x + r*cos(th), y + r*sin(th), nodeColor, ...
                        'EdgeColor', colors.nodeBorder, 'LineWidth', 1.2, ...
                        'HitTest', 'off');
                else
                    % Square for leaf
                    fill(ax, x + r*[-1 1 1 -1], y + r*[-1 -1 1 1], nodeColor, ...
                        'EdgeColor', colors.nodeBorder, 'LineWidth', 1.4, ...
                        'HitTest', 'off');
                end

                % Weight label inside node
                if showWeights && isfield(meta, 'weight')
                    text(ax, x, y + 0.03, sprintf('%.3g', meta.weight), ...
                        'Color', textColor, 'FontSize', 7.5, ...
                        'FontWeight', 'bold', ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', ...
                        'Interpreter', 'none');
                end

                % Symbol + codeword below leaf nodes
                if meta.isLeaf
                    sym = meta.symbol;
                    if ~ischar(sym), sym = num2str(sym); end
                    % Read codeword from CodeTable (node metadata .codeword is
                    % not back-filled; cwMap only updates the Map, not the graph)
                    cwStr = '';
                    if showCodewords
                        ct = tree.getCodeTable();
                        symsAll = ct.getSymbols();
                        cwAll   = ct.getCodewords();
                        for si = 1:numel(symsAll)
                            s2 = symsAll{si};
                            if ~ischar(s2), s2 = char(string(s2)); end
                            if strcmp(s2, sym)
                                cwStr = cwAll{si};
                                break;
                            end
                        end
                    end
                    if ~isempty(cwStr)
                        labelStr = sprintf('%s [%s]', sym, cwStr);
                    else
                        labelStr = sym;
                    end
                    text(ax, x, y - r - 0.10, labelStr, ...
                        'Color', colors.symbolLabel, 'FontSize', 8, ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'top', ...
                        'Interpreter', 'none');
                end
            end

            % ---- Axes styling ----------------------------------------- %
            ax.XColor = 'none';
            ax.YColor = 'none';
            ax.Box    = 'off';
            axis(ax, 'equal');
            axis(ax, 'tight');
            % Add 10% padding
            xl = ax.XLim; yl = ax.YLim;
            xpad = max(0.5, diff(xl)*0.08);
            ypad = max(0.5, diff(yl)*0.15);
            ax.XLim = [xl(1)-xpad, xl(2)+xpad];
            ax.YLim = [yl(1)-ypad, yl(2)+ypad];

            if ~isempty(titleStr)
                title(ax, titleStr, 'Color', colors.title, ...
                    'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
            end

            hold(ax, 'off');
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)

        function [xMap, yMap] = layoutTree_(obj, g, rootId)
            % LAYOUTTREE_  Compute (x,y) per node — fully iterative, no recursion.
            %
            % Algorithm:
            %   1. BFS from root: get depth of each node + build children list
            %   2. Iterative DFS (explicit stack) in LEFT-to-RIGHT order
            %      to number leaves sequentially
            %   3. Internal node x = mean of children x (bottom-up pass)
            %
            % All work is done in this one method to avoid MATLAB's
            % pass-by-value issue that would silently discard array writes
            % inside any helper function.

            N = g.getNodeCount();
            xMap = zeros(1, N);
            yMap = zeros(1, N);

            % ---- Step 1: BFS for depths + children lists ---------------- %
            depth      = -ones(1, N);
            depth(rootId) = 0;
            childrenOf = cell(1, N);
            for i = 1:N, childrenOf{i} = []; end

            bfsQueue = rootId;
            while ~isempty(bfsQueue)
                nid = bfsQueue(1);  bfsQueue(1) = [];
                ch  = g.neighbors(nid);   % out-neighbors (children)
                childrenOf{nid} = ch;
                for ci = 1:numel(ch)
                    cid = ch(ci);
                    if depth(cid) < 0
                        depth(cid) = depth(nid) + 1;
                        bfsQueue(end+1) = cid; %#ok<AGROW>
                    end
                end
            end

            % Y = -depth * spacing (root at top)
            for nid = 1:N
                if depth(nid) >= 0
                    yMap(nid) = -depth(nid) * obj.LEVEL_HEIGHT;
                end
            end

            % ---- Step 2: Iterative DFS — number leaves L→R ------------- %
            % Uses an explicit stack; processes children in the order
            % neighbors() returns them (ascending node-ID = label order,
            % since we add '0'-child before '1'-child in build()).
            leafCounter = 0;
            dfsStack    = rootId;
            while ~isempty(dfsStack)
                nid = dfsStack(end);  dfsStack(end) = [];
                node = g.getNode(nid);
                if isempty(node), continue; end
                if node.Metadata.isLeaf
                    leafCounter   = leafCounter + 1;
                    xMap(nid)     = leafCounter * obj.MIN_H_SPACING;
                else
                    % Push children in REVERSE order so leftmost is popped first
                    ch  = childrenOf{nid};
                    for ci = numel(ch):-1:1
                        dfsStack(end+1) = ch(ci); %#ok<AGROW>
                    end
                end
            end

            % ---- Step 3: Internal node x = mean of children x ----------- %
            % Process from deepest level up to root (bottom-up)
            validDepths = depth(depth >= 0);
            if isempty(validDepths), return; end
            maxDepth = max(validDepths);

            for d = maxDepth:-1:0
                for nid = 1:N
                    if depth(nid) ~= d, continue; end
                    node = g.getNode(nid);
                    if isempty(node) || node.Metadata.isLeaf, continue; end
                    ch = childrenOf{nid};
                    if isempty(ch), continue; end
                    xMap(nid) = mean(xMap(ch));
                end
            end
        end

        function colors = getColors_(obj)
            if isempty(obj.Parent)
                colors.edge         = [0.45, 0.55, 0.70];
                colors.edgeLabel    = [0.85, 0.85, 0.90];
                colors.internal     = [0.22, 0.45, 0.75];
                colors.internalText = [1.00, 1.00, 1.00];
                colors.leaf         = [0.92, 0.65, 0.20];
                colors.leafText     = [0.10, 0.10, 0.10];
                colors.nodeBorder   = [0.80, 0.80, 0.90];
                colors.symbolLabel  = [0.90, 0.90, 0.90];
                colors.title        = [0.95, 0.95, 1.00];
            else
                C = obj.Parent.Colors;
                colors.edge         = C.Grid;
                colors.edgeLabel    = C.Text;
                colors.internal     = C.Primary;
                colors.internalText = C.Background;
                colors.leaf         = C.Boundary;
                colors.leafText     = C.Background;
                colors.nodeBorder   = C.Text .* 0.6;
                colors.symbolLabel  = C.Text;
                colors.title        = C.Text;
            end
        end

        function v = getField_(~, s, field, default)
            if isfield(s, field)
                v = s.(field);
            else
                v = default;
            end
        end
    end
end

