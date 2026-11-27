classdef CodeTreeRenderer < handle
    % CODETREERENDERER  Premium prefix code tree visualization strategy.
    %
    % Renders source coding trees (Huffman, Shannon-Fano) using a top-down layout:
    %   - Level 0 (Root) is at (0, 1)
    %   - Leaves are evenly spaced at y = 0
    %   - Internal nodes sit at average parent-child centers.
    % Supports interactive path highlights in glowing blue/cyan.
    % Bypasses native LaTeX to avoid rendering failures on UIAxes.
    
    properties
        AxesHandle
        CoreObj
        
        Nodes       % Cell array of node structs
        
        % Graphic handles for dynamic updates / highlighting
        EdgeLines   % Array of line handles for parent-child branches
        EdgeLabels  % Array of text handles for branch labels ('0'/'1')
        NodeCircles % Array of patch/fill handles for node markers
        NodeTexts   % Cell array of text handles for leaf/internal labels
    end
    
    methods
        function obj = CodeTreeRenderer(axesHandle, coreObj)
            arguments
                axesHandle
                coreObj = []
            end
            
            obj.AxesHandle = axesHandle;
            
            if isempty(coreObj)
                try
                    obj.CoreObj = OceanViz.Core('Theme', 'Dark');
                catch
                    % Fallback to standard Neon/Dark theme if Core is unavailable
                    obj.CoreObj = struct();
                    obj.CoreObj.ThemeName = "Dark";
                    obj.CoreObj.Colors = struct(...
                        'Background', [0.08, 0.08, 0.10], ...
                        'Text', [0.95, 0.95, 0.95], ...
                        'Grid', [0.20, 0.22, 0.28], ...
                        'Primary', [0.0, 0.8, 1.0], ...
                        'Secondary', [1.0, 0.3, 0.5], ...
                        'Boundary', [1.0, 0.8, 0.2] ...
                    );
                end
            else
                obj.CoreObj = coreObj;
            end
            
            obj.styleAxes();
        end
        
        function styleAxes(obj)
            ax = obj.AxesHandle;
            bg = obj.CoreObj.Colors.Background;
            
            set(ax, 'Color', bg, ...
                    'XColor', 'none', ...
                    'YColor', 'none', ...
                    'Box', 'off');
            axis(ax, 'off');
        end
        
        function renderTree(obj, codeTable)
            % Rebuilds and draws the tree graphic from a CodeTable object.
            arguments
                obj
                codeTable
            end
            
            ax = obj.AxesHandle;
            hold(ax, 'on');
            
            % 1. Teardown existing handles cleanly
            obj.clearGraphics();
            
            % 2. Parse CodeTable and Build Tree Structure
            syms = codeTable.getSymbols();
            probs = codeTable.getProbabilities();
            codewords = codeTable.getCodewords();
            K = numel(syms);
            
            % Root node
            obj.Nodes = {struct(...
                'id', 1, ...
                'codeword', '', ...
                'isLeaf', false, ...
                'symbol', '', ...
                'prob', 0, ...
                'leftChild', 0, ...
                'rightChild', 0, ...
                'parent', 0, ...
                'x', 0, ...
                'y', 1.0, ...
                'depth', 0 ...
            )};
            
            % Dynamically construct binary tree matching the prefix codes
            for i = 1:K
                cw = codewords{i};
                p = probs(i);
                sym = syms{i};
                if ~ischar(sym), sym = char(string(sym)); end
                
                currId = 1;
                for ch = 1:length(cw)
                    bit = cw(ch);
                    if bit == '0'
                        if obj.Nodes{currId}.leftChild == 0
                            newId = numel(obj.Nodes) + 1;
                            obj.Nodes{newId} = struct(...
                                'id', newId, ...
                                'codeword', [obj.Nodes{currId}.codeword, '0'], ...
                                'isLeaf', false, ...
                                'symbol', '', ...
                                'prob', 0, ...
                                'leftChild', 0, ...
                                'rightChild', 0, ...
                                'parent', currId, ...
                                'x', 0, ...
                                'y', 0, ...
                                'depth', ch ...
                            );
                            obj.Nodes{currId}.leftChild = newId;
                        end
                        currId = obj.Nodes{currId}.leftChild;
                    elseif bit == '1'
                        if obj.Nodes{currId}.rightChild == 0
                            newId = numel(obj.Nodes) + 1;
                            obj.Nodes{newId} = struct(...
                                'id', newId, ...
                                'codeword', [obj.Nodes{currId}.codeword, '1'], ...
                                'isLeaf', false, ...
                                'symbol', '', ...
                                'prob', 0, ...
                                'leftChild', 0, ...
                                'rightChild', 0, ...
                                'parent', currId, ...
                                'x', 0, ...
                                'y', 0, ...
                                'depth', ch ...
                            );
                            obj.Nodes{currId}.rightChild = newId;
                        end
                        currId = obj.Nodes{currId}.rightChild;
                    end
                end
                
                % Associate leaf metadata
                obj.Nodes{currId}.isLeaf = true;
                obj.Nodes{currId}.symbol = sym;
                obj.Nodes{currId}.prob = p;
            end
            
            N = numel(obj.Nodes);
            
            % 3. Layout: Leaves evenly spaced at y = 0
            leafIndices = find(cellfun(@(n) n.isLeaf, obj.Nodes));
            leafCodewords = cellfun(@(n) n.codeword, obj.Nodes(leafIndices), 'UniformOutput', false);
            [~, sortIdx] = sort(leafCodewords);
            sortedLeafIndices = leafIndices(sortIdx);
            
            M = numel(sortedLeafIndices);
            for j = 1:M
                idx = sortedLeafIndices(j);
                if M > 1
                    obj.Nodes{idx}.x = -1.5 + 3.0 * (j - 1) / (M - 1);
                else
                    obj.Nodes{idx}.x = 0.0;
                end
                obj.Nodes{idx}.y = 0.0;
            end
            
            % 4. Layout: Internal nodes sit at average parent-child centers
            maxDepth = max(cellfun(@(n) n.depth, obj.Nodes));
            
            % Bottom-up pass to calculate coordinates
            for d = maxDepth-1 : -1 : 0
                depthIndices = find(cellfun(@(n) n.depth == d, obj.Nodes));
                for idx = depthIndices
                    if obj.Nodes{idx}.isLeaf, continue; end
                    lc = obj.Nodes{idx}.leftChild;
                    rc = obj.Nodes{idx}.rightChild;
                    
                    xs = []; ys = [];
                    if lc > 0
                        xs = [xs, obj.Nodes{lc}.x];
                        ys = [ys, obj.Nodes{lc}.y];
                    end
                    if rc > 0
                        xs = [xs, obj.Nodes{rc}.x];
                        ys = [ys, obj.Nodes{rc}.y];
                    end
                    
                    if ~isempty(xs)
                        obj.Nodes{idx}.x = mean(xs);
                        % Y is calculated bottom-up by averaging children's y and adding spacing
                        obj.Nodes{idx}.y = mean(ys) + 1.0 / max(1, maxDepth);
                    end
                end
            end
            
            % Normalise Y coordinates so root is exactly at 1.0, and leaves are at 0.0
            rootY = obj.Nodes{1}.y;
            if rootY > 0
                for idx = 1:N
                    % Force leaves to exactly 0.0
                    if obj.Nodes{idx}.isLeaf
                        obj.Nodes{idx}.y = 0.0;
                    else
                        obj.Nodes{idx}.y = obj.Nodes{idx}.y / rootY;
                    end
                end
            else
                obj.Nodes{1}.y = 1.0;
            end
            
            % 5. Drawing Edges and Nodes
            obj.EdgeLines = gobjects(1, N);
            obj.EdgeLabels = gobjects(1, N);
            obj.NodeCircles = gobjects(1, N);
            obj.NodeTexts = cell(1, N);
            
            gridCol = obj.CoreObj.Colors.Grid;
            primaryCol = obj.CoreObj.Colors.Primary;
            boundaryCol = obj.CoreObj.Colors.Boundary;
            textCol = obj.CoreObj.Colors.Text;
            
            % Constants for node circles
            LeafRadius = 0.14;
            InternalRadius = 0.05;
            
            % Draw Edge lines
            for i = 2:N
                pIdx = obj.Nodes{i}.parent;
                x1 = obj.Nodes{pIdx}.x; y1 = obj.Nodes{pIdx}.y;
                x2 = obj.Nodes{i}.x; y2 = obj.Nodes{i}.y;
                
                % Sleek futuristic line
                obj.EdgeLines(i) = line(ax, [x1, x2], [y1, y2], ...
                    'Color', gridCol, 'LineWidth', 1.5, 'HitTest', 'off');
                
                % Edge label ('0' or '1')
                bit = obj.Nodes{i}.codeword(end);
                mx = (x1 + x2) / 2;
                my = (y1 + y2) / 2 + 0.04;
                obj.EdgeLabels(i) = text(ax, mx, my, bit, ...
                    'Color', textCol * 0.7, 'FontSize', 9, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Interpreter', 'none');
            end
            
            % Draw Nodes
            for i = 1:N
                x = obj.Nodes{i}.x;
                y = obj.Nodes{i}.y;
                th = linspace(0, 2*pi, 30);
                
                if obj.Nodes{i}.isLeaf
                    r = LeafRadius;
                    % Draw beautiful leaf circle
                    obj.NodeCircles(i) = fill(ax, x + r*cos(th), y + r*sin(th), ...
                        boundaryCol, 'EdgeColor', textCol * 0.5, 'LineWidth', 1.5, 'HitTest', 'off');
                    
                    % Draw text: symbol name + probability below leaf node
                    sym = obj.Nodes{i}.symbol;
                    probVal = obj.Nodes{i}.prob;
                    obj.NodeTexts{i} = text(ax, x, y - r - 0.08, ...
                        sprintf('%s\np=%.3f', sym, probVal), ...
                        'Color', textCol, 'FontSize', 9, 'FontWeight', 'bold', ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                        'Interpreter', 'none');
                else
                    r = InternalRadius;
                    % Draw small dim internal node circle
                    obj.NodeCircles(i) = fill(ax, x + r*cos(th), y + r*sin(th), ...
                        primaryCol, 'EdgeColor', textCol * 0.4, 'LineWidth', 1.0, 'HitTest', 'off');
                end
            end
            
            % Adjust axes limits with nice padding
            minX = min(cellfun(@(n) n.x, obj.Nodes));
            maxX = max(cellfun(@(n) n.x, obj.Nodes));
            if minX == maxX
                minX = minX - 1.0;
                maxX = maxX + 1.0;
            end
            xPad = (maxX - minX) * 0.12 + 0.3;
            
            set(ax, 'XLim', [minX - xPad, maxX + xPad], 'YLim', [-0.35, 1.25]);
            axis(ax, 'off');
            
            hold(ax, 'off');
        end
        
        function highlightPathTo(obj, symbol)
            % Highlights the specific path of branches from the root node to
            % the selected symbol's leaf node in glowing blue/cyan.
            % Dims all other paths.
            arguments
                obj
                symbol
            end
            
            if isempty(obj.Nodes)
                return;
            end
            
            symbolStr = char(string(symbol));
            N = numel(obj.Nodes);
            
            % Find target leaf node ID
            targetId = 0;
            for i = 1:N
                if obj.Nodes{i}.isLeaf && strcmp(obj.Nodes{i}.symbol, symbolStr)
                    targetId = i;
                    break;
                end
            end
            
            % Trace path from target leaf up to root
            pathNodes = [];
            if targetId > 0
                curr = targetId;
                while curr > 0
                    pathNodes = [curr, pathNodes]; %#ok<AGROW>
                    curr = obj.Nodes{curr}.parent;
                end
            end
            
            % Color tokens
            gridCol = obj.CoreObj.Colors.Grid;
            primaryCol = obj.CoreObj.Colors.Primary;
            boundaryCol = obj.CoreObj.Colors.Boundary;
            textCol = obj.CoreObj.Colors.Text;
            
            % Glowing colors
            glowCyan = [0.0, 1.0, 1.0]; % Cyberpunk neon cyan
            
            % Update edges and labels
            for i = 2:N
                if ~isempty(pathNodes)
                    % Highlighted edge is in the active path
                    if ismember(i, pathNodes)
                        set(obj.EdgeLines(i), 'Color', glowCyan, 'LineWidth', 3.0);
                        set(obj.EdgeLabels(i), 'Color', [1.0, 1.0, 1.0], 'FontWeight', 'bold');
                    else
                        % Dim all other edges
                        set(obj.EdgeLines(i), 'Color', gridCol * 0.35, 'LineWidth', 0.8);
                        set(obj.EdgeLabels(i), 'Color', textCol * 0.25, 'FontWeight', 'normal');
                    end
                else
                    % Reset to default view (no highlighting)
                    set(obj.EdgeLines(i), 'Color', gridCol, 'LineWidth', 1.5);
                    set(obj.EdgeLabels(i), 'Color', textCol * 0.7, 'FontWeight', 'bold');
                end
            end
            
            % Update node circles and labels
            for i = 1:N
                if ~isempty(pathNodes)
                    if ismember(i, pathNodes)
                        % Highlighted active node
                        if obj.Nodes{i}.isLeaf
                            set(obj.NodeCircles(i), 'FaceColor', boundaryCol, ...
                                'EdgeColor', [1.0, 1.0, 1.0], 'LineWidth', 2.2);
                            if ~isempty(obj.NodeTexts{i}) && ishandle(obj.NodeTexts{i})
                                set(obj.NodeTexts{i}, 'Color', [1.0, 1.0, 1.0], 'FontWeight', 'bold');
                            end
                        else
                            set(obj.NodeCircles(i), 'FaceColor', glowCyan, ...
                                'EdgeColor', [1.0, 1.0, 1.0], 'LineWidth', 1.8);
                        end
                    else
                        % Dimmed inactive node
                        if obj.Nodes{i}.isLeaf
                            set(obj.NodeCircles(i), 'FaceColor', boundaryCol * 0.3, ...
                                'EdgeColor', textCol * 0.15, 'LineWidth', 0.8);
                            if ~isempty(obj.NodeTexts{i}) && ishandle(obj.NodeTexts{i})
                                set(obj.NodeTexts{i}, 'Color', textCol * 0.25, 'FontWeight', 'normal');
                            end
                        else
                            set(obj.NodeCircles(i), 'FaceColor', primaryCol * 0.25, ...
                                'EdgeColor', textCol * 0.15, 'LineWidth', 0.5);
                        end
                    end
                else
                    % Reset to default view
                    if obj.Nodes{i}.isLeaf
                        set(obj.NodeCircles(i), 'FaceColor', boundaryCol, ...
                            'EdgeColor', textCol * 0.5, 'LineWidth', 1.5);
                        if ~isempty(obj.NodeTexts{i}) && ishandle(obj.NodeTexts{i})
                            set(obj.NodeTexts{i}, 'Color', textCol, 'FontWeight', 'bold');
                        end
                    else
                        set(obj.NodeCircles(i), 'FaceColor', primaryCol, ...
                            'EdgeColor', textCol * 0.4, 'LineWidth', 1.0);
                    end
                end
            end
            
            drawnow limitrate;
        end
        
        function clearGraphics(obj)
            % Clean deletion of all handles
            if ~isempty(obj.EdgeLines)
                valid = ishandle(obj.EdgeLines);
                delete(obj.EdgeLines(valid));
                obj.EdgeLines = [];
            end
            if ~isempty(obj.EdgeLabels)
                valid = ishandle(obj.EdgeLabels);
                delete(obj.EdgeLabels(valid));
                obj.EdgeLabels = [];
            end
            if ~isempty(obj.NodeCircles)
                valid = ishandle(obj.NodeCircles);
                delete(obj.NodeCircles(valid));
                obj.NodeCircles = [];
            end
            if ~isempty(obj.NodeTexts)
                for i = 1:numel(obj.NodeTexts)
                    if ~isempty(obj.NodeTexts{i}) && ishandle(obj.NodeTexts{i})
                        delete(obj.NodeTexts{i});
                    end
                end
                obj.NodeTexts = {};
            end
        end
        
        function delete(obj)
            % Cleanup on destruction
            obj.clearGraphics();
        end
    end
end
