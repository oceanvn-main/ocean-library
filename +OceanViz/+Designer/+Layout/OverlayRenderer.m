classdef OverlayRenderer < handle
    % OVERLAYRENDERER  Draws legends, weight labels, and map titles on top of TrackDesigner.

    properties
        Designer_           % Parent TrackDesigner reference
        
        LegendHandles_ = {}
        WeightHandles_ = {}
        LegendPos1_ = [NaN NaN]
        LegendPos2_ = [NaN NaN]
        TitlePos_ = [NaN NaN]
        TitleHandle_
        MapTitle_ = 'TrackDesigner Network'
        ShowWeights_ = false
        WeightUnit_ = 'km'
        UnitsPerKm_ = 1.0;
        DraggingLegend_ = 0
        DragOffset_ = [0 0]
    end
    
    methods
        function obj = OverlayRenderer(designer)
            obj.Designer_ = designer;
        end

        function clearAll(obj)
            % CLEARALL  Nuke all visual overlay elements safely.
            obj.clearWeightLabels_();
            for k = 1:numel(obj.LegendHandles_)
                if isvalid(obj.LegendHandles_{k})
                    delete(obj.LegendHandles_{k});
                end
            end
            obj.LegendHandles_ = {};
            if ~isempty(obj.TitleHandle_) && isvalid(obj.TitleHandle_)
                delete(obj.TitleHandle_);
            end
        end
        
        function onToggleWeights_(obj, val)
            td = obj.Designer_;
            % Toggle edge weight label visibility
            obj.ShowWeights_ = logical(val);
            if obj.ShowWeights_
                obj.drawWeightLabels_();
                set(td.ToolButtons_.weights, 'BackgroundColor', [0.3 0.5 0.8]);
            else
                obj.clearWeightLabels_();
                set(td.ToolButtons_.weights, 'BackgroundColor', [0.2 0.2 0.25]);
            end
        end

        function onToggleWeightUnit_(obj, src)
            td = obj.Designer_;
            % Toggle between km and miles
            if get(src, 'Value') == 1
                obj.WeightUnit_ = 'mi';
                set(src, 'String', 'mi', 'ForegroundColor', [0.9 0.6 0.2]);
            else
                obj.WeightUnit_ = 'km';
                set(src, 'String', 'km', 'ForegroundColor', [0.4 0.9 0.4]);
            end
            % Redraw if weights are visible
            if obj.ShowWeights_
                obj.drawWeightLabels_();
            end
        end

        function drawWeightLabels_(obj)
            td = obj.Designer_;
            % DRAWWEIGHTLABELS_  Draw distance labels at midpoint of each edge curve.
            obj.clearWeightLabels_();
            E = td.Transit_.getRouteCount();
            for i = 1:E
                edge = td.Transit_.getRouteEdge(i);
                if isempty(edge), continue; end

                %Ask Math engine
                spline = td.Transit_.getSpline(i);
                w = edge.Weight;
                
                % Scale by the user-defined real-world kilometer ratio
                w = w / obj.UnitsPerKm_;

                % Convert if miles
                if strcmp(obj.WeightUnit_, 'mi')
                    w = w * 0.621371;
                end

                % Get midpoint position
                if ~isempty(spline) && ismethod(spline, 'evaluate')
                    midPt = spline.evaluate(0.5);
                    mx = midPt(1);
                    my = midPt(2);
                else
                    % Fallback: average of source and target positions
                    pSrc = td.Transit_.getStation(edge.Source).Position;
                    pTgt = td.Transit_.getStation(edge.Target).Position;
                    mx = (pSrc(1) + pTgt(1)) / 2;
                    my = (pSrc(2) + pTgt(2)) / 2;
                end

                % Format label — print to 1 decimal place to support smaller scaled values
                label = sprintf('%.1f', w);

                % Draw with background for readability
                hTxt = text(td.Axes_, mx, my, label, ...
                    'FontSize', 6, ...
                    'FontWeight', 'normal', ...
                    'Color', [0.1 0.1 0.5], ...
                    'BackgroundColor', [1 1 1 0.85], ...
                    'EdgeColor', [0.7 0.7 0.8], ...
                    'Margin', 1, ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.WeightHandles_{end+1} = hTxt;
            end
        end

        function clearWeightLabels_(obj)
            h = [obj.WeightHandles_{:}];
            if ~isempty(h), delete(h(isvalid(h))); end
            obj.WeightHandles_ = {};
        end

        function drawLegend_(obj)
            td = obj.Designer_;
            % DRAWLEGEND_  Draw two draggable legend boxes.
            %   Box 1: Train Services
            %   Box 2: Station Types + Path Notes
            %   Drag any title text to reposition its box.

            % Clear previous legend
            h = [obj.LegendHandles_{:}];
            if ~isempty(h), delete(h(isvalid(h))); end
            obj.LegendHandles_ = {};

            codes = td.Transit_.getLineNames();
            if isempty(codes), return; end

            % Data bounds for sizing
            positions = td.Transit_.getStationPositions();
            yMin = min(positions(:,2));
            yMax = max(positions(:,2));
            xMin = min(positions(:,1));
            xMax = max(positions(:,1));
            yRange = yMax - yMin;
            xRange = xMax - xMin;

            lineH = yRange * 0.025;
            lineBarW = xRange * 0.05;

            % Default positions if not yet placed
            if isnan(obj.LegendPos1_(1))
                obj.LegendPos1_ = [xMin, yMin + yRange * 0.35];
            end
            if isnan(obj.LegendPos2_(1))
                obj.LegendPos2_ = [xMin, yMin - yRange * 0.01];
            end
            if isnan(obj.TitlePos_(1))
                obj.TitlePos_ = [xMax - xRange * 0.01, yMax + yRange * 0.01];
            end

            % ============== MAP TITLE (draggable) ==============
            if ~isempty(obj.TitleHandle_) && isvalid(obj.TitleHandle_)
                delete(obj.TitleHandle_);
            end
            obj.TitleHandle_ = text(td.Axes_, obj.TitlePos_(1), obj.TitlePos_(2), 15, ...
                obj.MapTitle_, ...
                'FontSize', 14, 'FontWeight', 'bold', ...
                'Color', [0.05 0.05 0.05], ...
                'HorizontalAlignment', 'right', ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'on', 'PickableParts', 'visible', ...
                'ButtonDownFcn', @(~,~) obj.startLegendDrag_(3));

            % ============== BOX 1: Train Services ==============
            bx = obj.LegendPos1_(1);
            by = obj.LegendPos1_(2);
            nLines = numel(codes);

            % Title (draggable)
            h = text(td.Axes_, bx, by, 15, 'Train Services', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'on', 'PickableParts', 'visible', ...
                'ButtonDownFcn', @(~,~) obj.startLegendDrag_(1));
            obj.LegendHandles_{end+1} = h;

            entryY = by - lineH * 1.2;
            for i = 1:nLines
                lineObj = td.Transit_.getLine(codes{i});
                ey = entryY - (i-1) * lineH;

                hBar = plot3(td.Axes_, [bx, bx + lineBarW], [ey, ey], [15, 15], '-', ...
                    'Color', lineObj.Color, 'LineWidth', 5, ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hBar;

                nSta = lineObj.getStationCount();
                label = sprintf('%s  %s (%d)', lineObj.Code, lineObj.Name, nSta);
                hTxt = text(td.Axes_, bx + lineBarW * 1.2, ey, 15, label, ...
                    'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                    'VerticalAlignment', 'middle', ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hTxt;
            end

            % ============== BOX 2: Station Types + Path Notes ==============
            bx2 = obj.LegendPos2_(1);
            by2 = obj.LegendPos2_(2);

            % Station Types title (draggable)
            h = text(td.Axes_, bx2, by2, 15, 'Station Types', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'on', 'PickableParts', 'visible', ...
                'ButtonDownFcn', @(~,~) obj.startLegendDrag_(2));
            obj.LegendHandles_{end+1} = h;

            stEntryY = by2 - lineH * 1.2; %1.2 row-heights below the title
            types = {
                'o', [1 1 1], [0 0 0],        2.0, 'Interchange';
                'd', [1 1 1], [0.15 0.15 0.15], 1.5, 'Terminus';
                'o', [1 1 1], [0.4 0.4 0.4],   1.2, 'Standard'
            };
            for i = 1:size(types, 1)
                sy = stEntryY - (i-1) * lineH;
                hM = plot3(td.Axes_, bx2 + lineBarW * 0.4, sy, 15, ...
                    types{i,1}, ...
                    'MarkerFaceColor', types{i,2}, ...
                    'MarkerEdgeColor', types{i,3}, ...
                    'LineWidth', types{i,4}, ...
                    'MarkerSize', 10, ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hM;
                hTxt = text(td.Axes_, bx2 + lineBarW * 0.9, sy, 15, types{i,5}, ...
                    'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                    'VerticalAlignment', 'middle', ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hTxt;
            end

            % Path Notes
            notesY = stEntryY - 3 * lineH - lineH * 0.5; %Below (3 station types rows + half line H)
            h = text(td.Axes_, bx2, notesY, 15, 'Path Notes', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = h;

            noteEntryY = notesY - lineH * 1.2;
            c1 = [0.85 0.2 0.2]; c2 = [0 0.35 0.75];

            % Shared Path
            spY = noteEntryY;
            hL1 = plot3(td.Axes_, [bx2, bx2 + lineBarW], [spY+lineH*0.08, spY+lineH*0.08], [15, 15], '-', ...
                'Color', c1, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none');
            hL2 = plot3(td.Axes_, [bx2, bx2 + lineBarW], [spY-lineH*0.08, spY-lineH*0.08], [15, 15], '-', ...
                'Color', c2, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hL1;
            obj.LegendHandles_{end+1} = hL2;
            hTxt = text(td.Axes_, bx2 + lineBarW * 1.2, spY, 15, 'Shared Path', ...
                'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hTxt;

            % Line Crossing
            crY = noteEntryY - lineH; %one row below the "share path" entry
            hC1 = plot3(td.Axes_, [bx2, bx2 + lineBarW], [crY-lineH*0.12, crY+lineH*0.12], [15, 15], '-', ...
                'Color', c1, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none'); %Start at bottom-left, end at top-right
            hC2 = plot3(td.Axes_, [bx2, bx2 + lineBarW], [crY+lineH*0.12, crY-lineH*0.12], [15, 15], '-', ...
                'Color', c2, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none'); %Stat at top left, end at bottom-right
            obj.LegendHandles_{end+1} = hC1;
            obj.LegendHandles_{end+1} = hC2;
            hTxt = text(td.Axes_, bx2 + lineBarW * 1.2, crY, 15, 'Line Crossing', ...
                'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hTxt;

            % Distance label annotation
            distY = noteEntryY - 2 * lineH;
            hSample = text(td.Axes_, bx2 + lineBarW * 0.25, distY, 15, '5', ...
                'FontSize', 6, 'FontWeight', 'normal', ...
                'Color', [0.1 0.1 0.5], ...
                'BackgroundColor', [1 1 1 0.85], ...
                'EdgeColor', [0.7 0.7 0.8], ...
                'Margin', 1, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hSample;
            hTxt = text(td.Axes_, bx2 + lineBarW * 0.9, distY, 15, 'Distance (km/mi)', ...
                'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hTxt;
        end

        function startLegendDrag_(obj, boxId)
            td = obj.Designer_;
            % Begin dragging a legend box (1, 2, or 3=title)
            cp = get(td.Axes_, 'CurrentPoint');
            mousePos = cp(1, 1:2);
            if boxId == 1
                obj.DragOffset_ = mousePos - obj.LegendPos1_;
            elseif boxId == 2
                obj.DragOffset_ = mousePos - obj.LegendPos2_;
            else
                obj.DragOffset_ = mousePos - obj.TitlePos_;
            end
            obj.DraggingLegend_ = boxId;
            set(td.Figure_, 'WindowButtonMotionFcn', @(~,~) obj.onLegendDragMove_()); %Redaw, track the cursor
            set(td.Figure_, 'WindowButtonUpFcn', @(~,~) obj.onLegendDragEnd_()); %Stop dragging
        end

        function onLegendDragMove_(obj)
            td = obj.Designer_;
            if obj.DraggingLegend_ == 0, return; end
            cp = get(td.Axes_, 'CurrentPoint');
            newPos = cp(1, 1:2) - obj.DragOffset_;
            if obj.DraggingLegend_ == 1
                obj.LegendPos1_ = newPos;
            elseif obj.DraggingLegend_ == 2
                obj.LegendPos2_ = newPos;
            else
                obj.TitlePos_ = newPos;
            end
            obj.drawLegend_();
        end

        function onLegendDragEnd_(obj)
            td = obj.Designer_;
            obj.DraggingLegend_ = 0;
            set(td.Figure_, 'WindowButtonMotionFcn', '');
            set(td.Figure_, 'WindowButtonUpFcn', '');
        end

    end
end
