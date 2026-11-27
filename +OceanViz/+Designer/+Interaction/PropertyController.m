classdef PropertyController < handle
    % PROPERTYCONTROLLER  The "C" in MVC — translates every UI panel interaction into
    %   synchronized Math Engine + Graphics Engine commands.
    %
    %   This controller owns all callbacks from the right-hand Properties Panel
    %   (station name, elevation, label offset, font size), the Line Management
    %   Panel (create line, select active line, focus mode), the Curve Panel
    %   (draw order, curve type), and the Route-Here insertion feature.
    %
    %   It never draws anything itself. It reads raw strings from UI widgets,
    %   converts them to typed values, then commands Transit_ (Model) and
    %   Renderer_ (View) to update in lockstep.
    
    properties (SetAccess = private)
        Designer_   % Back-pointer to the parent TrackDesigner instance
    end
    
    methods
        function obj = PropertyController(designer)
            % PROPERTYCONTROLLER  Constructor. Binds this controller to a TrackDesigner.
            obj.Designer_ = designer;
        end
        
        function onApplyName(obj)
            % ONAPPLYNAME  Renames a station via the UI and logs the command properly for Undo.
            td = obj.Designer_;
            if td.SelectedStation_ == 0, return; end
            
            newName = get(td.PropPanel_.nameField, 'String');
            oldName = td.Transit_.getStation(td.SelectedStation_).Name;
            
            % Abort if nothing actually changed
            if strcmp(oldName, newName)
                return;
            end
            
            % Push to Command Pipeline for Ctrl+Z support
            cmd = OceanViz.Designer.Commands.RenameStationCommand(td, td.SelectedStation_, oldName, newName);
            if ~isempty(td.TimelineController_)
                td.TimelineController_.execute(cmd);
            else
                cmd.execute();
            end
        end

        function onApplyElevation(obj)
            % ONAPPLYELEVATION  Lifts a completely flat 2D station (and its connected tracks) high into 3D Space.
            td = obj.Designer_;
            if td.SelectedStation_ == 0, return; end
            valStr = get(td.PropPanel_.elevationField, 'String');
            val = str2double(valStr);
            if isnan(val), return; end % Catch if user typed words instead of numbers
            
            td.clearUndo_(); % No ElevationCommand object exists yet, so protect cache.
            
            node = td.Transit_.getStation(td.SelectedStation_);
            pos = node.Position;
            if length(pos) < 3
                pos = [pos(1), pos(2), val]; % Convert to 3D safely
            else
                pos(3) = val;
            end
            
            % 1. Mathematical Backend Update
            td.Transit_.updateStationPosition(td.SelectedStation_, pos);
            
            % 2. Visual Backend Update
            sH = td.Renderer_.StationHandles_{td.SelectedStation_};
            if ~isempty(sH) && isvalid(sH)
                set(sH, 'ZData', val);
            end
            lH = td.Renderer_.LabelHandles_{td.SelectedStation_};
            if ~isempty(lH) && isvalid(lH)
                lp = get(lH, 'Position');
                set(lH, 'Position', [lp(1), lp(2), val]);
            end
            
            % 3. Refresh adjacent Splines mathematically utilizing Math backend
            % If you lift a Station, you must mathematically bend the connecting tracks upwards!
            edges = td.Transit_.getGraph().getEdges();
            for eIdx = 1:numel(edges)
                e = edges{eIdx};
                if ~isempty(e) && (e.Source == td.SelectedStation_ || e.Target == td.SelectedStation_)
                    oldSpline = td.Transit_.getGraph().getSplineData(eIdx);
                    if ~isempty(oldSpline)
                        isSource = (e.Source == td.SelectedStation_);
                        newSpline = OceanViz.Designer.Math.GeometryBuilder.updateSplineEndpointZ(oldSpline, isSource, pos);
                        
                        td.Transit_.updateRouteSpline(eIdx, newSpline);
                        td.Renderer_.updateCurve(eIdx, newSpline);
                        if td.ActiveEditCurve_ == eIdx
                            td.Renderer_.updateControlPoints(eIdx, newSpline);
                        end
                    end
                end
            end
            td.updateStatus_();
        end
        
        function onRouteLineHere(obj)
            % ONROUTELINEHERE  Sever an active track entirely in half and perfectly connect it piece-by-piece to a new station.
            td = obj.Designer_;
            if td.SelectedStation_ == 0, return; end
            if isempty(td.ActiveLineCode_), return; end
            
            td.clearUndo_();

            pos = td.Transit_.getStation(td.SelectedStation_).Position;
            lineObj = td.Transit_.getLine(td.ActiveLineCode_);
            
            % 1. Activating the Radar to find the closest line segment
            [bestEdgeIdx, bestSplitParam] = OceanViz.Designer.Math.GeometryBuilder.findNearestCurveOnLine(td.Transit_, lineObj, pos);
            
            if bestEdgeIdx == 0
                uiwait(warndlg(sprintf('Cannot route active line "%s" here.\nNo existing edges for this line were found nearby.', td.ActiveLineCode_), 'Routing Failed'));
                return;
            end
            
            % Add UI Confirmation Alert before permanently mutating state
            msg = sprintf('Are you sure you want to merge Station %d into the %s (%s)?\n\nThis will permanently sever the track and clear your Undo history.', td.SelectedStation_, lineObj.Name, td.ActiveLineCode_);
            choice = questdlg(msg, 'Confirm Routing', 'Yes', 'Cancel', 'Cancel');
            if ~strcmp(choice, 'Yes')
                return;
            end
            
            % 2. Vaporize the continuous Master Track
            spline = td.Transit_.getSpline(bestEdgeIdx);
            e = td.Transit_.getRouteEdge(bestEdgeIdx);
            srcId = e.Source;
            tgtId = e.Target;
            
            td.Transit_.removeRoute(srcId, tgtId);
            td.Renderer_.deleteCurve(bestEdgeIdx);
            
            % 3. The Math Shatterer: Cleanly bisect the track into two newly anchored spline halves
            [spl1, spl2] = OceanViz.Designer.Math.GeometryBuilder.splitCurveAndSnap(spline, bestSplitParam, pos);
            
            % 4. Forging the New Reality: Add the two new tracks to the Math Database
            td.EdgeCount_ = td.EdgeCount_ + 1;
            e1Idx = td.EdgeCount_;
            td.Transit_.addRoute(srcId, td.SelectedStation_, spl1);
            
            td.EdgeCount_ = td.EdgeCount_ + 1;
            e2Idx = td.EdgeCount_;
            td.Transit_.addRoute(td.SelectedStation_, tgtId, spl2);
            
            % 5. Register the new station on the line and invalidate cached topology
            lineObj.registerStation(td.SelectedStation_);
            td.Transit_.invalidateLineCache();
            
            % 6. Resurrect the Graphics: Paint both new tracks and bind draggable handles
            edgeColor = lineObj.Color;
            td.Renderer_.drawCurve(e1Idx, spl1, 'Color', edgeColor);
            td.Renderer_.drawControlPoints(e1Idx, spl1, ...
                @(eIdx, cpIdx, newPos) td.InteractionHandler_.onControlPointMoved_(eIdx, cpIdx, newPos));
                
            td.Renderer_.drawCurve(e2Idx, spl2, 'Color', edgeColor);
            td.Renderer_.drawControlPoints(e2Idx, spl2, ...
                @(eIdx, cpIdx, newPos) td.InteractionHandler_.onControlPointMoved_(eIdx, cpIdx, newPos));
                
            % 7. Ensure station dots render on top of the newly drawn tracks
            td.Renderer_.bringStationsToFront();
            td.updateStationVisuals_(td.SelectedStation_);
            
            td.ActiveLineCode_ = td.ActiveLineCode_; % trigger ui refresh if needed
            td.updateStatus_();
            obj.onFocusModeChanged();
        end

        function onApplyOffset(obj)
            % ONAPPLYOFFSET  Moves the text label of the selected station by [dx, dy] pixels
            %   relative to the station dot. Useful for avoiding label overlap.
            td = obj.Designer_;
            if td.SelectedStation_ == 0, return; end
            dx = str2double(get(td.PropPanel_.offsetX, 'String'));
            dy = str2double(get(td.PropPanel_.offsetY, 'String'));
            if isnan(dx), dx = 0; end
            if isnan(dy), dy = 0.3; end
            td.Renderer_.setLabelOffset(td.SelectedStation_, [dx, dy]);
        end

        function onApplyAllOffsets(obj)
            % ONAPPLYALLOFFSETS  Batch-applies the same [dx, dy] label offset to every
            %   station on the map at once, for uniform label positioning.
            td = obj.Designer_;
            dx = str2double(get(td.PropPanel_.offsetX, 'String'));
            dy = str2double(get(td.PropPanel_.offsetY, 'String'));
            if isnan(dx), dx = 0; end
            if isnan(dy), dy = 0.3; end
            td.Renderer_.setAllLabelOffsets([dx, dy]);
        end

        function onCurveDrawOrderUp(obj)
            % ONCURVEDRAWORDERUP  "Bring to Front" — increments the Z-index draw order
            %   of the actively selected curve. If the curve was at order 0 (default),
            %   a visual halo effect is added to distinguish overlapping tracks.
            td = obj.Designer_;
            if td.ActiveEditCurve_ == 0, return; end
            eIdx = td.ActiveEditCurve_;
            current = td.Renderer_.getCurveDrawOrder(eIdx);
            newOrder = current + 1;
            td.Renderer_.setCurveDrawOrder(eIdx, newOrder);
            if current == 0
                td.Renderer_.ensureHalo(eIdx); % First promotion: add glow
            end
            td.Renderer_.reorderCurveLocal(eIdx, 'top');
            set(td.PropPanel_.drawOrderLabel, 'String', ...
                sprintf('Active Curve: Edge %d  |  Order: %d', eIdx, newOrder));
        end

        function onCurveDrawOrderDown(obj)
            % ONCURVEDRAWORDERDOWN  "Send to Back" — decrements the Z-index draw order.
            %   Clamps at 0 (floor). When a curve drops back to order 0, the
            %   visual halo is removed since it no longer overlaps others.
            td = obj.Designer_;
            if td.ActiveEditCurve_ == 0, return; end
            eIdx = td.ActiveEditCurve_;
            current = td.Renderer_.getCurveDrawOrder(eIdx);
            newOrder = max(0, current - 1);
            td.Renderer_.setCurveDrawOrder(eIdx, newOrder);
            if newOrder == 0
                td.Renderer_.removeHalo(eIdx); % Back to default: remove glow
            end
            td.Renderer_.reorderCurveLocal(eIdx, 'bottom');
            set(td.PropPanel_.drawOrderLabel, 'String', ...
                sprintf('Active Curve: Edge %d  |  Order: %d', eIdx, newOrder));
        end

        function onCreateLine(obj)
            % ONCREATELINE  Registers a brand-new transit line (e.g., "Red Line") into
            %   the Math Engine, picks a color from the 13-entry palette, refreshes
            %   the UI dropdown, and auto-selects the new line as active.
            td = obj.Designer_;
            code = strtrim(get(td.LinePanel_.codeField, 'String'));
            name = strtrim(get(td.LinePanel_.nameField, 'String'));
            if isempty(code), return; end
            
            % 13-color palette inspired by real-world transit systems
            palette = [
                0.85, 0.20, 0.20;   % Red (NSL)
                0.00, 0.60, 0.30;   % Green (EWL)
                0.60, 0.20, 0.70;   % Purple (NEL)
                0.95, 0.55, 0.10;   % Orange (CCL)
                0.00, 0.35, 0.75;   % Blue (DTL)
                0.60, 0.35, 0.15;   % Brown (TEL)
                0.00, 0.65, 0.65;   % Teal (JRL)
                0.60, 0.85, 0.20;   % Lime (CRL)
                0.45, 0.50, 0.50;   % Grey (LRT)
                0.10, 0.75, 0.85;   % Cyan
                0.90, 0.85, 0.10;   % Yellow
                0.85, 0.20, 0.65;   % Magenta
                0.95, 0.50, 0.65;   % Pink
            ];
            colorIdx = get(td.LinePanel_.colorPicker, 'Value');
            color = palette(colorIdx, :);

            % 1. Add line to the TransitNetwork Math Engine
            td.Transit_.addLine(code, name, color);

            % 2. Rebuild the dropdown widget so the new line appears
            td.refreshLineDropdown_();

            % 3. Auto-select the newly created line as the active drawing color
            items = get(td.LinePanel_.activeDropdown, 'String');
            lineIdx = find(strcmp(items, sprintf('%s (%s)', code, name)));
            if ~isempty(lineIdx)
                set(td.LinePanel_.activeDropdown, 'Value', lineIdx(1));
                td.ActiveLineCode_ = code;
                set(td.LinePanel_.colorIndicator, ...
                    'BackgroundColor', color, 'String', '');
            end

            td.updateStatus_();
            obj.onFocusModeChanged();
        end

        function onSelectActiveLine(obj, src)
            % ONSELECTACTIVELINE  Dropdown callback — switches the active transit line
            %   that all future track-drawing operations will use. Updates the color
            %   indicator swatch and triggers focus-mode filtering.
            td = obj.Designer_;
            idx = get(src, 'Value');
            items = get(src, 'String');
            if idx == 1
                td.ActiveLineCode_ = '';  % "None" selected
                set(td.LinePanel_.colorIndicator, ...
                    'BackgroundColor', [0.15, 0.15, 0.18], 'String', '');
            else
                selected = items{idx};
                code = strtrim(extractBefore(selected, ' ('));  % Parse code from "NSL (North-South)"
                td.ActiveLineCode_ = code;
                lineObj = td.Transit_.getLine(code);
                set(td.LinePanel_.colorIndicator, ...
                    'BackgroundColor', lineObj.Color, 'String', '');
            end
            td.updateStatus_();
            
            % Trigger focus update if active
            obj.onFocusModeChanged();
        end

        function onFocusModeChanged(obj)
            % ONFOCUSMODECHANGED  Toggles "Focus Mode" — dims all tracks and stations
            %   that do NOT belong to the currently active line. If the user had a
            %   station selected that isn't part of the focused line, it is deselected
            %   to prevent confusing edits on invisible geometry.
            td = obj.Designer_;
            if isempty(td.FocusCheckbox_) || ~isvalid(td.FocusCheckbox_)
                return;
            end
            
            isFocus = get(td.FocusCheckbox_, 'Value');
            activeCode = td.ActiveLineCode_;
            
            if isempty(td.Renderer_)
                return;
            end
            
            % Delegate the actual alpha/visibility toggling to the Renderer
            td.Renderer_.applyFocus(isFocus, activeCode, td.Transit_);
            
            % Safety: deselect any station that is now hidden by focus filtering
            if isFocus && td.SelectedStation_ > 0 && ~isempty(activeCode)
                lObj = td.Transit_.getLine(activeCode);
                if ~lObj.hasStation(td.SelectedStation_)
                    td.selectStation_(0);
                end
            end
        end

        function onSelectCurveType(obj, src)
            % ONSELECTCURVETYPE  Dropdown callback — sets the spline math engine
            %   that GeometryBuilder will use for all future track creations.
            %   'catmullrom' = curves through control points,
            %   'bezier'     = curves pulled by invisible gravity handles,
            %   'linear'     = straight lines only.
            td = obj.Designer_;
            idx = get(src, 'Value');
            switch idx
                case 1, td.CurveType_ = 'catmullrom';
                case 2, td.CurveType_ = 'bezier';
                case 3, td.CurveType_ = 'linear';
            end
        end

        function onApplyFontSize(obj)
            % ONAPPLYFONTSIZE  Globally sets the font size for every station label
            %   on the map. Defaults to 9pt if the user enters invalid input.
            td = obj.Designer_;
            sz = str2double(get(td.PropPanel_.fontSizeField, 'String'));
            if isnan(sz) || sz <= 0, sz = 9; end
            td.Renderer_.setLabelFontSize(sz);
        end
    end
end
