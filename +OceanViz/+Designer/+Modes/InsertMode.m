classdef InsertMode < OceanViz.Designer.Modes.BaseMode
    % INSERTMODE  Interaction mode for splitting a curve to insert a station.
    %
    %   Clicking exactly on an existing edge deletes the edge, inserts a
    %   new station at the intersection footprint, and splices two new
    %   curves (matching the original trajectory) in its place.
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode

    methods
        function onAxesClick(obj, handler, clickPos, rawPoint)
            obj.insertStationOnCurve_(handler, clickPos, rawPoint);
        end

        function onStationClick(~, ~, ~)
            % Ignored
        end

        function exit(~, handler)
            td = handler.Designer_;
            if ~isempty(td.Renderer_)
                td.Renderer_.destroyAllControlPoints();
            end
        end

        function msg = getStatusText(obj, designer)
            suffix = obj.getLineSuffix(designer);
            msg = ['Mode: INSERT  |  Click near a curve to add a station on it', suffix];
        end

        function name = getName(~)
            name = 'insert';
        end
    end

    methods (Access = private)
        function insertStationOnCurve_(~, handler, clickPos, rawPoint) %#ok<INUSL>
            td = handler.Designer_;
            
            % Confirm deletion and wipe undo stack
            ansStr = questdlg('Break this line to insert a station? (Cannot be undone)', 'Confirm', 'Yes', 'No', 'No');
            if ~strcmp(ansStr, 'Yes'), return; end
            
            if ~isempty(td.TimelineController_)
                td.TimelineController_.clear();
            end

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            [edgeIdx, ~] = td.Renderer_.findNearestCurve(rawPoint, threshold);
            if edgeIdx == 0, return; end

            e = td.Transit_.getRouteEdge(edgeIdx);
            if isempty(e), return; end
            srcId = e.Source;
            tgtId = e.Target;

            spline = td.Transit_.getSpline(edgeIdx);
            tVals = spline.parameterize(200);
            pts = spline.evaluate(tVals);

            % Math to find the exact closest point on the 3D curve

            pFront = rawPoint(1, :);
            pBack = rawPoint(2, :);
            vDir = pBack - pFront;
            vDir = vDir / norm(vDir);
            W = pts - pFront;
            projs = W(:,1)*vDir(1) + W(:,2)*vDir(2) + W(:,3)*vDir(3);
            W_perp = W - projs * vDir;
            dists = sqrt(sum(W_perp.^2, 2));

            [~, nearIdx] = min(dists);
            insertPos = pts(nearIdx, :);

            td.StationCount_ = td.StationCount_ + 1;
            name = sprintf('S%d', td.StationCount_);
            newId = td.Transit_.addStation(name, insertPos);

            if edgeIdx <= numel(td.Renderer_.CurveColors_) && ...
                    ~isempty(td.Renderer_.CurveColors_{edgeIdx})
                edgeColor = td.Renderer_.CurveColors_{edgeIdx};
            else
                edgeColor = [0.4, 0.8, 1.0];
            end

            td.Transit_.removeRoute(srcId, tgtId);
            td.Renderer_.deleteCurve(edgeIdx);

            tSplit = tVals(nearIdx);
            isCatmull = isa(spline, 'OceanMath.Geometry.CatmullRomSpline');

            if isCatmull
                t1 = linspace(0, tSplit, 4)';
                cp1 = spline.evaluate(t1);
                spl1 = OceanMath.Geometry.CatmullRomSpline(cp1);

                t2 = linspace(tSplit, 1, 4)';
                cp2 = spline.evaluate(t2);
                spl2 = OceanMath.Geometry.CatmullRomSpline(cp2);
            else
                fitCubic = @(Q) [ ...
                    Q(1,:); ...
                    (-5*Q(1,:) + 18*Q(2,:) - 9*Q(3,:) + 2*Q(4,:)) / 6; ...
                    (2*Q(1,:) - 9*Q(2,:) + 18*Q(3,:) - 5*Q(4,:)) / 6; ...
                    Q(4,:) ];

                t1 = linspace(0, tSplit, 4)';
                q1 = spline.evaluate(t1);
                cp1 = fitCubic(q1);
                spl1 = OceanMath.Geometry.BezierCurve(cp1);

                t2 = linspace(tSplit, 1, 4)';
                q2 = spline.evaluate(t2);
                cp2 = fitCubic(q2);
                spl2 = OceanMath.Geometry.BezierCurve(cp2);
            end

            td.EdgeCount_ = td.EdgeCount_ + 1;
            e1Idx = td.EdgeCount_;
            td.Transit_.addRoute(srcId, newId, spl1);

            td.EdgeCount_ = td.EdgeCount_ + 1;
            e2Idx = td.EdgeCount_;
            td.Transit_.addRoute(newId, tgtId, spl2);

            %Multi-Line Network Routing
            lineCodes = td.Transit_.getLineNames();
            for lcIdx = 1:numel(lineCodes)
                lc = lineCodes{lcIdx};
                lObj = td.Transit_.getLine(lc);
                if lObj.hasStation(srcId) && lObj.hasStation(tgtId)
                    lObj.registerStation(newId);
                end
            end


            if ~isempty(td.Renderer_)
                node = td.Transit_.getStation(newId);
                td.Renderer_.drawStation(node, @(nId) handler.onStationClick_(nId));

                [grabCb, moveCb, relCb] = handler.makeCallbacks_();
                td.Renderer_.drawCurve(e1Idx, spl1, 'Color', edgeColor);
                td.Renderer_.drawCurve(e2Idx, spl2, 'Color', edgeColor);
                td.Renderer_.drawControlPoints(e1Idx, spl1, grabCb, moveCb, relCb);
                td.Renderer_.drawControlPoints(e2Idx, spl2, grabCb, moveCb, relCb);
                td.Renderer_.bringStationsToFront();
            end

            td.updateStationVisuals_(newId);
            td.updateStatus_();
        end
    end
end
