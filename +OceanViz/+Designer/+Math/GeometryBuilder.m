classdef GeometryBuilder
    % GEOMETRYBUILDER  Static math utility for generating and transforming splines
    %
    % Extracts curve creation, intersection detection, splitting, and
    % 3D spatial adjustments out of the TrackDesigner GUI loop.
    
    methods (Static)
        
        function spline = generateConnectionSpline(pos1, pos2, curveType, nMid)
            % GENERATECONNECTIONSPLINE  Creates a 3D curve with perceptual offset
            % nMid default to 1.
            if length(pos1) < 3, pos1 = [pos1(1), pos1(2), 0]; end
            if length(pos2) < 3, pos2 = [pos2(1), pos2(2), 0]; end
            
            controlPts = zeros(nMid + 2, 3);
            %Locking station A in to first, and B into final
            controlPts(1, :)   = pos1; 
            controlPts(end, :) = pos2;

            for k = 1:nMid
                frac = k / (nMid + 1);
                mid = pos1 + frac * (pos2 - pos1);
                dir = pos2 - pos1;
                perp = [-dir(2), dir(1), 0];
                perpNorm = sqrt(sum(perp.^2));
                if perpNorm > 1e-12
                    perp = perp / perpNorm;
                end
                offset = 0.3 * perpNorm * sin(pi * frac);
                controlPts(k+1, :) = mid + offset * perp;
            end

            switch lower(curveType)
                case 'linear'
                    spline = OceanMath.Geometry.BezierCurve([pos1; pos2]);
                case 'bezier'
                    spline = OceanMath.Geometry.BezierCurve(controlPts);
                otherwise % 'catmullrom'
                    spline = OceanMath.Geometry.CatmullRomSpline(controlPts);
            end
        end
        
        function newSpline = updateSplineEndpointZ(oldSpline, isSource, zPos)
            % UPDATESPLINEENDPOINTZ  Modifies Z-height of start or end point
            cps = oldSpline.getControlPoints();
            if size(cps, 2) < 3
                cps = [cps, zeros(size(cps,1), 1)]; %update third dimension as 0
            end
            if isSource
                cps(1, :) = zPos;
            else
                cps(end, :) = zPos;
            end
            
            if isa(oldSpline, 'OceanMath.Geometry.CatmullRomSpline')
                newSpline = OceanMath.Geometry.CatmullRomSpline(cps);
            else
                newSpline = OceanMath.Geometry.BezierCurve(cps);
            end
        end
        
        function [bestEdgeIdx, bestSplitParam] = findNearestCurveOnLine(transitNetwork, lineObj, targetPos)
            % FINDNEARESTCURVEONLINE Returns exact parametric intersection closest to point. Use for detect the insertion point in the curve
            bestEdgeIdx = 0;
            minDist = inf;
            bestSplitParam = 0;
            
            edges = transitNetwork.getGraph().getEdges();
            for i = 1:numel(edges)
                e = edges{i};
                if isempty(e), continue; end
                
                s = e.Source;
                t = e.Target;
                if lineObj.hasStation(s) && lineObj.hasStation(t) %Filter to check if the edge belongs to the line
                    spl = transitNetwork.getSpline(i); 
                    if ~isempty(spl)
                        tVals = spl.parameterize(200); %Smash into 200 discrete dots
                        pts = spl.evaluate(tVals);
                        minDim = min(size(pts,2), numel(targetPos)); %Take the min dimensional size
                        dists = sqrt(sum((pts(:, 1:minDim) - targetPos(1:minDim)).^2, 2)); %Min distance from target point to the c
                        [d, nearIdx] = min(dists);
                        if d < minDist
                            minDist = d;
                            bestEdgeIdx = i;
                            bestSplitParam = tVals(nearIdx);
                        end
                    end
                end
            end
        end
        
        function [spl1, spl2] = splitCurveAndSnap(spline, tSplit, snapPos)
            % SPLITCURVEANDSNAP Breaks curve optimally at tSplit and anchors to snapPos
            isCatmull = isa(spline, 'OceanMath.Geometry.CatmullRomSpline');
            
            testPt = spline.evaluate(0);
            if numel(testPt) > numel(snapPos)
                snapPos(end+1:numel(testPt)) = 0;
            end
            
            if isCatmull
                t1 = linspace(0, tSplit, 4)';
                cp1 = spline.evaluate(t1);
                cp1(end, :) = snapPos; %Snap the end point of the first curve to the snap postions (start of the station icon)
                spl1 = OceanMath.Geometry.CatmullRomSpline(cp1);
                
                t2 = linspace(tSplit, 1, 4)';
                cp2 = spline.evaluate(t2);
                cp2(1, :) = snapPos; %Snap the start point of the second curve to the snap postions (end of the station icon)
                spl2 = OceanMath.Geometry.CatmullRomSpline(cp2);
            else
                %Notice. Control points are not on the curve. So we need to figure out what are they to define the shape of two splitted-curve
                fitCubic = @(Q) [ ...
                    Q(1,:); ...
                    (-5*Q(1,:) + 18*Q(2,:) - 9*Q(3,:) + 2*Q(4,:)) / 6; ...
                    (2*Q(1,:) - 9*Q(2,:) + 18*Q(3,:) - 5*Q(4,:)) / 6; ...
                    Q(4,:) ];

                t1 = linspace(0, tSplit, 4)';
                q1 = spline.evaluate(t1);
                q1(end, :) = snapPos;
                cp1 = fitCubic(q1);
                spl1 = OceanMath.Geometry.BezierCurve(cp1);

                t2 = linspace(tSplit, 1, 4)';
                q2 = spline.evaluate(t2);
                q2(1, :) = snapPos;
                cp2 = fitCubic(q2);
                spl2 = OceanMath.Geometry.BezierCurve(cp2);
            end
        end
        
    end
end
