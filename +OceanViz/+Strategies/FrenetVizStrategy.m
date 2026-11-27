classdef FrenetVizStrategy < handle
    % FRENETVIZSTRATEGY  Render Frenet-Serret visualizations on parametric curves.
    %
    % Provides three rendering modes:
    %   'tnb'         - Tangent (red), Normal (green), Binormal (blue) quiver arrows
    %   'comb'        - Curvature comb: normal-direction ticks proportional to kappa
    %   'osculating'  - Osculating circle at a given parameter value
    %   'all'         - All three combined
    %
    % Usage (static):
    %   curve = OceanMath.Geometry.CatmullRomSpline(pts);
    %   FrenetVizStrategy.render(ax, curve, viz.Colors)
    %   FrenetVizStrategy.render(ax, curve, viz.Colors, 'Mode', 'tnb', 'ArrowScale', 0.8)
    %
    %   % Animate TNB at a single point:
    %   h = FrenetVizStrategy.createTNBHandles(ax, 0.5);
    %   FrenetVizStrategy.updateTNB(h, curve, 0.3, 0.5);
    %
    % See also: OceanMath.Geometry.FrenetFrame, OceanMath.Geometry.IParametricCurve

    methods (Static)

        function handles = render(ax, curve, colors, options)
            % RENDER  Draw Frenet visualizations on axes.
            %
            %   handles = FrenetVizStrategy.render(ax, curve, colors, ...)
            %
            %   Returns a struct of graphics handles for later cleanup/update.
            arguments
                ax
                curve
                colors
                options.Mode        (1,:) char = 'all'
                options.TSamples    (:,1) double = []
                options.CombN       (1,1) double {mustBePositive, mustBeInteger} = 40
                options.CombScale   (1,1) double = 1.0
                options.ArrowScale  (1,1) double = 0.5
                options.FocusT      (1,1) double = 0.5
                options.ShowLabels  (1,1) logical = true
                options.CurveAlpha  (1,1) double = 0.4
            end

            frenet = OceanMath.Geometry.FrenetFrame(curve);
            handles = struct();

            doTNB = any(strcmpi(options.Mode, {'tnb','all'}));
            doComb = any(strcmpi(options.Mode, {'comb','all'}));
            doOsc = any(strcmpi(options.Mode, {'osculating','all'}));

            % --- Draw the curve itself (faint background) ---
            tCurve = curve.parameterize(200);
            pts = curve.evaluate(tCurve);
            if isfield(colors, 'Primary')
                curveColor = colors.Primary;
            else
                curveColor = [0.5 0.8 1.0];
            end
            handles.Curve = plot(ax, pts(:,1), pts(:,2), '-', ...
                'Color', [curveColor, options.CurveAlpha], ...
                'LineWidth', 2);
            hold(ax, 'on');

            % --- TNB quiver arrows ---
            if doTNB
                if isempty(options.TSamples)
                    options.TSamples = linspace(0.02, 0.98, 10)';
                end
                handles.TNB = OceanViz.Strategies.FrenetVizStrategy.drawTNB_( ...
                    ax, curve, frenet, options.TSamples, ...
                    options.ArrowScale, options.ShowLabels);
            end

            % --- Curvature comb ---
            if doComb
                handles.Comb = OceanViz.Strategies.FrenetVizStrategy.drawComb_( ...
                    ax, curve, frenet, options.CombN, options.CombScale);
            end

            % --- Osculating circle ---
            if doOsc
                handles.Osculating = OceanViz.Strategies.FrenetVizStrategy.drawOsculating_( ...
                    ax, curve, frenet, options.FocusT);
            end
        end

        % ============================================================== %
        %  Live TNB Handles (for animation)
        % ============================================================== %

        function h = createTNBHandles(ax, arrowScale)
            % CREATETNBHANDLES  Create quiver handles for live TNB animation.
            %
            %   h = FrenetVizStrategy.createTNBHandles(ax, 0.5)
            %
            %   Returns struct with T, N, B quiver handles and optional labels.
            arguments
                ax
                arrowScale (1,1) double = 0.5
            end

            hold(ax, 'on');

            % T = red, N = green, B = blue
            h.T = quiver(ax, NaN, NaN, NaN, NaN, 0, ...
                'Color', [1.0, 0.3, 0.3], 'LineWidth', 2.5, ...
                'MaxHeadSize', 0.6);
            h.N = quiver(ax, NaN, NaN, NaN, NaN, 0, ...
                'Color', [0.3, 1.0, 0.3], 'LineWidth', 2.5, ...
                'MaxHeadSize', 0.6);
            h.B = quiver(ax, NaN, NaN, NaN, NaN, 0, ...
                'Color', [0.3, 0.5, 1.0], 'LineWidth', 2.5, ...
                'MaxHeadSize', 0.6);

            % Labels
            h.TLabel = text(ax, NaN, NaN, 'T', 'Color', [1.0, 0.3, 0.3], ...
                'FontWeight', 'bold', 'FontSize', 10);
            h.NLabel = text(ax, NaN, NaN, 'N', 'Color', [0.3, 1.0, 0.3], ...
                'FontWeight', 'bold', 'FontSize', 10);
            h.BLabel = text(ax, NaN, NaN, 'B', 'Color', [0.3, 0.5, 1.0], ...
                'FontWeight', 'bold', 'FontSize', 10);

            h.ArrowScale = arrowScale;
        end

        function updateTNB(h, curve, t, arrowScale)
            % UPDATETNB  Update live TNB quiver handles at parameter t.
            %
            %   FrenetVizStrategy.updateTNB(h, curve, 0.3, 0.5)
            arguments
                h
                curve
                t          (1,1) double
                arrowScale (1,1) double = 0.5
            end

            frenet = OceanMath.Geometry.FrenetFrame(curve);
            pos = curve.evaluate(t);
            [Tv, Nv, ~] = frenet.frame(t);

            % Scale for visual clarity
            Ts = Tv * arrowScale;
            Ns = Nv * arrowScale;

            x = pos(1); y = pos(2);

            % Update quiver positions
            set(h.T, 'XData', x, 'YData', y, 'UData', Ts(1), 'VData', Ts(2));
            set(h.N, 'XData', x, 'YData', y, 'UData', Ns(1), 'VData', Ns(2));

            % Update labels
            set(h.TLabel, 'Position', [x + Ts(1)*1.15, y + Ts(2)*1.15, 0]);
            set(h.NLabel, 'Position', [x + Ns(1)*1.15, y + Ns(2)*1.15, 0]);

            % B is only meaningful in 3D, hide in 2D
            if curve.getDimension() < 3
                set(h.B, 'Visible', 'off');
                set(h.BLabel, 'Visible', 'off');
            end
        end

        % ============================================================== %
        %  Osculating Circle Handles (for animation)
        % ============================================================== %

        function h = createOsculatingHandle(ax)
            % CREATEOSCULATINGHANDLE  Create a plot handle for a live osculating circle.
            arguments
                ax
            end
            hold(ax, 'on');
            h.Circle = plot(ax, NaN, NaN, '--', ...
                'Color', [1.0, 0.85, 0.3, 0.7], 'LineWidth', 1.5);
            h.Center = plot(ax, NaN, NaN, '+', ...
                'Color', [1.0, 0.85, 0.3], 'MarkerSize', 8, 'LineWidth', 1.5);
            h.RadiusText = text(ax, NaN, NaN, '', ...
                'Color', [1.0, 0.85, 0.3], 'FontSize', 9);
        end

        function updateOsculating(h, curve, t)
            % UPDATEOSCULATING  Redraw the osculating circle at parameter t.
            arguments
                h
                curve
                t (1,1) double
            end

            frenet = OceanMath.Geometry.FrenetFrame(curve);
            pos = curve.evaluate(t);
            kappa = frenet.curvature(t);
            [~, Nv, ~] = frenet.frame(t);

            if kappa < 1e-8
                % Nearly straight — hide circle
                set(h.Circle, 'XData', NaN, 'YData', NaN);
                set(h.Center, 'XData', NaN, 'YData', NaN);
                set(h.RadiusText, 'String', 'R = inf');
                return;
            end

            R = 1 / kappa;
            % Hide if radius too large (nearly straight — circle not meaningful)
            if R > 10
                set(h.Circle, 'XData', NaN, 'YData', NaN);
                set(h.Center, 'XData', NaN, 'YData', NaN);
                set(h.RadiusText, 'String', '');
                return;
            end

            center = pos + Nv * R;

            theta = linspace(0, 2*pi, 64);
            cx = center(1) + R * cos(theta);
            cy = center(2) + R * sin(theta);

            set(h.Circle, 'XData', cx, 'YData', cy);
            set(h.Center, 'XData', center(1), 'YData', center(2));
            set(h.RadiusText, 'Position', [center(1)+0.15, center(2)+0.15, 0], ...
                'String', sprintf('R=%.2f', R));
        end
    end

    % ================================================================== %
    methods (Static, Access = private)

        function handles = drawTNB_(ax, curve, frenet, tSamples, arrowScale, showLabels)
            % DRAWTNB_  Render TNB arrows at sample points.
            M = numel(tSamples);
            pts = curve.evaluate(tSamples);
            [Tv, Nv, Bv] = frenet.frame(tSamples);
            d = curve.getDimension();

            Ts = Tv * arrowScale;
            Ns = Nv * arrowScale;

            handles.hT = quiver(ax, pts(:,1), pts(:,2), Ts(:,1), Ts(:,2), 0, ...
                'Color', [1.0, 0.3, 0.3], 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
                'DisplayName', 'T (tangent)');
            handles.hN = quiver(ax, pts(:,1), pts(:,2), Ns(:,1), Ns(:,2), 0, ...
                'Color', [0.3, 1.0, 0.3], 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
                'DisplayName', 'N (normal)');

            if d >= 3
                Bs = Bv * arrowScale;
                handles.hB = quiver3(ax, pts(:,1), pts(:,2), pts(:,3), ...
                    Bs(:,1), Bs(:,2), Bs(:,3), 0, ...
                    'Color', [0.3, 0.5, 1.0], 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
                    'DisplayName', 'B (binormal)');
            end

            if showLabels && M > 0
                % Label the first arrow set
                idx = ceil(M/2);
                text(ax, pts(idx,1)+Ts(idx,1)*1.2, pts(idx,2)+Ts(idx,2)*1.2, 'T', ...
                    'Color', [1.0, 0.3, 0.3], 'FontWeight', 'bold', 'FontSize', 10);
                text(ax, pts(idx,1)+Ns(idx,1)*1.2, pts(idx,2)+Ns(idx,2)*1.2, 'N', ...
                    'Color', [0.3, 1.0, 0.3], 'FontWeight', 'bold', 'FontSize', 10);
            end
        end

        function handles = drawComb_(ax, curve, frenet, combN, combScale)
            % DRAWCOMB_  Render curvature comb along the curve.
            tSamples = linspace(0.01, 0.99, combN)';
            pts = curve.evaluate(tSamples);
            kappa = frenet.curvature(tSamples);
            [~, Nv, ~] = frenet.frame(tSamples);

            % Scale curvature by visual factor
            % Normalize kappa so the tallest tooth = combScale
            maxK = max(kappa);
            if maxK < 1e-10
                scaledK = zeros(size(kappa));
            else
                scaledK = (kappa / maxK) * combScale;
            end

            handles.Lines = gobjects(combN, 1);
            for i = 1:combN
                endPt = pts(i,:) + Nv(i,:) * scaledK(i);
                handles.Lines(i) = plot(ax, ...
                    [pts(i,1), endPt(1)], [pts(i,2), endPt(2)], '-', ...
                    'Color', [0.9, 0.6, 1.0, 0.6], 'LineWidth', 1.2);
            end

            % Connect the comb tips with a smooth line
            tips = pts + Nv .* scaledK;
            handles.TipLine = plot(ax, tips(:,1), tips(:,2), '-', ...
                'Color', [0.9, 0.6, 1.0, 0.4], 'LineWidth', 1);
        end

        function handles = drawOsculating_(ax, curve, frenet, focusT)
            % DRAWOSCULATING_  Render the osculating circle at a single point.
            pos = curve.evaluate(focusT);
            kappa = frenet.curvature(focusT);
            [~, Nv, ~] = frenet.frame(focusT);

            if kappa < 1e-8
                handles.Circle = plot(ax, NaN, NaN);
                handles.Center = plot(ax, NaN, NaN);
                return;
            end

            R = 1 / kappa;
            if R > 10
                handles.Circle = plot(ax, NaN, NaN);
                handles.Center = plot(ax, NaN, NaN);
                handles.Label = text(ax, NaN, NaN, '');
                return;
            end
            center = pos + Nv * R;

            theta = linspace(0, 2*pi, 64);
            cx = center(1) + R * cos(theta);
            cy = center(2) + R * sin(theta);

            handles.Circle = plot(ax, cx, cy, '--', ...
                'Color', [1.0, 0.85, 0.3, 0.7], 'LineWidth', 1.5, ...
                'DisplayName', 'Osculating Circle');
            handles.Center = plot(ax, center(1), center(2), '+', ...
                'Color', [1.0, 0.85, 0.3], 'MarkerSize', 8, 'LineWidth', 1.5);
            handles.Label = text(ax, center(1)+0.15, center(2)+0.15, ...
                sprintf('R=%.2f', R), ...
                'Color', [1.0, 0.85, 0.3], 'FontSize', 9);
        end
    end
end
