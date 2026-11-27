classdef LikelihoodOptimizationStrategy < OceanViz.Strategies.BaseStrategy
    % LIKELIHOODOPTIMIZATIONSTRATEGY OceanViz 4-panel dashboard & dedicated animated movie for solver comparison.
    %
    % NAMESPACE: OceanViz.Strategies
    %
    % Data Contract (data struct fields):
    %   data.PhiGrid    — (N x N) double phi meshgrid
    %   data.ThetaGrid  — (N x N) double theta meshgrid
    %   data.LGrid      — (N x N) double log-likelihood surface grid
    %   data.TrueParam  — (1 x 2) double [phi_true, theta_true]
    %   data.StartParam — (1 x 2) double [phi_0, theta_0]
    %   data.Histories  — struct containing solver telemetry (.Steepest, .Newton, .DFP, .BFGS)
    %
    % Options Contract:
    %   options.Animate   — logical (default false). Set true for dedicated full-screen trajectory movie.
    %   options.PauseTime — double (default 0.08). Delay in seconds between movie frames.
    %   options.SaveGif   — string (default ""). Output filename to export animated GIF (e.g., "trajectory.gif").
    %
    % Author: Ocean Library System
    % Date: 2026-09-03

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data     struct
                options  struct = struct()
            end

            colors = obj.Parent.Colors;

            % Required fields validation
            required = {'PhiGrid', 'ThetaGrid', 'LGrid', 'TrueParam', 'StartParam', 'Histories'};
            for k = 1:numel(required)
                if ~isfield(data, required{k})
                    error('LikelihoodOptimizationStrategy:MissingField', ...
                        'data must contain field: %s', required{k});
                end
            end

            % Parse options
            isAnimate = isfield(options, 'Animate') && options.Animate;
            pauseTime = 0.08;
            if isfield(options, 'PauseTime') && options.PauseTime > 0
                pauseTime = options.PauseTime;
            end
            saveGif = "";
            if isfield(options, 'SaveGif') && options.SaveGif ~= ""
                saveGif = string(options.SaveGif);
            end

            % Extract parent figure and remove placeholder axis
            if ishandle(ax) && isvalid(ax)
                fig = ax.Parent;
                delete(ax);
            else
                fig = gcf;
            end

            % High-Contrast Neon Palette designed for maximum separation on dark maps
            neonColors = struct();
            neonColors.Steepest = [1.0, 0.15, 0.85]; % Hot Neon Magenta
            neonColors.Newton   = [0.0, 0.90, 1.00]; % Electric Cyan
            neonColors.DFP      = [1.0, 0.50, 0.00]; % Vivid Electric Orange / Coral
            neonColors.BFGS     = [1.0, 0.95, 0.00]; % Bright Electric Yellow

            if isAnimate
                obj.renderDedicatedMovieWindow(fig, data, colors, neonColors, pauseTime, saveGif);
            else
                obj.renderStaticDashboard(fig, data, colors, neonColors);
            end
        end
    end

    methods (Access = private)
        function renderStaticDashboard(obj, fig, data, colors, neonColors)
            set(fig, 'Color', colors.Background);
            set(fig, 'Position', [40, 40, 1250, 800]);

            t = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'normal');
            sgtitle(t, 'Navigating the Likelihood Surface: Optimization Engines', ...
                'Color', colors.Text, 'FontSize', 14, 'FontWeight', 'bold');

            ax1 = nexttile(t, 1);
            obj.plotContourTerrain(ax1, data, colors, neonColors, "Full Surface");

            ax2 = nexttile(t, 2);
            obj.plotScoreConvergence(ax2, data.Histories, colors, neonColors);

            ax3 = nexttile(t, 3);
            obj.plotGradientNorm(ax3, data.Histories, colors, neonColors);

            ax4 = nexttile(t, 4);
            obj.plotStepNorm(ax4, data.Histories, colors, neonColors);
        end

        function renderDedicatedMovieWindow(obj, fig, data, colors, neonColors, pauseTime, saveGif)
            set(fig, 'Color', colors.Background);
            set(fig, 'Position', [100, 80, 950, 750]);

            ax = axes('Parent', fig, 'Position', [0.10, 0.10, 0.78, 0.82]);

            fields = {'Steepest', 'Newton', 'DFP', 'BFGS'};
            maxFrames = 1;
            for k = 1:numel(fields)
                fn = fields{k};
                if isfield(data.Histories, fn) && ~isempty(data.Histories.(fn)) && isfield(data.Histories.(fn), 'x')
                    maxFrames = max(maxFrames, size(data.Histories.(fn).x, 1));
                end
            end

            % Frame-by-frame animation loop on dedicated single axis
            for frame = 1:maxFrames
                cla(ax);
                
                % Build truncated history up to step 'frame'
                data_frame = data;
                data_frame.Histories = struct();
                for k = 1:numel(fields)
                    fn = fields{k};
                    if isfield(data.Histories, fn) && ~isempty(data.Histories.(fn))
                        h = data.Histories.(fn);
                        N = size(h.x, 1);
                        idx = min(frame, N);
                        h_sub = struct();
                        h_sub.x = h.x(1:idx, :);
                        h_sub.f = h.f(1:idx);
                        h_sub.grad_norm = h.grad_norm(1:idx);
                        h_sub.step_norm = h.step_norm(1:idx);
                        data_frame.Histories.(fn) = h_sub;
                    end
                end

                stepTitle = sprintf('Likelihood Optimization Trajectory — Iteration %d / %d', frame-1, maxFrames-1);
                obj.plotContourTerrain(ax, data_frame, colors, neonColors, stepTitle);

                drawnow;

                % Optional GIF Exporter
                if saveGif ~= ""
                    f_img = getframe(fig);
                    [im, cm] = rgb2ind(f_img.cdata, 256);
                    if frame == 1
                        imwrite(im, cm, saveGif, 'gif', 'Loopcount', inf, 'DelayTime', pauseTime);
                    else
                        imwrite(im, cm, saveGif, 'gif', 'WriteMode', 'append', 'DelayTime', pauseTime);
                    end
                end

                pause(pauseTime);
            end
        end

        function plotContourTerrain(obj, ax, data, colors, neonColors, titleStr)
            hold(ax, 'on');

            % 1. Custom Smooth Dark Surface Colormap (Navy -> Deep Purple -> Plum -> Warm Copper)
            r = [linspace(0.06, 0.20, 80), linspace(0.20, 0.55, 96), linspace(0.55, 0.90, 80)]';
            g = [linspace(0.06, 0.10, 80), linspace(0.10, 0.20, 96), linspace(0.20, 0.45, 80)]';
            b = [linspace(0.14, 0.30, 80), linspace(0.30, 0.35, 96), linspace(0.35, 0.15, 80)]';
            oceanDarkMap = [r, g, b];

            % 2. Robust Color Limit Clipping for Likelihood Surface
            L = data.LGrid;
            L_max = max(L(:));
            c_min = L_max - 500;
            c_max = L_max;

            % 3. Render filled 2D log-likelihood contours
            [~, h_c] = contourf(ax, data.PhiGrid, data.ThetaGrid, L, 35, ...
                'LineColor', [0.15 0.15 0.22], 'LineWidth', 0.4);
            h_c.Annotation.LegendInformation.IconDisplayStyle = 'off'; % Hide background mesh from legend!

            colormap(ax, oceanDarkMap);
            caxis(ax, [c_min, c_max]);

            cb = colorbar(ax);
            cb.Color = colors.Text;
            cb.Label.String = 'Log-Likelihood Score L(\phi, \theta)';
            cb.Label.Color = colors.Text;

            % Explicit Legend Items Collection
            legHandles = [];
            legLabels = {};

            % 4. Plot initial starting coordinate and true peak
            h_start = plot(ax, data.StartParam(1), data.StartParam(2), 'o', ...
                'MarkerSize', 11, 'MarkerFaceColor', [0.1 0.8 1.0], 'MarkerEdgeColor', [1 1 1], ...
                'LineWidth', 2.0);
            legHandles(end+1) = h_start;
            legLabels{end+1} = 'Start \theta^{(0)}';

            h_true = plot(ax, data.TrueParam(1), data.TrueParam(2), 'p', ...
                'MarkerSize', 16, 'MarkerFaceColor', [1.0 0.15 0.15], 'MarkerEdgeColor', [1 1 1], ...
                'LineWidth', 2.0);
            legHandles(end+1) = h_true;
            legLabels{end+1} = 'True MLE \theta^*';

            % 5. Overlay trajectory paths for each solver
            hist = data.Histories;
            fields = {'Steepest', 'Newton', 'DFP', 'BFGS'};
            labels = {'Steepest Ascent', 'Newton-Raphson', 'DFP Quasi-Newton', 'BFGS Quasi-Newton'};

            for k = 1:numel(fields)
                fn = fields{k};
                if isfield(hist, fn) && ~isempty(hist.(fn)) && isfield(hist.(fn), 'x') && ~isempty(hist.(fn).x)
                    x_path = hist.(fn).x;
                    col = neonColors.(fn);
                    
                    % Path line with contrasting dark outline
                    h_path = plot(ax, x_path(:,1), x_path(:,2), '-o', ...
                        'Color', col, 'LineWidth', 2.6, 'MarkerSize', 5.5, ...
                        'MarkerFaceColor', col, 'MarkerEdgeColor', [0.05 0.05 0.05]);
                    
                    % Highlight current active front head with crisp white halo
                    plot(ax, x_path(end,1), x_path(end,2), 'o', ...
                        'MarkerSize', 9.5, 'MarkerFaceColor', col, 'MarkerEdgeColor', [1 1 1], 'LineWidth', 2.0);
                    
                    legHandles(end+1) = h_path;
                    legLabels{end+1} = labels{k};
                end
            end

            xlabel(ax, 'Autoregressive Parameter \phi', 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
            ylabel(ax, 'Moving Average Parameter \theta', 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
            title(ax, titleStr, 'Color', colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
            
            obj.applyStyle(ax, colors);
            xlim(ax, [min(data.PhiGrid(:)), max(data.PhiGrid(:))]);
            ylim(ax, [min(data.ThetaGrid(:)), max(data.ThetaGrid(:))]);

            if ~isempty(legHandles)
                legend(ax, legHandles, legLabels, 'Location', 'northeast', ...
                    'TextColor', colors.Text, 'Color', [colors.Background 0.88], 'EdgeColor', [0.4 0.4 0.4]);
            end
            hold(ax, 'off');
        end

        function plotScoreConvergence(obj, ax, histories, colors, neonColors)
            hold(ax, 'on');

            fields = {'Steepest', 'Newton', 'DFP', 'BFGS'};
            labels = {'Steepest Ascent', 'Newton-Raphson', 'DFP', 'BFGS'};

            for k = 1:numel(fields)
                fn = fields{k};
                if isfield(histories, fn) && ~isempty(histories.(fn)) && isfield(histories.(fn), 'f')
                    f_val = histories.(fn).f;
                    iters = 0:(length(f_val)-1);
                    col = neonColors.(fn);
                    plot(ax, iters, f_val, '-o', ...
                        'Color', col, 'LineWidth', 2, 'MarkerSize', 4, ...
                        'MarkerFaceColor', col, 'DisplayName', labels{k});
                end
            end

            xlabel(ax, 'Iteration m', 'Color', colors.Text);
            ylabel(ax, 'Log-Likelihood Score L^{(m)}', 'Color', colors.Text);
            title(ax, 'Log-Likelihood Convergence Trajectory', 'Color', colors.Text, 'FontWeight', 'bold');
            obj.applyStyle(ax, colors);
            legend(ax, 'Location', 'southeast', 'TextColor', colors.Text, 'Color', [colors.Background 0.8], 'EdgeColor', 'none');
            hold(ax, 'off');
        end

        function plotGradientNorm(obj, ax, histories, colors, neonColors)
            hold(ax, 'on');

            fields = {'Steepest', 'Newton', 'DFP', 'BFGS'};
            labels = {'Steepest Ascent', 'Newton-Raphson', 'DFP', 'BFGS'};

            for k = 1:numel(fields)
                fn = fields{k};
                if isfield(histories, fn) && ~isempty(histories.(fn)) && isfield(histories.(fn), 'grad_norm')
                    g_norm = histories.(fn).grad_norm;
                    iters = 0:(length(g_norm)-1);
                    col = neonColors.(fn);
                    semilogy(ax, iters, g_norm + 1e-16, '-o', ...
                        'Color', col, 'LineWidth', 2, 'MarkerSize', 4, ...
                        'MarkerFaceColor', col, 'DisplayName', labels{k});
                end
            end

            xlabel(ax, 'Iteration m', 'Color', colors.Text);
            ylabel(ax, 'Gradient Norm ||g^{(m)}|| (log scale)', 'Color', colors.Text);
            title(ax, 'Gradient Norm Decay Rate', 'Color', colors.Text, 'FontWeight', 'bold');
            obj.applyStyle(ax, colors);
            legend(ax, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', [colors.Background 0.8], 'EdgeColor', 'none');
            hold(ax, 'off');
        end

        function plotStepNorm(obj, ax, histories, colors, neonColors)
            hold(ax, 'on');

            fields = {'Steepest', 'Newton', 'DFP', 'BFGS'};
            labels = {'Steepest Ascent', 'Newton-Raphson', 'DFP', 'BFGS'};

            for k = 1:numel(fields)
                fn = fields{k};
                if isfield(histories, fn) && ~isempty(histories.(fn)) && isfield(histories.(fn), 'step_norm')
                    s_norm = histories.(fn).step_norm;
                    iters = 0:(length(s_norm)-1);
                    col = neonColors.(fn);
                    semilogy(ax, iters, s_norm + 1e-16, '-o', ...
                        'Color', col, 'LineWidth', 2, 'MarkerSize', 4, ...
                        'MarkerFaceColor', col, 'DisplayName', labels{k});
                end
            end

            xlabel(ax, 'Iteration m', 'Color', colors.Text);
            ylabel(ax, 'Step Distance ||\Delta\theta^{(m)}|| (log scale)', 'Color', colors.Text);
            title(ax, 'Step Size Trajectory', 'Color', colors.Text, 'FontWeight', 'bold');
            obj.applyStyle(ax, colors);
            legend(ax, 'Location', 'northeast', 'TextColor', colors.Text, 'Color', [colors.Background 0.8], 'EdgeColor', 'none');
            hold(ax, 'off');
        end

        function applyStyle(~, ax, colors)
            set(ax, 'Color', colors.Background, ...
                    'XColor', colors.Text, ...
                    'YColor', colors.Text, ...
                    'GridColor', colors.Grid, ...
                    'GridAlpha', 0.4, ...
                    'LineWidth', 1.2);
            grid(ax, 'on');
        end
    end
end
