classdef ScatterInterpStrategy < OceanViz.Strategies.BaseStrategy
    % SCATTERINTERPSTRATEGY  OceanViz strategy for scattered interpolation.
    %
    %   Renders a multi-panel dashboard for Ex 4.1 scattered data interpolation:
    %     Panel 1 (1D) or Panel 1 (2D):
    %       Ground truth function + scatter training samples
    %     Panel 2:
    %       Fitted interpolation surface/curve
    %     Panel 3:
    %       Pointwise absolute error (heatmap for 2D, error ribbon for 1D)
    %     Panel 4 (optional):
    %       K-Fold CV RMSE curve (if CVResult is provided)
    %
    %   Called via OceanViz.Core:
    %     viz = OceanViz.Core('Theme', 'Dark');
    %     viz.plot(ax, 'ScatterInterp', data, 'Title', 'Ex 4.1 RBF');
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data    struct
                options struct = struct()
            end

            % Unpack required fields
            X_tr   = data.X_train;
            Y_tr   = data.Y_train;
            model  = data.model;
            f_true = data.f_true;
            dim    = data.dim;
            lo     = data.domain(1);
            hi     = data.domain(2);
            ngrid  = data.ngrid;
            hasCV  = isfield(data, 'CVResult') && ~isempty(data.CVResult);

            % Extract title if present
            titleStr = "Scattered Data Interpolation";
            if isfield(options, 'Title') && options.Title ~= "" && options.Title ~= "Analysis"
                titleStr = options.Title;
            elseif isfield(data, 'Title') && data.Title ~= ""
                titleStr = data.Title;
            end

            % Determine number of panels
            nPanels = 3 + hasCV;

            % We expect ax to be a parent figure — build a tiled layout
            fig = ax.Parent;
            delete(ax);   % remove the placeholder axis

            t = tiledlayout(fig, 1, nPanels, ...
                'TileSpacing', 'compact', 'Padding', 'normal');
            
            sgtitle_str = sprintf('%s  [%s]', titleStr, model.getName());
            title(t, sgtitle_str, 'Color', obj.Parent.Colors.Text, ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

            % Retrieve theme colours
            colors = obj.getColors_();

            if dim == 1
                obj.render1D_(t, X_tr, Y_tr, model, f_true, ...
                    lo, hi, ngrid, colors, hasCV, data);
            else
                obj.render2D_(t, X_tr, Y_tr, model, f_true, ...
                    lo, hi, ngrid, colors, hasCV, data);
            end
        end
    end

    methods (Access = private)

        % ── 1-D rendering ─────────────────────────────────────────────────
        function render1D_(obj, t, X_tr, Y_tr, model, f_true, ...
                lo, hi, ng, colors, hasCV, data)

            xg = linspace(lo, hi, ng)';
            yg_true = f_true(xg);
            yg_hat  = model.predict(xg);
            err     = abs(yg_hat - yg_true);

            % Shared Y bounds for 1D comparisons
            y_all = [yg_true; yg_hat; Y_tr];
            ymin  = min(y_all); ymax = max(y_all);
            pad   = 0.05 * max(ymax - ymin, 1e-6);
            ylims = [ymin - pad, ymax + pad];

            % Panel 1 — Ground truth + training samples
            ax1 = nexttile(t);
            obj.applyAxStyle_(ax1, colors);
            plot(ax1, xg, yg_true, '-', 'Color', colors.accent, ...
                'LineWidth', 2, 'DisplayName', 'Ground truth');
            hold(ax1, 'on');
            scatter(ax1, X_tr(:,1), Y_tr, 40, colors.scatter, 'o', ...
                'filled', 'MarkerEdgeColor', 'none', 'DisplayName', 'Train samples');
            legend(ax1, 'Location', 'best', 'TextColor', colors.text, ...
                'Color', colors.bg, 'EdgeColor', colors.grid);
            title(ax1, 'Ground Truth + Samples', 'Color', colors.text, 'FontSize', 11);
            xlabel(ax1, 'x'); ylabel(ax1, 'f(x)');
            ylim(ax1, ylims);

            % Panel 2 — Fitted interpolation
            ax2 = nexttile(t);
            obj.applyAxStyle_(ax2, colors);
            plot(ax2, xg, yg_true, '--', 'Color', colors.accent, ...
                'LineWidth', 1.2, 'DisplayName', 'Ground truth');
            hold(ax2, 'on');
            plot(ax2, xg, yg_hat, '-',  'Color', colors.fit, ...
                'LineWidth', 2,   'DisplayName', model.getName());
            scatter(ax2, X_tr(:,1), Y_tr, 30, colors.scatter, 'o', 'filled', ...
                'MarkerEdgeColor', 'none', 'DisplayName', 'Samples');
            legend(ax2, 'Location', 'best', 'TextColor', colors.text, ...
                'Color', colors.bg, 'EdgeColor', colors.grid, 'Interpreter', 'none');
            title(ax2, sprintf('Fitted: %s', model.getName()), ...
                'Color', colors.text, 'FontSize', 11, 'Interpreter', 'none');
            xlabel(ax2, 'x'); ylabel(ax2, '\hat{f}(x)');
            ylim(ax2, ylims);

            % Panel 3 — Absolute error ribbon
            ax3 = nexttile(t);
            obj.applyAxStyle_(ax3, colors);
            area(ax3, xg, err, 'FaceColor', colors.error, 'FaceAlpha', 0.6, ...
                'EdgeColor', 'none', 'DisplayName', '|error|');
            legend(ax3, 'Location', 'best', 'TextColor', colors.text, ...
                'Color', colors.bg, 'EdgeColor', colors.grid);
            title(ax3, sprintf('|Error| (RMSE=%.4f)', sqrt(mean(err.^2))), ...
                'Color', colors.text, 'FontSize', 11);
            xlabel(ax3, 'x'); ylabel(ax3, '|f - \hat{f}|');

            % Panel 4 — CV curve (optional)
            if hasCV
                ax4 = nexttile(t);
                obj.applyAxStyle_(ax4, colors);
                data.CVResult.plotCurve(ax4);
            end
        end

        % ── 2-D rendering ─────────────────────────────────────────────────
        function render2D_(obj, t, X_tr, Y_tr, model, f_true, ...
                lo, hi, ng, colors, hasCV, data)

            xv  = linspace(lo, hi, ng);
            [Gx, Gy] = meshgrid(xv, xv);
            Xgrid = [Gx(:), Gy(:)];
            Ztrue = reshape(f_true(Xgrid), ng, ng);
            Zhat  = reshape(model.predict(Xgrid), ng, ng);
            Zerr  = abs(Zhat - Ztrue);

            % Shared Z and Color limits across Panel 1 and Panel 2
            z_all = [Ztrue(:); Zhat(:); Y_tr];
            zmin  = min(z_all); zmax = max(z_all);
            pad   = 0.05 * max(zmax - zmin, 1e-6);
            zlims = [zmin - pad, zmax + pad];
            clims = [zmin, zmax];

            % Panel 1 — Ground truth surface
            ax1 = nexttile(t);
            obj.applyAxStyle_(ax1, colors);
            surf(ax1, Gx, Gy, Ztrue, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
            hold(ax1, 'on');
            scatter3(ax1, X_tr(:,1), X_tr(:,2), Y_tr, 40, ...
                colors.scatter, 'filled', 'MarkerEdgeColor', 'w');
            colormap(ax1, 'parula');
            title(ax1, 'Ground Truth', 'Color', colors.text, 'FontSize', 11);
            xlabel(ax1,'x_1'); ylabel(ax1,'x_2'); zlabel(ax1,'f');
            legend(ax1, {'Surface', 'Train pts'}, 'Location', 'northeast', ...
                'TextColor', colors.text, 'Color', colors.bg, 'EdgeColor', colors.grid);
            view(ax1, -37.5, 30);
            zlim(ax1, zlims);
            set(ax1, 'CLim', clims);

            % Panel 2 — Fitted surface
            ax2 = nexttile(t);
            obj.applyAxStyle_(ax2, colors);
            surf(ax2, Gx, Gy, Zhat, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
            hold(ax2, 'on');
            scatter3(ax2, X_tr(:,1), X_tr(:,2), Y_tr, 40, ...
                colors.scatter, 'filled', 'MarkerEdgeColor', 'w');
            colormap(ax2, 'parula');
            title(ax2, sprintf('Fitted: %s', model.getName()), ...
                'Color', colors.text, 'FontSize', 11, 'Interpreter', 'none');
            xlabel(ax2,'x_1'); ylabel(ax2,'x_2'); zlabel(ax2,'\hat{f}');
            legend(ax2, {'Fit', 'Train pts'}, 'Location', 'northeast', ...
                'TextColor', colors.text, 'Color', colors.bg, 'EdgeColor', colors.grid);
            view(ax2, -37.5, 30);
            zlim(ax2, zlims);
            set(ax2, 'CLim', clims);

            % Panel 3 — Absolute error heatmap
            ax3 = nexttile(t);
            obj.applyAxStyle_(ax3, colors);
            imagesc(ax3, xv, xv, Zerr);
            colormap(ax3, 'hot');
            cb = colorbar(ax3);
            cb.Color = colors.text;
            set(ax3, 'YDir', 'normal');
            hold(ax3, 'on');
            scatter(ax3, X_tr(:,1), X_tr(:,2), 25, 'w', 'filled', ...
                'MarkerEdgeColor', 'k', 'DisplayName', 'Train pts');
            legend(ax3, 'Location', 'northeast', 'TextColor', colors.text, ...
                'Color', colors.bg, 'EdgeColor', colors.grid);
            title(ax3, sprintf('|Error| (RMSE=%.4f)', sqrt(mean(Zerr(:).^2))), ...
                'Color', colors.text, 'FontSize', 11);
            xlabel(ax3,'x_1'); ylabel(ax3,'x_2');

            % Panel 4 — CV curve (optional)
            if hasCV
                ax4 = nexttile(t);
                obj.applyAxStyle_(ax4, colors);
                data.CVResult.plotCurve(ax4);
            end
        end

        % ── Shared helpers ────────────────────────────────────────────────
        function applyAxStyle_(~, ax, colors)
            set(ax, 'Color', colors.bg, 'XColor', colors.text, ...
                'YColor', colors.text, 'ZColor', colors.text, ...
                'GridColor', colors.grid, 'GridAlpha', 0.4, ...
                'LineWidth', 1.1, 'FontName', 'Helvetica');
            grid(ax, 'on');
        end

        function colors = getColors_(obj)
            if ~isempty(obj.Parent) && isfield(obj.Parent, 'Colors')
                colors.bg   = obj.Parent.Colors.Background;
                colors.text = obj.Parent.Colors.Text;
                colors.grid = obj.Parent.Colors.Grid;
            else
                colors.bg   = [0.09 0.09 0.12];
                colors.text = [0.90 0.90 0.90];
                colors.grid = [0.30 0.30 0.35];
            end
            colors.accent  = [0.20 0.75 1.00];   % cyan — ground truth
            colors.fit     = [1.00 0.65 0.10];   % amber — fit
            colors.scatter = [0.95 0.35 0.50];   % rose  — samples
            colors.error   = [1.00 0.35 0.35];   % red   — error
        end
    end
end
