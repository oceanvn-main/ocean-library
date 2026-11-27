classdef DeblockingStrategy < OceanViz.Strategies.BaseStrategy
    % DEBLOCKINGSTRATEGY  OceanViz 6-panel dashboard for POCS deblocking experiments.
    %
    %   Data Contract (data struct fields):
    %     data.Original      — (H x W x 3) double Ground truth image
    %     data.Compressed    — (H x W x 3) double JPEG-degraded image
    %     data.Deblocked     — (H x W x 3) double POCS deblocked output
    %     data.GuidedOut     — (H x W x 3) double Guided-upsampled output (optional)
    %     data.History       — struct with .RMSE field (convergence curve)
    %     data.Metrics       — struct array with .Label, .PSNR, .SSIM fields
    %     data.QualityFactor — scalar JPEG quality factor used
    %
    %   Panel layout (2 rows x 3 cols):
    %     [1] Ground Truth      [2] JPEG Compressed    [3] POCS Deblocked
    %     [4] Error Heatmap     [5] Guided Upsampled   [6] POCS Convergence Curve
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax       % ignored — strategy creates its own figure layout
                data     struct
                options  struct
            end

            colors = obj.Parent.Colors;

            % Validate required fields
            required = {'Original', 'Compressed', 'Deblocked', 'History'};
            for k = 1:numel(required)
                if ~isfield(data, required{k})
                    error('DeblockingStrategy:MissingField', ...
                        'data must contain field: %s', required{k});
                end
            end

            % Create figure
            fig = obj.Parent.createFigure('Ex 4.3 — Image Deblocking: POCS + Guided Restoration');
            set(fig, 'Position', [30, 30, 1400, 820]);

            bg = colors.Background;
            set(fig, 'Color', bg);

            % --- Panel 1: Ground Truth ---
            ax1 = subplot(2, 3, 1, 'Parent', fig);
            obj.showImage(ax1, data.Original, 'Ground Truth (Clean)', colors);

            % --- Panel 2: JPEG Compressed ---
            ax2 = subplot(2, 3, 2, 'Parent', fig);
            qf_str = '';
            if isfield(data, 'QualityFactor')
                qf_str = sprintf('  [Q=%d]', data.QualityFactor);
            end
            title2 = sprintf('JPEG Compressed%s', qf_str);
            obj.showImage(ax2, data.Compressed, title2, colors);

            % --- Panel 3: POCS Deblocked ---
            ax3 = subplot(2, 3, 3, 'Parent', fig);
            obj.showImage(ax3, data.Deblocked, 'POCS Deblocked', colors);

            % --- Panel 4: Absolute Error Heatmap (Compressed vs GT) ---
            ax4 = subplot(2, 3, 4, 'Parent', fig);
            err_compressed = mean(abs(double(data.Compressed) - double(data.Original)), 3);
            obj.showHeatmap(ax4, err_compressed, 'Error: Compressed vs GT', colors, 'hot');

            % --- Panel 5: Guided Upsampled (if available) ---
            ax5 = subplot(2, 3, 5, 'Parent', fig);
            if isfield(data, 'GuidedOut') && ~isempty(data.GuidedOut)
                err_pocs = mean(abs(double(data.Deblocked) - double(data.Original)), 3);
                obj.showHeatmap(ax5, err_pocs, 'Error: POCS vs GT', colors, 'hot');
            else
                obj.showImage(ax5, data.Deblocked, 'POCS Deblocked', colors);
            end

            % --- Panel 6: POCS Convergence Curve ---
            ax6 = subplot(2, 3, 6, 'Parent', fig);
            obj.plotConvergence(ax6, data.History, data, colors);
        end
    end

    methods (Access = private)
        function showImage(~, ax, img, titleStr, colors)
            % Normalize and display RGB image
            I = double(img);
            if max(I(:)) > 1.0
                I = I / 255.0;
            end
            I = max(0, min(1, I));
            imshow(I, 'Parent', ax);
            axis(ax, 'image');
            axis(ax, 'off');
            title(ax, titleStr, 'Color', colors.Text, ...
                'FontSize', 10, 'FontWeight', 'bold');
            set(ax, 'Color', colors.Background);
        end

        function showHeatmap(~, ax, err_map, titleStr, colors, cmap)
            imagesc(ax, err_map);
            axis(ax, 'image');
            axis(ax, 'off');
            colormap(ax, cmap);
            cb = colorbar(ax);
            cb.Color = colors.Text;
            cb.Label.String = 'Abs. Error';
            cb.Label.Color  = colors.Text;
            title(ax, titleStr, 'Color', colors.Text, ...
                'FontSize', 10, 'FontWeight', 'bold');
            set(ax, 'Color', colors.Background);
        end

        function plotConvergence(~, ax, history, data, colors)
            hold(ax, 'on');

            rmse = history.RMSE;
            iters = 1:numel(rmse);

            plot(ax, iters, rmse, '-o', ...
                'Color', colors.Primary, 'LineWidth', 2, ...
                'MarkerSize', 4, 'MarkerFaceColor', colors.Primary, ...
                'DisplayName', 'Relative Change');

            % Overlay a dashed convergence threshold line if history has Tolerance
            if isfield(data, 'Tolerance') && ~isempty(data.Tolerance)
                yline(ax, data.Tolerance, '--', ...
                    'Color', colors.Secondary, 'LineWidth', 1.5, ...
                    'Label', sprintf('Tol=%.0e', data.Tolerance), ...
                    'LabelHorizontalAlignment', 'left', ...
                    'FontSize', 8);
            end

            set(ax, 'Color', colors.Background, 'XColor', colors.Text, ...
                'YColor', colors.Text, 'GridColor', colors.Grid);
            grid(ax, 'on');
            xlabel(ax, 'Iteration', 'Color', colors.Text);
            ylabel(ax, 'Relative Change (‖Δf‖ / ‖f‖)', 'Color', colors.Text);
            title(ax, 'POCS Convergence', 'Color', colors.Text, ...
                'FontSize', 10, 'FontWeight', 'bold');

            % Quality metrics table as text legend below
            if isfield(data, 'Metrics') && ~isempty(data.Metrics)
                lines_str = {};
                for k = 1:numel(data.Metrics)
                    lines_str{end+1} = sprintf('%s: PSNR=%.1f dB  SSIM=%.3f', ...
                        data.Metrics(k).Label, data.Metrics(k).PSNR, data.Metrics(k).SSIM); %#ok<AGROW>
                end
                legend_str = strjoin(lines_str, '\n');
                text(ax, 0.02, 0.08, legend_str, ...
                    'Units', 'normalized', ...
                    'Color', colors.Text, ...
                    'FontSize', 7.5, ...
                    'VerticalAlignment', 'bottom', ...
                    'BackgroundColor', colors.Background, ...
                    'Interpreter', 'none');
            end

            hold(ax, 'off');
        end
    end
end
