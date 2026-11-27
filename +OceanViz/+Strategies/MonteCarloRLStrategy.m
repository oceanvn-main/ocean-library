classdef MonteCarloRLStrategy < OceanViz.Interfaces.IVizStrategy
    % MONTECARLORLDSTRATEGY OceanViz strategy for Chapter 5 Monte Carlo RL results.
    %
    % Renders all canonical Sutton & Barto Chapter 5 visualizations:
    %   - State-value function V(s) as 3D surfaces (Figs 5.1, 5.3)
    %   - Optimal policy heatmaps (hit/stick decision boundary, Fig 5.2)
    %   - Convergence curves (average return vs episodes)
    %   - 4-panel combined Blackjack dashboard
    %
    % MODES (pass via options.Mode):
    %   "ValueSurface3D"    - Two 3D V(s) surfaces (no-ace / usable-ace)
    %   "PolicyHeatmap"     - 2D hit/stick decision map
    %   "ConvergenceCurve"  - Avg return or RMS error vs episodes
    %   "Combined"          - Full 4-panel Blackjack dashboard (default)
    %
    % DATA CONTRACT (data struct fields):
    %   data.VTable         - (N x 1) state-value vector from MonteCarloSolver
    %   data.QTable         - (N x A) action-value matrix
    %   data.PolicyTable    - (N x 1) greedy action indices (1-based)
    %   data.Environment    - IEpisodeEnvironment instance (for state layout)
    %   data.History        - MonteCarloSolver.History struct (for convergence)
    %   data.AllStates      - (N x 3) state matrix [player_sum, dealer, ace]
    %
    % LEGEND CONTRACT: All legends are styled to the active OceanViz theme
    % (Background / Text / Grid colors) for dark-mode readability.
    %
    % Package:   OceanViz.Strategies
    % Reference: Sutton & Barto (2018) Ch.5 Figures 5.1-5.4

    methods
        function draw(obj, ax, data, options, theme)
            % DRAW Dispatches to the appropriate rendering subroutine.
            %
            % Inputs:
            %   ax      - MATLAB axes handle
            %   data    - data struct (see class doc for contract)
            %   options - struct; options.Mode selects rendering mode
            %   theme   - OceanViz theme struct (.Background, .Text, .Grid)

            mode = "Combined";
            if isfield(options, 'Mode') && ~isempty(options.Mode)
                mode = string(options.Mode);
            end

            switch mode
                case "ValueSurface3D"
                    obj.render_value_surfaces_(ax, data, options, theme);
                case "PolicyHeatmap"
                    obj.render_policy_heatmap_(ax, data, options, theme);
                case "ConvergenceCurve"
                    obj.render_convergence_(ax, data, options, theme);
                case "ISComparison"
                    obj.render_is_comparison_(ax, data, options, theme);
                case "Combined"
                    obj.render_combined_dashboard_(data, options, theme);
                otherwise
                    obj.render_combined_dashboard_(data, options, theme);
            end
        end
    end

    % -----------------------------------------------------------------------
    methods (Access = private)

        function render_value_surfaces_(~, ax, data, options, theme)
            % RENDER_VALUE_SURFACES_ 3D surface of V(s) for both ace conditions.
            %
            % Replicates Sutton & Barto Figs 5.1 (no usable ace) and 5.3.

            all_states = data.AllStates;   % (N x 3): [player_sum, dealer, usable_ace]
            V          = data.VTable;      % (N x 1)

            ps_range  = 12:21;   % player sum axis
            dc_range  = 1:10;    % dealer showing axis
            ua_flag   = 0;       % default: no usable ace panel

            if isfield(options, 'UsableAce')
                ua_flag = options.UsableAce;
            end

            title_str = sprintf('V(s) — %s Usable Ace', ...
                conditional_str_(ua_flag == 1, 'With', 'No'));
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end

            % Extract V values for the requested ace condition
            V_grid = nan(length(ps_range), length(dc_range));

            for row = 1:size(all_states, 1)
                ps = all_states(row, 1);
                dc = all_states(row, 2);
                ua = all_states(row, 3);
                if ua == ua_flag && ps >= 12 && ps <= 21 && dc >= 1 && dc <= 10
                    V_grid(ps - 11, dc) = V(row);
                end
            end

            [X, Y] = meshgrid(dc_range, ps_range);

            hold(ax, 'on');
            s_h = surf(ax, X, Y, V_grid, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.92);
            s_h.DisplayName = 'V(s)';

            colormap(ax, turbo);
            cb = colorbar(ax);
            cb.Label.String = 'State Value V(s)';
            cb.Label.Color  = theme.Text;
            cb.Color        = theme.Text;

            view(ax, [-30, 30]);
            grid(ax, 'on');
            set(ax, ...
                'Color',    theme.Background, ...
                'XColor',   theme.Text,       ...
                'YColor',   theme.Text,       ...
                'ZColor',   theme.Text,       ...
                'GridColor', theme.Grid       ...
            );

            xlabel(ax, 'Dealer Showing',  'Color', theme.Text, 'FontSize', 10);
            ylabel(ax, 'Player Sum',      'Color', theme.Text, 'FontSize', 10);
            zlabel(ax, 'V(s)',            'Color', theme.Text, 'FontSize', 10);
            title(ax, title_str, 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');
            xticks(ax, dc_range);
            yticks(ax, ps_range);
            xlim(ax, [1, 10]);
            ylim(ax, [12, 21]);

            legend(ax, 'Location', 'northeast', ...
                'TextColor', theme.Text, ...
                'Color',     theme.Background, ...
                'EdgeColor', 'none');
            hold(ax, 'off');
        end

        % -------------------------------------------------------------------
        function render_policy_heatmap_(~, ax, data, options, theme)
            % RENDER_POLICY_HEATMAP_ 2D decision boundary: hit vs stick.
            %
            % Replicates the policy panels of Sutton & Barto Fig 5.2.

            all_states  = data.AllStates;   % (N x 3)
            policy_acts = data.PolicyTable; % (N x 1) — 1=stick, 2=hit (1-based)

            ps_range = 12:21;
            dc_range = 1:10;
            ua_flag  = 0;

            if isfield(options, 'UsableAce')
                ua_flag = options.UsableAce;
            end

            title_str = sprintf('pi*(s) — %s Usable Ace', ...
                conditional_str_(ua_flag == 1, 'With', 'No'));
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end

            % Build policy grid (0 = stick, 1 = hit)
            pi_grid = nan(length(ps_range), length(dc_range));
            for row = 1:size(all_states, 1)
                ps = all_states(row, 1);
                dc = all_states(row, 2);
                ua = all_states(row, 3);
                if ua == ua_flag && ps >= 12 && ps <= 21 && dc >= 1 && dc <= 10
                    pi_grid(ps - 11, dc) = policy_acts(row) - 1; % Convert to 0/1
                end
            end

            hold(ax, 'on');
            imagesc(ax, dc_range, ps_range, pi_grid);
            set(ax, 'YDir', 'normal');

            % Colormap: cyan = stick, magenta = hit
            cmap = [0.05, 0.70, 0.80;   % stick (0) — teal
                    0.90, 0.20, 0.55];  % hit   (1) — rose
            colormap(ax, cmap);
            cb = colorbar(ax);
            cb.Ticks      = [0.25, 0.75];
            cb.TickLabels = {'STICK (0)', 'HIT (1)'};
            cb.Label.Color = theme.Text;
            cb.Color       = theme.Text;
            clim(ax, [0, 1]);

            grid(ax, 'on');
            set(ax, ...
                'Color',     theme.Background, ...
                'XColor',    theme.Text,       ...
                'YColor',    theme.Text,       ...
                'GridColor', theme.Grid        ...
            );

            xlabel(ax, 'Dealer Showing', 'Color', theme.Text, 'FontSize', 10);
            ylabel(ax, 'Player Sum',     'Color', theme.Text, 'FontSize', 10);
            title(ax, title_str, 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');
            xticks(ax, dc_range);
            yticks(ax, ps_range);

            % Legend entries
            h_stick = patch(ax, NaN, NaN, cmap(1,:));
            h_hit   = patch(ax, NaN, NaN, cmap(2,:));
            h_stick.DisplayName = 'STICK';
            h_hit.DisplayName   = 'HIT';
            legend(ax, [h_stick, h_hit], 'Location', 'northeast', ...
                'TextColor', theme.Text, ...
                'Color',     theme.Background, ...
                'EdgeColor', 'none');
            hold(ax, 'off');
        end

        % -------------------------------------------------------------------
        function render_convergence_(~, ax, data, options, theme)
            % RENDER_CONVERGENCE_ Average return vs. episodes convergence curve.

            history = data.History;

            title_str = sprintf('Convergence — %s', history.Algorithm);
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end

            hold(ax, 'on');
            h_ret = plot(ax, history.CheckPoints, history.AvgReturn, ...
                'Color', [0.20, 0.85, 0.90], 'LineWidth', 1.8);
            h_ret.DisplayName = 'Avg Return per Episode';

            % Smoothed trend line
            if numel(history.AvgReturn) > 10
                w    = ones(1, 5) / 5;
                smth = conv(history.AvgReturn, w, 'same');
                h_sm = plot(ax, history.CheckPoints, smth, ...
                    'Color', [1.0, 0.60, 0.10], 'LineWidth', 2.5, 'LineStyle', '--');
                h_sm.DisplayName = 'Smoothed (5-pt MA)';
            end

            yline(ax, 0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
                'DisplayName', 'Zero baseline');

            grid(ax, 'on');
            set(ax, ...
                'Color',     theme.Background, ...
                'XColor',    theme.Text,       ...
                'YColor',    theme.Text,       ...
                'GridColor', theme.Grid        ...
            );

            xlabel(ax, 'Episodes', 'Color', theme.Text, 'FontSize', 10);
            ylabel(ax, 'Average Return', 'Color', theme.Text, 'FontSize', 10);
            title(ax, title_str, 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');

            legend(ax, 'Location', 'southeast', ...
                'TextColor', theme.Text, ...
                'Color',     theme.Background, ...
                'EdgeColor', 'none');
            hold(ax, 'off');
        end

        % -------------------------------------------------------------------
        function render_combined_dashboard_(obj, data, options, theme)
            % RENDER_COMBINED_DASHBOARD_ 4-panel Blackjack dashboard.
            %
            % Layout:
            %   Panel 1 (top-left) : V(s) — No Usable Ace (3D surface)
            %   Panel 2 (top-right): V(s) — Usable Ace (3D surface)
            %   Panel 3 (bot-left) : Policy — No Usable Ace (heatmap)
            %   Panel 4 (bot-right): Convergence Curve

            title_str = 'Monte Carlo RL — Blackjack Dashboard';
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end

            fig = figure('Name', title_str, ...
                'Color',    theme.Background, ...
                'Position', [80, 80, 1400, 820]);

            % Panel 1: V(s) No Usable Ace
            ax1 = subplot(2, 2, 1, 'Parent', fig);
            opts1 = struct('UsableAce', 0, 'Title', 'V(s) — No Usable Ace');
            obj.render_value_surfaces_(ax1, data, opts1, theme);

            % Panel 2: V(s) Usable Ace
            ax2 = subplot(2, 2, 2, 'Parent', fig);
            opts2 = struct('UsableAce', 1, 'Title', 'V(s) — Usable Ace');
            obj.render_value_surfaces_(ax2, data, opts2, theme);

            % Panel 3: Policy heatmap (No Usable Ace — cleaner to show)
            ax3 = subplot(2, 2, 3, 'Parent', fig);
            opts3 = struct('UsableAce', 0, 'Title', 'pi*(s) — No Usable Ace');
            obj.render_policy_heatmap_(ax3, data, opts3, theme);

            % Panel 4: Convergence
            ax4 = subplot(2, 2, 4, 'Parent', fig);
            opts4 = struct('Title', sprintf('Convergence (%s)', data.History.Algorithm));
            obj.render_convergence_(ax4, data, opts4, theme);

            sgtitle(fig, title_str, ...
                'Color',      theme.Text, ...
                'FontSize',   14,         ...
                'FontWeight', 'bold');
        end

        % -------------------------------------------------------------------
        function render_is_comparison_(~, ax, data, options, theme)
            % RENDER_IS_COMPARISON_ Log-scale MSE comparison for Ordinary vs Weighted IS.
            %
            % Replicates Sutton & Barto Figure 5.3 (Example 5.4).

            episodes = data.Episodes;
            mse_ord  = data.MSE_Ordinary;
            mse_wei  = data.MSE_Weighted;

            title_str = 'Weighted vs. Ordinary Importance Sampling (Figure 5.3)';
            if isfield(options, 'Title') && ~isempty(options.Title)
                title_str = options.Title;
            end

            hold(ax, 'on');
            h_ord = semilogx(ax, episodes, mse_ord, 'Color', [0.30, 0.85, 0.40], 'LineWidth', 2.0);
            h_ord.DisplayName = 'Ordinary importance sampling';

            h_wei = semilogx(ax, episodes, mse_wei, 'Color', [0.95, 0.30, 0.35], 'LineWidth', 2.0);
            h_wei.DisplayName = 'Weighted importance sampling';

            yline(ax, 0, '--', 'Color', [0.40, 0.40, 0.45], 'LineWidth', 1.0, 'HandleVisibility', 'off');

            grid(ax, 'on');
            box(ax, 'on');
            set(ax, ...
                'Color',     theme.Background, ...
                'XColor',    theme.Text,       ...
                'YColor',    theme.Text,       ...
                'GridColor', theme.Grid,       ...
                'XScale',    'log'             ...
            );

            xlabel(ax, 'Episodes (log scale)', 'Color', theme.Text, 'FontSize', 11, 'FontWeight', 'bold');
            ylabel(ax, 'Mean square error (avg over 100 runs)', 'Color', theme.Text, 'FontSize', 11, 'FontWeight', 'bold');
            title(ax, title_str, 'Color', theme.Text, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
            xlim(ax, [1, max(episodes)]);
            
            max_y = max([max(mse_ord), max(mse_wei)]);
            top_y = min(max(max_y * 1.10, 2.5), 5.5);
            ylim(ax, [-0.05, top_y]);
            yticks(ax, 0:1:floor(top_y));

            legend(ax, [h_ord, h_wei], 'Location', 'northeast', ...
                'TextColor', theme.Text, ...
                'Color',     theme.Background, ...
                'EdgeColor', 'none', ...
                'FontSize',  10);
            hold(ax, 'off');
        end
    end
end

% ---------------------------------------------------------------------------
function s = conditional_str_(cond, a, b)
    if cond, s = a; else, s = b; end
end
