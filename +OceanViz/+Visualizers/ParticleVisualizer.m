classdef ParticleVisualizer < OceanViz.Visualizers.IVisualizer
    % PARTICLEVISUALIZER  Animates a "particle" tracking the current state.
    %
    % Renders a simulation as a moving point (particle) on the physical
    % space defined by the ISpatialMapper. Draws:
    %   - Background geometry (grid lines or graph edges) once at init
    %   - A bright "current position" marker at the active state
    %   - A faded trajectory trail showing recent history
    %
    % Works with any ISpatialMapper (GridMapper2D, GraphMapper, etc.).
    % The same ParticleVisualizer code renders on a 2D grid, a graph
    % node layout, or a 2D PCA embedding — the mapper handles the geometry.
    %
    % Usage:
    %   mc     = OceanMath.Information.Processes.MarkovChain(P);
    %   sim    = OceanMath.Simulation.MarkovSimulator(mc);
    %   hist   = OceanMath.Simulation.SimulationHistory(sim);
    %   mapper = OceanViz.Adapters.GridMapper2D(4, 4);
    %   viz    = OceanViz.Visualizers.ParticleVisualizer(mapper, hist);
    %
    %   viz.attachToAxes(ax);          % pass any MATLAB axes or UIAxes
    %   viz.attachToSimulator(sim);    % now every step auto-updates the plot
    %   sim.runFor(100);               % watch it animate
    %
    % See also: OceanViz.Visualizers.IVisualizer
    %           OceanViz.Adapters.ISpatialMapper

    % ------------------------------------------------------------------ %
    properties (Access = private)
        % Graphic handle objects — cached for fast update (no redraw)
        HBackground_    % grid lines / graph edges (static, never redrawn)
        HTrail_         % faded trajectory trail (scatter handle)
        HParticle_      % bright current-position marker (scatter handle)
        HLabel_         % text label showing current state index

        TrailLength_    (1,1) double = 30    % number of past steps to show
    end

    properties (Access = private, Constant)
        % ---- Color palette ---- %
        COLOR_BG_GRID   = [0.88 0.90 0.93]  % light steel blue
        COLOR_TRAIL_RGB = [0.30 0.60 0.90]  % trail: ocean blue
        COLOR_PARTICLE  = [0.99 0.40 0.10]  % particle: warm orange
        COLOR_AXES_BG   = [0.10 0.12 0.16]  % dark canvas background
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = ParticleVisualizer(mapper, history, options)
            % PARTICLEVISUALIZER  Construct the visualizer with dependencies.
            %
            % INPUTS:
            %   mapper  - ISpatialMapper instance
            %   history - SimulationHistory instance
            %
            % Name-Value:
            %   TrailLength (double): steps of history to show. Default: 30.
            arguments
                mapper  OceanViz.Adapters.ISpatialMapper
                history OceanMath.Simulation.SimulationHistory
                options.TrailLength (1,1) double {mustBePositive} = 30
            end
            obj@OceanViz.Visualizers.IVisualizer(mapper, history);
            obj.TrailLength_ = options.TrailLength;
        end

        % ---- IVisualizer Implementation ---- %

        function initialize(obj, axesHandle)
            % INITIALIZE  Draw static background and create dynamic handles.
            cla(axesHandle, 'reset');
            hold(axesHandle, 'on');

            % Style the axes canvas
            axesHandle.Color           = obj.COLOR_AXES_BG;
            axesHandle.XColor          = [0.6 0.6 0.6];
            axesHandle.YColor          = [0.6 0.6 0.6];
            axesHandle.GridColor       = [0.3 0.3 0.3];
            axesHandle.GridAlpha       = 0.4;
            axesHandle.Box             = 'on';
            axesHandle.DataAspectRatio = [1 1 1];

            % Draw background geometry (mapper-specific)
            obj.drawBackground_(axesHandle);

            % Create dynamic graphic handles (scatter with empty data initially)
            % These are updated in-place: no redraw, just set XData/YData.
            obj.HTrail_ = scatter(axesHandle, NaN, NaN, 30, ...
                obj.COLOR_TRAIL_RGB, 'filled', ...
                'MarkerFaceAlpha', 0.25, ...
                'MarkerEdgeColor', 'none');

            obj.HParticle_ = scatter(axesHandle, NaN, NaN, 180, ...
                obj.COLOR_PARTICLE, 'filled', ...
                'MarkerEdgeColor', 'white', ...
                'LineWidth', 2.0);

            obj.HLabel_ = text(axesHandle, NaN, NaN, '', ...
                'Color', 'white', ...
                'FontSize', 9, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontWeight', 'bold');

            hold(axesHandle, 'off');

            % Render starting position
            obj.updateToStep(1);
        end

        function updateToStep(obj, t)
            % UPDATETOSTEP  Update particle and trail to time step t.
            %
            % O(TrailLength_) per call — no static elements are redrawn.
            % Uses XData/YData property updates on cached handles.
            if ~isvalid(obj.AxesHandle_)
                return;
            end

            nRecorded = obj.History_.getLength();
            t = max(1, min(t, nRecorded));

            % --- Current position ---
            curState  = obj.History_.getStateAtStep(t);
            curCoords = obj.Mapper_.mapState(curState);

            set(obj.HParticle_, 'XData', curCoords(1), 'YData', curCoords(2));
            set(obj.HLabel_,   'Position', [curCoords(1), curCoords(2)+0.35, 0], ...
                'String', sprintf('s=%d', curState));

            % --- Trail (past TrailLength_ steps) ---
            tStart = max(1, t - obj.TrailLength_);
            if tStart < t
                trailStates = obj.History_.getStateSequence();
                trailStates = trailStates(tStart:t-1);
                trailCoords = obj.Mapper_.mapSequence(trailStates);

                % Fade alpha proportionally to age (oldest = most transparent)
                n        = size(trailCoords, 1);
                alphaVec = linspace(0.05, 0.5, n)';

                set(obj.HTrail_, ...
                    'XData', trailCoords(:,1), ...
                    'YData', trailCoords(:,2), ...
                    'AlphaData', alphaVec, ...
                    'MarkerFaceAlpha', 'flat');
            end

            drawnow limitrate;   % flush the graphics queue without blocking
        end

        function clear(obj)
            % CLEAR  Reset dynamic elements to empty/invisible.
            if isvalid(obj.AxesHandle_)
                set(obj.HParticle_, 'XData', NaN, 'YData', NaN);
                set(obj.HTrail_,    'XData', NaN, 'YData', NaN);
                set(obj.HLabel_,    'String', '');
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function drawBackground_(obj, ax)
            % DRAWBACKGROUND_  Render mapper-appropriate static geometry.
            %
            % For GridMapper2D: draws navigable edges between nodes only.
            %   MATLAB's own grid lines and tick marks are hidden so they
            %   don't create a confusing double-grid effect.
            % For GraphMapper : draws edges between states.
            % For others      : draws state positions as faint dots.

            if isa(obj.Mapper_, 'OceanViz.Adapters.GridMapper2D')
                % --- Clean canvas: hide axes decoration ---
                ax.XTick    = [];
                ax.YTick    = [];
                ax.XColor   = 'none';
                ax.YColor   = 'none';
                ax.Box      = 'off';

                N      = obj.Mapper_.getNumStates();
                allPos = obj.Mapper_.mapSequence((1:N)');
                xs     = unique(allPos(:,1));
                ys     = unique(allPos(:,2));

                margin = 0.4;   % small padding around the outermost nodes
                xlim(ax, [min(xs)-margin, max(xs)+margin]);
                ylim(ax, [min(ys)-margin, max(ys)+margin]);

                % -- Draw only the navigable edges (horizontal + vertical) --
                edgeColor = [0.40 0.45 0.55];   % muted blue-grey edges
                for k = 1:N
                    c = allPos(k, 1);  r = allPos(k, 2);
                    % Right edge
                    right = allPos(:,1) == c + (xs(2)-xs(1)) & allPos(:,2) == r;
                    if any(right)
                        line(ax, [c, allPos(right,1)], [r, r], ...
                            'Color', edgeColor, 'LineWidth', 1.2);
                    end
                    % Up edge
                    above = allPos(:,1) == c & allPos(:,2) == r + (ys(2)-ys(1));
                    if any(above)
                        line(ax, [c, c], [r, allPos(above,2)], ...
                            'Color', edgeColor, 'LineWidth', 1.2);
                    end
                end

                % -- Node dots --
                scatter(ax, allPos(:,1), allPos(:,2), 50, obj.COLOR_BG_GRID, ...
                    'filled', 'MarkerFaceAlpha', 0.7);

            elseif isa(obj.Mapper_, 'OceanViz.Adapters.GraphMapper')
                % --- Graph edges & labels ---
                N      = obj.Mapper_.getNumStates();
                allPos = obj.Mapper_.mapSequence((1:N)');
                A      = obj.Mapper_.getAdjacency();
                edgeColor = [0.45 0.50 0.60];

                if ~isempty(A)
                    [rows, cols] = find(A);
                    for i = 1:length(rows)
                        u = rows(i); v = cols(i);
                        % Draw each edge once (assumes undirected or handles weight)
                        if u < v 
                            p1 = allPos(u, :); p2 = allPos(v, :);
                            line(ax, [p1(1) p2(1)], [p1(2) p2(2)], ...
                                'Color', edgeColor, 'LineWidth', 1.2, ...
                                'HandleVisibility', 'off');
                            
                            % Draw weight label at midpoint
                            mid = (p1 + p2)/2;
                            w   = A(u,v);
                            text(ax, mid(1), mid(2), sprintf('%.1f', w), ...
                                'Color', [0.7 0.7 0.7], 'FontSize', 8, ...
                                'BackgroundColor', obj.COLOR_AXES_BG, ...
                                'Margin', 0.5, ...
                                'HorizontalAlignment', 'center', ...
                                'VerticalAlignment', 'middle');
                        end
                    end
                end

                % State dots
                scatter(ax, allPos(:,1), allPos(:,2), 80, obj.COLOR_BG_GRID, ...
                    'filled', 'MarkerFaceAlpha', 0.85);

                % Label nodes (s1, s2, ...)
                for k = 1:N
                    text(ax, allPos(k,1), allPos(k,2), sprintf('s%d', k), ...
                        'Color', [0.1 0.1 0.1], 'FontSize', 8, ...
                        'FontWeight', 'bold', ...
                        'HorizontalAlignment', 'center');
                end
                
                margin = 0.8;
                xlim(ax, [min(allPos(:,1))-margin, max(allPos(:,1))+margin]);
                ylim(ax, [min(allPos(:,2))-margin, max(allPos(:,2))+margin]);

            else
                % --- Generic: state positions as faint markers ---
                N      = obj.Mapper_.getNumStates();
                allPos = obj.Mapper_.mapSequence((1:N)');
                scatter(ax, allPos(:,1), allPos(:,2), 30, obj.COLOR_BG_GRID, ...
                    'filled', 'MarkerFaceAlpha', 0.4);
                xlim(ax, [min(allPos(:,1))-1, max(allPos(:,1))+1]);
                ylim(ax, [min(allPos(:,2))-1, max(allPos(:,2))+1]);
            end
        end
    end
end
