classdef Surface < OceanViz.Strategies.BaseStrategy
    % SURFACE Universal Strategy for 2D/3D Manifolds and Probability Maps.
    % 
    % This class implements the "Pluggable Projection" pattern, where the 
    % actual geometric rendering (2D vs 3D) is delegated to specialized 
    % projection objects.
    
    methods
        function plot(obj, ax, data, options)
            % PLOT Executes the manifold rendering pipeline.
            
            arguments
                obj, ax, data, options
            end
            
            % 1. Data Normalization
            if isfield(data, 'X'); gridX = data.X; else; gridX = data.GridX1; end
            if isfield(data, 'Y'); gridY = data.Y; else; gridY = data.GridX2; end
            if isfield(data, 'Z'); gridZ = data.Z; else; gridZ = data.ProbMap; end
            
            % TENSOR CHECK: If ProbMap is flattened (N_grid x K), reshape it to (H x W x K)
            if ismatrix(gridZ) && size(gridZ, 1) == numel(gridX) && size(gridZ, 2) > 1
                H = size(gridX, 1); W = size(gridX, 2); K = size(gridZ, 2);
                gridZ = reshape(gridZ, H, W, K);
            end
            
            colors = obj.Parent.Colors;
            hold(ax, 'on');
            
            % 2. Resolve Projection Object (Action as Objective)
            projection = options.Projection;
            if ischar(projection) || isstring(projection)
                if strcmp(projection, '3D')
                    projection = OceanViz.Projections.Manifold();
                else
                    projection = OceanViz.Projections.Planar();
                end
            end
            
            % 2b. Apply ZMode (Interpretation Mode)
            if isfield(options, 'ZMode') && isprop(projection, 'ZMode')
                projection.ZMode = options.ZMode;
            end
            
            % 3. Delegate Rendering Action
            projection.project(obj, ax, gridX, gridY, gridZ, data, options, colors);
            
            % 4. Common Decoration
            grid(ax, 'on');
            hold(ax, 'off');
        end
        
        function drawScatter(obj, ax, data, dim)
            % SHARED SCATTER LOGIC
            % Used by projection objects to maintain aesthetic consistency.
            colors = obj.Parent.Colors;
            
            % Classification Points
            if isfield(data, 'RawX') && isfield(data, 'RawY') && isfield(data, 'RawC')
                uniqueC = unique(data.RawC);
                palette = obj.Parent.getColorSequence(length(uniqueC));
                markers = ['o', 's', '^', 'd', 'v', 'p', 'h'];
                
                for i = 1:length(uniqueC)
                    idx = (data.RawC == uniqueC(i));
                    m = markers(mod(i-1, length(markers))+1);
                    
                    if dim == 3
                        % Handle heights: defaults to class index if RawZ missing
                        z_val = ones(sum(idx), 1) * uniqueC(i); 
                        if isfield(data, 'RawZ'); z_val = data.RawZ(idx); end
                        
                        scatter3(ax, data.RawX(idx), data.RawY(idx), z_val, 45, ...
                            m, 'Filled', 'MarkerFaceColor', palette(i,:), ...
                            'MarkerEdgeColor', colors.Background);
                    else
                        scatter(ax, data.RawX(idx), data.RawY(idx), 45, ...
                            m, 'Filled', 'MarkerFaceColor', palette(i,:), ...
                            'MarkerEdgeColor', colors.Background, 'MarkerFaceAlpha', 0.9);
                    end
                end
            
            % Simple Regression Points
            elseif isfield(data, 'RawX') && isfield(data, 'RawY') && isfield(data, 'RawZ')
                if dim == 3
                    scatter3(ax, data.RawX, data.RawY, data.RawZ, 40, 'o', 'Filled', ...
                        'MarkerFaceColor', colors.Scatter, 'MarkerEdgeColor', colors.Background);
                else
                    scatter(ax, data.RawX, data.RawY, 30, 'o', 'Filled', ...
                        'MarkerFaceColor', colors.Scatter, 'MarkerEdgeColor', colors.Background);
                end
            end
        end
    end
end