classdef SurfaceManifoldStrategy < OceanViz.Interfaces.IVizStrategy
    % SURFACEMANIFOLDSTRATEGY Concrete implementation for 3D Surface Plots.
    % NAMESPACE: OceanViz.Strategies
    
    methods
        function draw(obj, ax, data, options, theme)
            % DRAW Implementation of the IVizStrategy contract.
            
            % 1. Grid Generation
            gridData = obj.generateGrid(data.X);
            
            % 2. Map Grid back to Input Space
            if size(data.X, 2) > 2
                X_in = [gridData.gx(:) gridData.gy(:)] * gridData.coeff(:,1:2)' + mean(data.X);
            else
                X_in = [gridData.gx(:) gridData.gy(:)];
            end
            
            % 3. Mode Configuration
            [Z_surf, scatterData, plotOpts] = obj.configureMode(data, X_in, gridData, options, theme);
            
            % 4. Execute Rendering
            obj.renderManifold(ax, gridData, Z_surf, scatterData, plotOpts);
        end
    end
    
    methods (Access = private)
        
        function gridData = generateGrid(~, X)
            gridData = struct();
            if size(X, 2) > 2
                [coeff, score] = pca(X);
                gridData.X_vis = score(:, 1:2);
                gridData.coeff = coeff;
            else
                gridData.X_vis = X;
                gridData.coeff = eye(2);
            end
            res = 40; 
            [gridData.gx, gridData.gy] = meshgrid(...
                linspace(min(gridData.X_vis(:,1)), max(gridData.X_vis(:,1)), res), ...
                linspace(min(gridData.X_vis(:,2)), max(gridData.X_vis(:,2)), res));
        end
        
        function [Z, scat, opts] = configureMode(~, data, X_in, gridData, options, theme)
            PredictFcn = data.Strategy.PredictFcn;
            Model = data.Model;
            Level = data.Level;
            
            scat = struct('X', gridData.X_vis(:,1), 'Y', gridData.X_vis(:,2), 'Z', [], 'Color', [], 'Map', []);
            
            % 1. CRITICAL FIX: Pass 'ComplexityLevel' to opts so layerSurface can read it
            opts = struct('Theme', theme, ...
                          'ShowGradient', isfield(options, 'ShowGradient') && options.ShowGradient, ...
                          'GroundTruthFcn', [], ...
                          'ComplexityLevel', Level); 
            
            if isfield(options, 'GroundTruthFcn'), opts.GroundTruthFcn = options.GroundTruthFcn; end
            
            if isfield(data, 'Classes') && ~isempty(data.Classes) 
                % === CLASSIFICATION ===
                probs = PredictFcn(Model, X_in, Level);
                epsilon = 1e-10;
                titleStr = ""; if isfield(options, 'Title'), titleStr = options.Title; end
                isEntropy = contains(lower(titleStr), 'entropy') || contains(lower(titleStr), 'uncertainty');
                
                if isEntropy
                    H_vals = -sum(probs .* log(probs + epsilon), 2);
                    Z = reshape(H_vals, size(gridData.gx));
                    probs_real = PredictFcn(Model, data.X, Level);
                    H_real = -sum(probs_real .* log(probs_real + epsilon), 2);
                    scat.Z = H_real + 0.05; scat.Color = data.Y; scat.Map = options.Colors;
                    opts.Mode = 'Entropy'; opts.ZLabel = 'Entropy (Bits)'; opts.Classes = data.Classes;
                else
                    [~, Y_pred] = max(probs, [], 2);
                    Z = reshape(Y_pred, size(gridData.gx));
                    scat.Z = data.Y + 0.1; scat.Color = data.Y; scat.Map = options.Colors;
                    opts.Mode = 'Discrete'; opts.ZLabel = 'Class'; opts.Classes = data.Classes; opts.ShowGradient = false;
                end
            else
                % === REGRESSION ===
                Y_pred = PredictFcn(Model, X_in, Level);
                Z = reshape(Y_pred, size(gridData.gx));
                scat.Z = data.Y; scat.Color = theme.Scatter;
                opts.Mode = 'Continuous'; opts.ZLabel = 'Predicted Y';
            end
            if isfield(options, 'Title'), opts.Title = options.Title; else, opts.Title = ''; end
        end
        
        function renderManifold(obj, ax, gridData, Z, scatterData, opts)
            hold(ax, 'on');
            handles = [];
            
            hSurf = obj.layerSurface(ax, gridData, Z, opts);
            if isvalid(hSurf), handles = [handles; hSurf]; end
            
            obj.layerVectorFields(ax, gridData, Z, opts);
            
            hTruth = obj.layerGroundTruth(ax, gridData, opts);
            if ~isempty(hTruth) && isvalid(hTruth), handles = [handles; hTruth]; end
            
            [hScatter, ~] = obj.layerScatter(ax, gridData, Z, scatterData, opts);
            if ~isempty(hScatter), handles = [handles; hScatter]; end
            
            obj.styleAxes(ax, opts);
            
            if ~isempty(handles)
                bgCol = opts.Theme.Background;
                if iscolumn(bgCol), bgCol = bgCol'; end
                legend(ax, handles, 'Location', 'best', 'Interpreter', 'latex', ...
                    'TextColor', opts.Theme.Text, 'EdgeColor', 'none', 'Color', [bgCol 0.8]);
            end
            hold(ax, 'off');
        end
        
        function s = layerSurface(~, ax, gridData, Z, opts)
            s = surf(ax, gridData.gx, gridData.gy, Z);
            faceCol = opts.Theme.Primary; 
            
            % 2. CRITICAL FIX: Define legendName BEFORE using it in set()
            if isfield(opts, 'ComplexityLevel')
                legendName = sprintf('Model Surface (d=%d)', opts.ComplexityLevel);
            else
                legendName = 'Model Surface';
            end
            
            set(s, 'EdgeColor', 'none', ...
                   'FaceColor', faceCol, ... 
                   'FaceAlpha', 0.7, ...
                   'DisplayName', legendName); % Now legendName exists!
                   
            camlight(ax, 'headlight'); 
            lighting(ax, 'gouraud');
        end
        
        function layerVectorFields(~, ax, gridData, Z, opts)
            if strcmp(opts.Mode, 'Discrete'), return; end
            if ~opts.ShowGradient, return; end
            [Fx, Fy] = gradient(Z);
            step = max(1, floor(size(gridData.gx, 1) / 15));
            idx_x = 1:step:size(gridData.gx, 2); idx_y = 1:step:size(gridData.gx, 1);
            X_q = gridData.gx(idx_y, idx_x); Y_q = gridData.gy(idx_y, idx_x); Z_q = Z(idx_y, idx_x);
            U_q = Fx(idx_y, idx_x); V_q = Fy(idx_y, idx_x); W_q = zeros(size(U_q));
            quiver3(ax, X_q, Y_q, Z_q, U_q, V_q, W_q, 1.5, 'Color', opts.Theme.Text, 'LineWidth', 1, 'DisplayName', 'Gradient');
        end
        
        function m = layerGroundTruth(~, ax, gridData, opts)
            m = [];
            if isempty(opts.GroundTruthFcn), return; end
            if size(gridData.coeff, 2) > 1 
                 X_in = [gridData.gx(:) gridData.gy(:)] * gridData.coeff(:,1:2)' + mean(gridData.X_vis); 
            else
                 X_in = [gridData.gx(:) gridData.gy(:)]; 
            end
            try
                Z_true = reshape(opts.GroundTruthFcn(X_in), size(gridData.gx));
                m = mesh(ax, gridData.gx, gridData.gy, Z_true, 'EdgeColor', opts.Theme.Secondary, 'FaceAlpha', 0, 'LineWidth', 1.5, 'DisplayName', 'Ground Truth');
            catch
            end
        end
        
        function [handles, resid_handle] = layerScatter(~, ax, gridData, Z, scatterData, opts)
            handles = []; resid_handle = [];
            textColor = opts.Theme.Text; if iscolumn(textColor), textColor = textColor'; end
            
            if strcmp(opts.Mode, 'Discrete') || strcmp(opts.Mode, 'Entropy')
                classes = unique(scatterData.Color); nClasses = length(classes); cMap = scatterData.Map; 
                for i = 1:nClasses
                    clsID = classes(i); idx = (scatterData.Color == clsID);
                    if isfield(opts, 'Classes') && clsID <= length(opts.Classes), lbl = string(opts.Classes(clsID)); else, lbl = sprintf('Class %d', clsID); end
                    if strcmp(opts.Mode, 'Entropy'), edgeCol = [1 1 1]; else, edgeCol = textColor; end
                    h = scatter3(ax, scatterData.X(idx), scatterData.Y(idx), scatterData.Z(idx), 25, cMap(clsID,:), 'filled', 'MarkerEdgeColor', edgeCol, 'DisplayName', lbl);
                    handles = [handles; h];
                end
                if strcmp(opts.Mode, 'Entropy')
                     colormap(ax, 'jet'); c = colorbar(ax); c.Label.String = opts.ZLabel; c.Label.Interpreter = 'latex'; c.Label.Color = textColor; c.Color = textColor;
                else
                     colormap(ax, cMap);
                end
            else
                scatCol = opts.Theme.Scatter; if iscolumn(scatCol), scatCol = scatCol'; end
                h = scatter3(ax, scatterData.X, scatterData.Y, scatterData.Z, 25, 'MarkerEdgeColor', textColor, 'MarkerFaceColor', scatCol, 'MarkerFaceAlpha', 0.6, 'DisplayName', 'Data');
                handles = [handles; h];
                Z_interp = interp2(gridData.gx, gridData.gy, Z, scatterData.X, scatterData.Y);
                lineColor = [textColor, 0.3]; 
                resid_handle = plot3(ax, [scatterData.X, scatterData.X]', [scatterData.Y, scatterData.Y]', [scatterData.Z, Z_interp]', 'Color', lineColor, 'DisplayName', 'Residuals');
            end
            axis(ax, 'vis3d');
        end
        
        function styleAxes(~, ax, opts)
            view(ax, [-30, 30]); rotate3d(ax, 'on'); grid(ax, 'on');
            zlabel(ax, opts.ZLabel, 'Interpreter', 'latex');
            title(ax, opts.Title, 'Interpreter', 'latex');
        end
    end
end