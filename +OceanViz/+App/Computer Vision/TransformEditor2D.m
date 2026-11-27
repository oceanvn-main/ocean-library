classdef TransformEditor2D < handle
    % TRANSFORMEDITOR2D  Interactive 2D Transform Editor (Szeliski Exercise 2.2).
    %
    % Interactive programmatic application to drag shape corners under Translation,
    % Rigid, Similarity, Affine, and Perspective models with real-time matrix display.
    %
    % Built as a plain MATLAB OOP class for perfect version control.
    %
    % Usage:
    %   app = TransformEditor2D();
    %   app.show();
    
    properties (Access = private)
        % main window handles
        Fig_
        Sidebar_
        CanvasAx_
        
        % controls & readouts
        RadioGroup_
        RadioButtons_ = cell(1, 5)
        MatrixLabels_ = cell(3, 3) % 3x3 grid of labels
        BtnSave_
        BtnLoad_
        
        % State
        Shapes_ = {}           % Cell array of shape structs: {base_corners, H}
        SelectedIdx_ = -1      % Selected shape index
        DraggedCornerIdx_ = -1 % Active drag corner handle index
        
        % Rubber-band drawing state
        IsShiftPressed_ = false
        DrawingActive_ = false
        DrawStartX_
        DrawStartY_
        DrawLineHandle_        % Handle to temporary line shape
        
        % Styling
        Colors_
        VizCore_
        
        % Plot handles to clear/update
        ShapePatches_ = []
        HandleLines_ = []
    end
    
    properties (Access = private, Constant)
        COLOR_BG        = [0.11, 0.11, 0.13]  % gunmetal background
        COLOR_PANEL     = [0.17, 0.19, 0.23]  % sidebar panel
        COLOR_CANVAS    = [0.10, 0.12, 0.16]  % canvas background
        COLOR_BTN       = [0.22, 0.25, 0.32]  % button background
        COLOR_GRID      = [0.25, 0.27, 0.33]  % grid lines
        COLOR_TEXT      = [0.90, 0.90, 0.90]  % primary text
        COLOR_ACCENT    = [0.20, 0.60, 1.00]  % active accent
        COLOR_SECONDARY = [1.00, 0.40, 0.40]  % inactive accent
        COLOR_MUTED     = [0.55, 0.58, 0.63]
        
        FIG_WIDTH       = 1100
        FIG_HEIGHT      = 700
        SIDEBAR_W       = 280
    end
    
    methods
        function obj = TransformEditor2D()
            % Construct the app and prepare styling
            try
                obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
                obj.Colors_ = obj.VizCore_.Colors;
            catch
                % Fallback theme if OceanViz is missing or fails
                obj.Colors_.Background = obj.COLOR_BG;
                obj.Colors_.Text = obj.COLOR_TEXT;
                obj.Colors_.Primary = obj.COLOR_ACCENT;
                obj.Colors_.Secondary = obj.COLOR_SECONDARY;
            end
            
            % Setup user interface
            obj.buildUI_();
            obj.drawGrid_();
            obj.renderCanvas_();
            obj.updateMatrixDisplay_();
        end
        
        function show(obj)
            % SHOW  Make the application window visible.
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end
        
        function delete(obj)
            % Clean teardown of figure window
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
    end
    
    methods (Access = private)
        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0 = (scr(3) - obj.FIG_WIDTH) / 2;
            y0 = (scr(4) - obj.FIG_HEIGHT) / 2;
            
            obj.Fig_ = uifigure(...
                'Name', '2D Transform Editor — Szeliski Ex 2.2', ...
                'Position', [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color', obj.COLOR_BG, ...
                'WindowButtonDownFcn', @(~,e) obj.onMouseDown_(e), ...
                'WindowButtonMotionFcn', @(~,e) obj.onMouseDrag_(e), ...
                'WindowButtonUpFcn', @(~,e) obj.onMouseUp_(e), ...
                'KeyPressFcn', @(~,e) obj.onKeyPress_(e), ...
                'KeyReleaseFcn', @(~,e) obj.onKeyRelease_(e));
            
            % --- Sidebar control panel ---
            obj.Sidebar_ = uipanel(obj.Fig_, ...
                'Position', [10, 10, obj.SIDEBAR_W, obj.FIG_HEIGHT - 20], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
            
            % Motion Model Title
            uilabel(obj.Sidebar_, ...
                'Text', 'Motion Model', ...
                'Position', [15, obj.FIG_HEIGHT - 55, obj.SIDEBAR_W - 30, 25], ...
                'FontSize', 14, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            % Radio Group
            obj.RadioGroup_ = uibuttongroup(obj.Sidebar_, ...
                'Position', [15, obj.FIG_HEIGHT - 230, obj.SIDEBAR_W - 30, 165], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'SelectionChangedFcn', @(~,~) obj.updateMatrixDisplay_());
            
            modes = {'Translation', 'Rigid', 'Similarity', 'Affine', 'Perspective'};
            for k = 1:5
                obj.RadioButtons_{k} = uiradiobutton(obj.RadioGroup_, ...
                    'Text', modes{k}, ...
                    'Position', [10, 135 - (k-1)*32, 200, 22], ...
                    'FontSize', 11, ...
                    'FontColor', obj.COLOR_TEXT);
            end
            
            % Divider 1
            uilabel(obj.Sidebar_, ...
                'Text', '', ...
                'Position', [15, obj.FIG_HEIGHT - 250, obj.SIDEBAR_W - 30, 1], ...
                'BackgroundColor', obj.COLOR_GRID);
            
            % Matrix Readout Title
            uilabel(obj.Sidebar_, ...
                'Text', 'Pose Matrix (H)', ...
                'Position', [15, obj.FIG_HEIGHT - 290, obj.SIDEBAR_W - 30, 25], ...
                'FontSize', 14, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            % Matrix 3x3 Labels Grid
            gridY = obj.FIG_HEIGHT - 390;
            for r = 1:3
                for c = 1:3
                    obj.MatrixLabels_{r,c} = uilabel(obj.Sidebar_, ...
                        'Text', '0.00', ...
                        'Position', [25 + (c-1)*75, gridY + (3-r)*30, 70, 24], ...
                        'FontSize', 10, ...
                        'FontWeight', 'bold', ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'BackgroundColor', obj.COLOR_CANVAS, ...
                        'HorizontalAlignment', 'center');
                end
            end
            
            % Divider 2
            uilabel(obj.Sidebar_, ...
                'Text', '', ...
                'Position', [15, obj.FIG_HEIGHT - 420, obj.SIDEBAR_W - 30, 1], ...
                'BackgroundColor', obj.COLOR_GRID);
            
            % Persistence Section
            uilabel(obj.Sidebar_, ...
                'Text', 'Persistence', ...
                'Position', [15, obj.FIG_HEIGHT - 460, obj.SIDEBAR_W - 30, 25], ...
                'FontSize', 13, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            obj.BtnSave_ = uibutton(obj.Sidebar_, ...
                'Text', 'Save Pose', ...
                'Position', [15, obj.FIG_HEIGHT - 505, 115, 32], ...
                'BackgroundColor', obj.COLOR_ACCENT, ...
                'FontColor', [1 1 1], ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.savePose_());
            
            obj.BtnLoad_ = uibutton(obj.Sidebar_, ...
                'Text', 'Load Pose', ...
                'Position', [145, obj.FIG_HEIGHT - 505, 115, 32], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.loadPose_());
            
            % Instruction Box
            instr = sprintf('Instructions:\n- Hold SHIFT & Drag to create a new rectangle.\n- Drag a corner handle of the selected shape to deform it.\n- Press Delete/Backspace to remove the active shape.');
            uilabel(obj.Sidebar_, ...
                'Text', instr, ...
                'Position', [15, 15, obj.SIDEBAR_W - 30, 120], ...
                'FontSize', 10, ...
                'FontColor', obj.COLOR_MUTED);
            
            % --- Main Canvas Axes ---
            canvasW = obj.FIG_WIDTH - obj.SIDEBAR_W - 40;
            obj.CanvasAx_ = uiaxes(obj.Fig_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, canvasW, obj.FIG_HEIGHT - 20], ...
                'Color', obj.COLOR_CANVAS, ...
                'XColor', 'none', ...
                'YColor', 'none', ...
                'XLim', [0, 800], ...
                'YLim', [0, 600]);
            
            disableDefaultInteractivity(obj.CanvasAx_);
            hold(obj.CanvasAx_, 'on');
        end
        
        function drawGrid_(obj)
            % Draw horizontal grid lines
            for y = 50:50:550
                plot(obj.CanvasAx_, [0, 800], [y, y], 'Color', obj.COLOR_GRID, 'LineWidth', 0.5, 'HandleVisibility', 'off');
            end
            % Draw vertical grid lines
            for x = 50:50:750
                plot(obj.CanvasAx_, [x, x], [0, 600], 'Color', obj.COLOR_GRID, 'LineWidth', 0.5, 'HandleVisibility', 'off');
            end
        end
        
        function corners = getShapeCorners(obj, shape)
            pts_hom = [shape.base_corners, ones(4, 1)]'; % [3 x 4]
            warped = shape.H * pts_hom; % [3 x 4]
            w = warped(3, :);
            w(abs(w) < 1e-15) = 1e-15;
            corners = [warped(1, :) ./ w; warped(2, :) ./ w]'; % [4 x 2]
        end
        
        function renderCanvas_(obj)
            % Delete existing shape plots & handles
            if ~isempty(obj.ShapePatches_)
                delete(obj.ShapePatches_);
                obj.ShapePatches_ = [];
            end
            if ~isempty(obj.HandleLines_)
                delete(obj.HandleLines_);
                obj.HandleLines_ = [];
            end
            
            % Draw all shapes
            for idx = 1:numel(obj.Shapes_)
                shape = obj.Shapes_{idx};
                corners = obj.getShapeCorners(shape);
                
                isActive = (idx == obj.SelectedIdx_);
                if isActive
                    faceCol = [0.08, 0.20, 0.30];
                    edgeCol = obj.COLOR_ACCENT;
                    edgeW = 2.0;
                else
                    faceCol = [0.17, 0.24, 0.31];
                    edgeCol = obj.COLOR_MUTED;
                    edgeW = 1.0;
                end
                
                % Draw filled polygon
                p = patch(obj.CanvasAx_, ...
                    'XData', corners(:,1), ...
                    'YData', corners(:,2), ...
                    'FaceColor', faceCol, ...
                    'FaceAlpha', 0.35, ...
                    'EdgeColor', edgeCol, ...
                    'LineWidth', edgeW);
                obj.ShapePatches_(end+1) = p;
                
                % Draw corner handles for active shape
                if isActive
                    for c_idx = 1:4
                        cx = corners(c_idx, 1);
                        cy = corners(c_idx, 2);
                        h = plot(obj.CanvasAx_, cx, cy, 'o', ...
                            'MarkerSize', 8, ...
                            'MarkerFaceColor', obj.COLOR_ACCENT, ...
                            'MarkerEdgeColor', [1 1 1], ...
                            'LineWidth', 1.0);
                        obj.HandleLines_(end+1) = h;
                    end
                end
            end
        end
        
        function updateMatrixDisplay_(obj)
            if obj.SelectedIdx_ ~= -1
                H = obj.Shapes_{obj.SelectedIdx_}.H;
            else
                H = eye(3);
            end
            
            for r = 1:3
                for c = 1:3
                    obj.MatrixLabels_{r,c}.Text = sprintf('%.2f', H(r,c));
                end
            end
        end
        
        function inside = pointInPolygon(~, px, py, corners)
            % Simple Ray-Casting wrapping MATLAB's inpolygon
            inside = inpolygon(px, py, corners(:,1), corners(:,2));
        end
        
        % --- Mouse Events ---
        function onMouseDown_(obj, event)
            % Get coordinate within Axes
            cp = obj.CanvasAx_.CurrentPoint;
            x = cp(1,1);
            y = cp(1,2);
            
            % Check if Shift is held down -> start drawing
            if obj.IsShiftPressed_
                obj.DrawingActive_ = true;
                obj.DrawStartX_ = x;
                obj.DrawStartY_ = y;
                obj.DrawLineHandle_ = plot(obj.CanvasAx_, [x,x,x,x,x], [y,y,y,y,y], '--', 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.0);
                return;
            end
            
            % 1. Check if clicked near a corner handle of the selected shape
            if obj.SelectedIdx_ ~= -1
                shape = obj.Shapes_{obj.SelectedIdx_};
                corners = obj.getShapeCorners(shape);
                for idx = 1:4
                    dist = norm(corners(idx, :) - [x, y]);
                    if dist < 12 % 12px click radius
                        obj.DraggedCornerIdx_ = idx;
                        return;
                    end
                end
            end
            
            % 2. Check if clicked inside any shape (top to bottom search)
            for idx = numel(obj.Shapes_):-1:1
                shape = obj.Shapes_{idx};
                corners = obj.getShapeCorners(shape);
                if obj.pointInPolygon(x, y, corners)
                    obj.SelectedIdx_ = idx;
                    obj.renderCanvas_();
                    obj.updateMatrixDisplay_();
                    return;
                end
            end
            
            % 3. Clicked empty space -> deselect
            obj.SelectedIdx_ = -1;
            obj.renderCanvas_();
            obj.updateMatrixDisplay_();
        end
        
        function onMouseDrag_(obj, event)
            cp = obj.CanvasAx_.CurrentPoint;
            x = cp(1,1);
            y = cp(1,2);
            
            % If drawing: update dashed rectangle box
            if obj.DrawingActive_ && ~isempty(obj.DrawLineHandle_) && isvalid(obj.DrawLineHandle_)
                x0 = obj.DrawStartX_;
                y0 = obj.DrawStartY_;
                obj.DrawLineHandle_.XData = [x0, x, x, x0, x0];
                obj.DrawLineHandle_.YData = [y0, y0, y, y, y0];
                return;
            end
            
            % If dragging a corner handle
            if obj.SelectedIdx_ ~= -1 && obj.DraggedCornerIdx_ ~= -1
                shape = obj.Shapes_{obj.SelectedIdx_};
                mode = obj.RadioGroup_.SelectedObject.Text;
                corners = obj.getShapeCorners(shape);
                k = obj.DraggedCornerIdx_;
                
                Q = corners; % Copy target corners
                
                if strcmp(mode, 'Translation')
                    % Shift all corners equally
                    dx = x - corners(k, 1);
                    dy = y - corners(k, 2);
                    Q = Q + [dx, dy];
                    
                elseif strcmp(mode, 'Rigid')
                    % Rotate other corners around opposite corner
                    opp = mod(k + 1, 4) + 1; % opposite corner in 1-based index (k+2 mod 4)
                    % Let's compute exact opposite:
                    % Corners layout: 1:bottom-left, 2:bottom-right, 3:top-right, 4:top-left
                    % Opposite index: 1<->3, 2<->4
                    opp_map = [3, 4, 1, 2];
                    opp = opp_map(k);
                    p_opp = corners(opp, :);
                    
                    v_old = corners(k, :) - p_opp;
                    v_new = [x, y] - p_opp;
                    
                    if norm(v_old) > 1e-5 && norm(v_new) > 1e-5
                        theta = atan2(v_new(2), v_new(1)) - atan2(v_old(2), v_old(1));
                        R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
                        for j = 1:4
                            if j ~= opp
                                Q(j, :) = p_opp + (R * (corners(j, :) - p_opp)')';
                            end
                        end
                    end
                    
                elseif strcmp(mode, 'Similarity')
                    % Scale & Rotate around opposite corner
                    opp_map = [3, 4, 1, 2];
                    opp = opp_map(k);
                    p_opp = corners(opp, :);
                    
                    % Form system mapping p_opp to itself and corners(k) to mouse
                    A = [
                        p_opp(1), -p_opp(2), 1, 0; ...
                        p_opp(2),  p_opp(1), 0, 1; ...
                        corners(k,1), -corners(k,2), 1, 0; ...
                        corners(k,2),  corners(k,1), 0, 1
                    ];
                    B = [p_opp(1); p_opp(2); x; y];
                    try
                        c = A \ B;
                        S = [c(1), -c(2), c(3); c(2), c(1), c(4); 0, 0, 1];
                        for j = 1:4
                            pt_h = S * [corners(j, 1); corners(j, 2); 1];
                            Q(j, :) = pt_h(1:2)';
                        end
                    catch
                    end
                    
                elseif strcmp(mode, 'Affine')
                    % Map active corner to mouse, opposite and adjacent stay fixed
                    opp_map = [3, 4, 1, 2];
                    opp = opp_map(k);
                    % Keep one adjacent fixed as well
                    adj_map = [4, 1, 2, 3];
                    adj = adj_map(k);
                    
                    p_opp = corners(opp, :);
                    p_adj = corners(adj, :);
                    
                    M = [
                        p_opp(1), p_opp(2), 1; ...
                        p_adj(1), p_adj(2), 1; ...
                        corners(k,1), corners(k,2), 1
                    ];
                    try
                        c_u = M \ [p_opp(1); p_adj(1); x];
                        c_v = M \ [p_opp(2); p_adj(2); y];
                        A = [c_u'; c_v'; 0, 0, 1];
                        for j = 1:4
                            pt_h = A * [corners(j, 1); corners(j, 2); 1];
                            Q(j, :) = pt_h(1:2)';
                        end
                    catch
                    end
                    
                elseif strcmp(mode, 'Perspective')
                    % Pin all other 3 corners, warp dragged corner to mouse
                    Q(k, :) = [x, y];
                end
                
                % Fit H mapping shape.base_corners to the target corners Q
                obj.Shapes_{obj.SelectedIdx_}.H = obj.fitTransformation_(mode, shape.base_corners, Q);
                obj.renderCanvas_();
                obj.updateMatrixDisplay_();
            end
        end
        
        function onMouseUp_(obj, event)
            if obj.DrawingActive_
                obj.DrawingActive_ = false;
                if ~isempty(obj.DrawLineHandle_)
                    delete(obj.DrawLineHandle_);
                    obj.DrawLineHandle_ = [];
                end
                
                % Calculate bounds
                cp = obj.CanvasAx_.CurrentPoint;
                x = cp(1,1);
                y = cp(1,2);
                x0 = obj.DrawStartX_;
                y0 = obj.DrawStartY_;
                w = abs(x - x0);
                h = abs(y - y0);
                
                if w > 15 && h > 15
                    cx = (x + x0) / 2;
                    cy = (y + y0) / 2;
                    
                    shape.base_corners = [
                        -w/2, -h/2; ...
                         w/2, -h/2; ...
                         w/2,  h/2; ...
                        -w/2,  h/2
                    ];
                    shape.H = eye(3);
                    shape.H(1, 3) = cx;
                    shape.H(2, 3) = cy;
                    
                    obj.Shapes_{end+1} = shape;
                    obj.SelectedIdx_ = numel(obj.Shapes_);
                    obj.renderCanvas_();
                    obj.updateMatrixDisplay_();
                end
            end
            obj.DraggedCornerIdx_ = -1;
        end
        
        function onKeyPress_(obj, event)
            if strcmp(event.Key, 'shift')
                obj.IsShiftPressed_ = true;
            elseif strcmp(event.Key, 'delete') || strcmp(event.Key, 'backspace')
                if obj.SelectedIdx_ ~= -1
                    obj.Shapes_(obj.SelectedIdx_) = [];
                    obj.SelectedIdx_ = -1;
                    obj.renderCanvas_();
                    obj.updateMatrixDisplay_();
                end
            end
        end
        
        function onKeyRelease_(obj, event)
            if strcmp(event.Key, 'shift')
                obj.IsShiftPressed_ = false;
            end
        end
        
        % --- Solver mathematics ---
        function H = fitTransformation_(obj, mode, src, dst)
            % Fits a transformation matrix H mapping src -> dst
            if strcmp(mode, 'Translation')
                tx = mean(dst(:,1)) - mean(src(:,1));
                ty = mean(dst(:,2)) - mean(src(:,2));
                H = eye(3);
                H(1, 3) = tx;
                H(2, 3) = ty;
                
            elseif strcmp(mode, 'Rigid')
                c_src = mean(src, 1);
                c_dst = mean(dst, 1);
                src_c = src - c_src;
                dst_c = dst - c_dst;
                M = src_c' * dst_c;
                [U, ~, V] = svd(M);
                R = V * U';
                if det(R) < 0
                    V(:, 2) = V(:, 2) * -1;
                    R = V * U';
                end
                t = c_dst' - R * c_src';
                H = eye(3);
                H(1:2, 1:2) = R;
                H(1:2, 3) = t;
                
            elseif strcmp(mode, 'Similarity')
                A = zeros(8, 4);
                B = zeros(8, 1);
                for i = 1:4
                    x = src(i,1); y = src(i,2);
                    u = dst(i,1); v = dst(i,2);
                    A(2*i-1, :) = [x, -y, 1, 0];
                    A(2*i, :)   = [y,  x, 0, 1];
                    B(2*i-1)    = u;
                    B(2*i)      = v;
                end
                c = A \ B;
                H = eye(3);
                H(1, 1) = c(1); H(1, 2) = -c(2); H(1, 3) = c(3);
                H(2, 1) = c(2); H(2, 2) = c(1);  H(2, 3) = c(4);
                
            elseif strcmp(mode, 'Affine')
                A = [src, ones(4,1)];
                c_u = A \ dst(:,1);
                c_v = A \ dst(:,2);
                H = eye(3);
                H(1, :) = c_u';
                H(2, :) = c_v';
                
            elseif strcmp(mode, 'Perspective')
                A = zeros(8, 8);
                B = zeros(8, 1);
                for i = 1:4
                    x = src(i,1); y = src(i,2);
                    u = dst(i,1); v = dst(i,2);
                    A(2*i-1, :) = [x, y, 1, 0, 0, 0, -u*x, -u*y];
                    A(2*i, :)   = [0, 0, 0, x, y, 1, -v*x, -v*y];
                    B(2*i-1)    = u;
                    B(2*i)      = v;
                end
                h = A \ B;
                H = eye(3);
                H(1, 1) = h(1); H(1, 2) = h(2); H(1, 3) = h(3);
                H(2, 1) = h(4); H(2, 2) = h(5); H(2, 3) = h(6);
                H(3, 1) = h(7); H(3, 2) = h(8); H(3, 3) = 1.0;
            end
        end
        
        % --- Save / Load ---
        function savePose_(obj)
            if isempty(obj.Shapes_)
                uialert(obj.Fig_, 'No shapes available to save.', 'Save Pose');
                return;
            end
            
            [file, path] = uiputfile('*.json', 'Save Transform Pose', 'Results/shapes_2d.json');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            % Convert cells to structures array
            saveData = [];
            for k = 1:numel(obj.Shapes_)
                saveData(k).base_corners = obj.Shapes_{k}.base_corners;
                saveData(k).H = obj.Shapes_{k}.H;
            end
            
            try
                jsonText = jsonencode(saveData, 'PrettyPrint', true);
                fid = fopen(fullfile(path, file), 'w');
                if fid == -1, error('Could not open file.'); end
                fprintf(fid, '%s', jsonText);
                fclose(fid);
                uialert(obj.Fig_, 'Transform configuration successfully saved!', 'Success', 'Icon', 'info');
            catch ME
                uialert(obj.Fig_, sprintf('Failed to save configuration:\n%s', ME.message), 'Error');
            end
        end
        
        function loadPose_(obj)
            [file, path] = uigetfile('*.json', 'Load Transform Pose', 'Results/');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            
            try
                fid = fopen(fullfile(path, file), 'r');
                if fid == -1, error('Could not open file.'); end
                rawText = fread(fid, '*char')';
                fclose(fid);
                
                loadedData = jsondecode(rawText);
                obj.Shapes_ = {};
                for k = 1:numel(loadedData)
                    shape.base_corners = loadedData(k).base_corners;
                    shape.H = loadedData(k).H;
                    obj.Shapes_{k} = shape;
                end
                
                obj.SelectedIdx_ = -1;
                obj.renderCanvas_();
                obj.updateMatrixDisplay_();
                uialert(obj.Fig_, sprintf('Loaded %d shapes successfully!', numel(obj.Shapes_)), 'Success', 'Icon', 'info');
            catch ME
                uialert(obj.Fig_, sprintf('Failed to load configuration:\n%s', ME.message), 'Error');
            end
        end
    end
end
