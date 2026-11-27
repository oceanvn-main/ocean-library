classdef Viewer3D < handle
    % VIEWER3D  Interactive 3D Primitives Viewer (Szeliski Exercise 2.3).
    %
    % Interactive programmatic application to project 3D wireframe and solid
    % primitives (points, lines, polygons) using Perspective/Orthographic models
    % with camera/object control, Painter's algorithm depth-sorting, and scene importing.
    %
    % Built as a plain MATLAB OOP class for perfect version control.
    %
    % Usage:
    %   app = Viewer3D();
    %   app.show();
    
    properties (Access = private)
        % Main window handles
        Fig_
        Sidebar_
        TabGroup_
        TabCam_
        TabObj_
        TabImport_
        CanvasAx_
        
        % Scene Primitives
        Points_ = {}      % Cell array of [4 x 1] homogeneous coordinates
        Lines_ = {}       % Cell array of [1 x 2] indices (0-based)
        Polygons_ = {}    % Cell array of [1 x N] indices (0-based)
        PolyColors_ = {}  % Cell array of RGB vectors [r, g, b]
        
        % UI slider variables (Euler in degrees, translation, scale)
        Cam_rx_ = 15.0; Cam_ry_ = -25.0; Cam_rz_ = 0.0;
        Cam_tx_ = 0.0;  Cam_ty_ = 1.0;   Cam_tz_ = -8.0;
        
        Obj_rx_ = 0.0;  Obj_ry_ = 0.0;   Obj_rz_ = 0.0;
        Obj_tx_ = 0.0;  Obj_ty_ = 0.0;   Obj_tz_ = 0.0;
        Obj_scale_ = 1.0;
        
        ProjType_ = 'Perspective'; % 'Perspective' or 'Orthographic'
        FocalLength_ = 400.0;
        OrthoScale_ = 80.0;
        
        % Mouse dragging state
        LastMouseX_
        LastMouseY_
        IsMouseBtnDown_ = false
        
        % Control handle lists for callbacks
        SliderControls_ = struct()
        ImportTextArea_
        
        % Render Plot Handles for clearing
        RenderHandles_ = []
        
        % Styling
        Colors_
        VizCore_
        
        % Texture mapping properties
        TextureImg_
        TextureName_ = 'ocean_img.jpg'
        PolyTexCoords_ = {}
        EnableTexture_ = false
        Plot_ImgHandle_
        Chk_EnableTexture_
        Btn_LoadTexture_
    end
    
    properties (Access = private, Constant)
        COLOR_BG        = [0.11, 0.11, 0.13]  % gunmetal background
        COLOR_PANEL     = [0.17, 0.19, 0.23]  % sidebar panel
        COLOR_CANVAS    = [0.10, 0.12, 0.16]  % canvas background
        COLOR_BTN       = [0.22, 0.25, 0.32]  % button background
        COLOR_GRID      = [0.25, 0.27, 0.33]  % grid lines
        COLOR_TEXT      = [0.90, 0.90, 0.90]  % primary text
        COLOR_ACCENT    = [0.20, 0.60, 1.00]  % active accent
        COLOR_SECONDARY = [1.00, 0.40, 0.40]  % secondary accent
        COLOR_MUTED     = [0.55, 0.58, 0.63]
        
        FIG_WIDTH       = 1150
        FIG_HEIGHT      = 750
        SIDEBAR_W       = 320
    end
    
    methods
        function obj = Viewer3D()
            % Construct the app and prepare styling
            try
                obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
                obj.Colors_ = obj.VizCore_.Colors;
            catch
                % Fallback theme
                obj.Colors_.Background = obj.COLOR_BG;
                obj.Colors_.Text = obj.COLOR_TEXT;
                obj.Colors_.Primary = obj.COLOR_ACCENT;
                obj.Colors_.Secondary = obj.COLOR_SECONDARY;
            end
            
            % Load default texture image
            script_dir = fileparts(mfilename('fullpath'));
            project_root = fullfile(script_dir, '..', '..', '..');
            img_path = fullfile(project_root, 'Data', 'images', obj.TextureName_);
            try
                obj.TextureImg_ = double(imread(img_path)) / 255.0;
            catch
                obj.TextureImg_ = repmat(reshape([0.2, 0.25, 0.32], 1, 1, 3), 256, 256);
            end

            % Load default house scene
            obj.loadDefaultHouse_();
            
            % Setup UI
            obj.buildUI_();
            obj.redraw_();
        end
        
        function show(obj)
            % SHOW  Make the application window visible.
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end
        
        function delete(obj)
            % Clean teardown
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
        
        function testRedraw(obj)
            obj.EnableTexture_ = true;
            obj.redraw_();
        end
        
        function data = getPrivateData(obj)
            data = struct();
            data.Points_ = obj.Points_;
            data.Lines_ = obj.Lines_;
            data.Polygons_ = obj.Polygons_;
            data.PolyColors_ = obj.PolyColors_;
            data.PolyTexCoords_ = obj.PolyTexCoords_;
            data.TextureImg_ = obj.TextureImg_;
            data.CanvasAx_ = obj.CanvasAx_;
            data.ProjType_ = obj.ProjType_;
            data.FocalLength_ = obj.FocalLength_;
            data.OrthoScale_ = obj.OrthoScale_;
            data.Cam_rx_ = obj.Cam_rx_;
            data.Cam_ry_ = obj.Cam_ry_;
            data.Cam_rz_ = obj.Cam_rz_;
            data.Cam_tx_ = obj.Cam_tx_;
            data.Cam_ty_ = obj.Cam_ty_;
            data.Cam_tz_ = obj.Cam_tz_;
            data.Obj_rx_ = obj.Obj_rx_;
            data.Obj_ry_ = obj.Obj_ry_;
            data.Obj_rz_ = obj.Obj_rz_;
            data.Obj_tx_ = obj.Obj_tx_;
            data.Obj_ty_ = obj.Obj_ty_;
            data.Obj_tz_ = obj.Obj_tz_;
            data.Obj_scale_ = obj.Obj_scale_;
        end
    end
    
    methods (Access = private)
        function loadDefaultHouse_(obj)
            % Clear
            obj.Points_ = {};
            obj.Lines_ = {};
            obj.Polygons_ = {};
            obj.PolyColors_ = {};
            
            % 10 vertices
            vertices = [
                -1.5, -1.5, -1.0; ... % 0: floor
                 1.5, -1.5, -1.0; ... % 1
                 1.5,  1.5, -1.0; ... % 2
                -1.5,  1.5, -1.0; ... % 3
                -1.5, -1.5,  0.5; ... % 4: wall tops
                 1.5, -1.5,  0.5; ... % 5
                 1.5,  1.5,  0.5; ... % 6
                -1.5,  1.5,  0.5; ... % 7
                 0.0, -1.5,  1.5; ... % 8: roof front
                 0.0,  1.5,  1.5      % 9: roof back
            ];
            for k = 1:size(vertices, 1)
                obj.Points_{k} = [vertices(k, 1); vertices(k, 2); vertices(k, 3); 1.0];
            end
            
            % 0-based lines indices
            lines = {
                [0, 1], [1, 2], [2, 3], [3, 0], ...
                [4, 5], [5, 6], [6, 7], [7, 4], ...
                [0, 4], [1, 5], [2, 6], [3, 7], ...
                [4, 8], [5, 8], [6, 9], [7, 9], [8, 9]
            };
            obj.Lines_ = lines;
            
            % 0-based polygons indices
            polys = {
                [0, 1, 2, 3], ...
                [0, 1, 5, 4], ...
                [1, 2, 6, 5], ...
                [2, 3, 7, 6], ...
                [3, 0, 4, 7], ...
                [4, 5, 8],    ...
                [6, 7, 9],    ...
                [4, 7, 9, 8], ...
                [5, 6, 9, 8]
            };
            obj.Polygons_ = polys;
            
            % Hex/RGB colors
            obj.PolyColors_ = {
                [0.17, 0.24, 0.31], ... % floor
                [0.20, 0.29, 0.37], ... % front
                [0.50, 0.55, 0.55], ... % right
                [0.58, 0.65, 0.65], ... % back
                [0.74, 0.76, 0.78], ... % left
                [0.90, 0.49, 0.13], ... % roof front
                [0.83, 0.33, 0.00], ... % roof back
                [0.91, 0.30, 0.24], ... % roof left
                [0.75, 0.22, 0.17]     % roof right
            };
            
            % Default homogeneous texture coordinates (u, v, w)
            obj.PolyTexCoords_ = {
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % floor
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % front
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % right
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % back
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % left
                [0, 1, 1; 1, 1, 1; 0.5, 0, 1], ...        % roof front
                [0, 1, 1; 1, 1, 1; 0.5, 0, 1], ...        % roof back
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % roof left
                [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1]      % roof right
            };
        end
        
        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0 = (scr(3) - obj.FIG_WIDTH) / 2;
            y0 = (scr(4) - obj.FIG_HEIGHT) / 2;
            
            obj.Fig_ = uifigure(...
                'Name', '3D Primitives Viewer — Szeliski Ex 2.3', ...
                'Position', [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color', obj.COLOR_BG, ...
                'WindowButtonDownFcn', @(~,e) obj.onMouseDown_(e), ...
                'WindowButtonMotionFcn', @(~,e) obj.onMouseDrag_(e), ...
                'WindowButtonUpFcn', @(~,e) obj.onMouseUp_(e));
            
            % --- Sidebar panel ---
            obj.Sidebar_ = uipanel(obj.Fig_, ...
                'Position', [10, 10, obj.SIDEBAR_W, obj.FIG_HEIGHT - 20], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
            
            % Tab Group inside sidebar
            obj.TabGroup_ = uitabgroup(obj.Sidebar_, ...
                'Position', [5, 5, obj.SIDEBAR_W - 10, obj.FIG_HEIGHT - 35]);
            
            obj.TabCam_ = uitab(obj.TabGroup_, 'Title', 'Camera', 'BackgroundColor', obj.COLOR_PANEL, 'Scrollable', 'on');
            obj.TabObj_ = uitab(obj.TabGroup_, 'Title', 'Object', 'BackgroundColor', obj.COLOR_PANEL, 'Scrollable', 'on');
            obj.TabImport_ = uitab(obj.TabGroup_, 'Title', 'Import', 'BackgroundColor', obj.COLOR_PANEL);
            
            % --- Tab 1: Camera & Projection ---
            y = obj.FIG_HEIGHT - 45;
            uilabel(obj.TabCam_, 'Text', 'Projection Type', 'Position', [10, y, 150, 22], 'FontColor', obj.COLOR_TEXT, 'FontWeight', 'bold');
            projDropdown = uidropdown(obj.TabCam_, ...
                'Items', {'Perspective', 'Orthographic'}, ...
                'Position', [140, y, 120, 24], ...
                'Value', obj.ProjType_, ...
                'ValueChangedFcn', @(d,~) obj.changeProjType_(d.Value));
            
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Focal Length (f)', 100, 1500, obj.FocalLength_, y, 'FocalLength_');
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Ortho Scale (s)', 10, 300, obj.OrthoScale_, y, 'OrthoScale_');
            
            y = y - 35;
            uilabel(obj.TabCam_, ...
                'Text', 'Camera Extrinsics', ...
                'Position', [10, y, 200, 22], ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Cam Pitch (RX)', -180, 180, obj.Cam_rx_, y, 'Cam_rx_');
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Cam Yaw (RY)', -180, 180, obj.Cam_ry_, y, 'Cam_ry_');
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Cam Roll (RZ)', -180, 180, obj.Cam_rz_, y, 'Cam_rz_');
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Cam TX', -10, 10, obj.Cam_tx_, y, 'Cam_tx_');
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Cam TY', -10, 10, obj.Cam_ty_, y, 'Cam_ty_');
            y = y - 55;
            obj.createSlider_(obj.TabCam_, 'Cam TZ', -30, -1, obj.Cam_tz_, y, 'Cam_tz_');
            
            % Texture Settings Panel at the bottom of TabCam_
            y = y - 40;
            uilabel(obj.TabCam_, ...
                'Text', 'Texture Mapping (Ex 3.27)', ...
                'Position', [10, y, 200, 20], ...
                'FontSize', 11, 'FontWeight', 'bold', 'FontColor', obj.COLOR_ACCENT);
                
            y = y - 30;
            obj.Chk_EnableTexture_ = uicheckbox(obj.TabCam_, ...
                'Position', [10, y, 130, 22], ...
                'Text', 'Enable Textures', ...
                'FontColor', obj.COLOR_TEXT, ...
                'Value', obj.EnableTexture_, ...
                'ValueChangedFcn', @(cb,~) obj.toggleTexture_(cb.Value));
                
            obj.Btn_LoadTexture_ = uibutton(obj.TabCam_, ...
                'Position', [145, y - 2, 120, 24], ...
                'Text', 'Load Texture...', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.loadCustomTexture_());
            
            % Help instruction at the bottom of the tab controls
            y = y - 50;
            uilabel(obj.TabCam_, ...
                'Text', 'Hint: Drag mouse on canvas to rotate camera.', ...
                'Position', [10, y, 270, 45], ...
                'FontSize', 10, ...
                'FontColor', obj.COLOR_MUTED, ...
                'WordWrap', 'on');
            
            % --- Tab 2: Object Transform ---
            y = obj.FIG_HEIGHT - 45;
            uilabel(obj.TabObj_, ...
                'Text', 'Object Model Transform', ...
                'Position', [10, y, 200, 22], ...
                'FontSize', 13, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj RX', -180, 180, obj.Obj_rx_, y, 'Obj_rx_');
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj RY', -180, 180, obj.Obj_ry_, y, 'Obj_ry_');
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj RZ', -180, 180, obj.Obj_rz_, y, 'Obj_rz_');
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj TX', -10, 10, obj.Obj_tx_, y, 'Obj_tx_');
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj TY', -10, 10, obj.Obj_ty_, y, 'Obj_ty_');
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj TZ', -10, 10, obj.Obj_tz_, y, 'Obj_tz_');
            y = y - 55;
            obj.createSlider_(obj.TabObj_, 'Obj Scale', 0.1, 5.0, obj.Obj_scale_, y, 'Obj_scale_');
            
            % --- Tab 3: Scene Importer ---
            y = obj.FIG_HEIGHT - 85;
            uilabel(obj.TabImport_, ...
                'Text', 'Scene Primitives Command', ...
                'Position', [10, y, 250, 22], ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'FontColor', obj.COLOR_ACCENT);
            
            y = y - 180;
            obj.ImportTextArea_ = uitextarea(obj.TabImport_, ...
                'Position', [10, y, 275, 170], ...
                'BackgroundColor', obj.COLOR_CANVAS, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10);
            
            default_cmds = sprintf('# Format:\n# P x y z\n# L idx1 idx2\n# POLY idx1 idx2 idx3 ...\n# (indices are 0-based)\n');
            obj.ImportTextArea_.Value = default_cmds;
            
            y = y - 45;
            uibutton(obj.TabImport_, ...
                'Text', 'Import Scene', ...
                'Position', [10, y, 130, 32], ...
                'BackgroundColor', obj.COLOR_ACCENT, ...
                'FontColor', [1 1 1], ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.importScene_());
            
            uibutton(obj.TabImport_, ...
                'Text', 'Load Preset', ...
                'Position', [155, y, 130, 32], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.showPresetPopup_());
            
            % --- Main Canvas Axes ---
            canvasW = obj.FIG_WIDTH - obj.SIDEBAR_W - 40;
            obj.CanvasAx_ = uiaxes(obj.Fig_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, canvasW, obj.FIG_HEIGHT - 20], ...
                'Color', obj.COLOR_CANVAS, ...
                'XColor', 'none', ...
                'YColor', 'none', ...
                'XLim', [0, canvasW], ...
                'YLim', [0, obj.FIG_HEIGHT - 20]);
            
            disableDefaultInteractivity(obj.CanvasAx_);
            hold(obj.CanvasAx_, 'on');
        end
        
        function createSlider_(obj, parent, labelText, minVal, maxVal, initVal, yPos, propName)
            uilabel(parent, ...
                'Text', labelText, ...
                'Position', [10, yPos + 18, 180, 18], ...
                'FontSize', 9, ...
                'FontColor', obj.COLOR_TEXT);
            
            valLabel = uilabel(parent, ...
                'Text', sprintf('%.2f', initVal), ...
                'Position', [210, yPos + 18, 60, 18], ...
                'FontSize', 9, ...
                'FontColor', obj.COLOR_MUTED, ...
                'HorizontalAlignment', 'right');
            
            slider = uislider(parent, ...
                'Limits', [minVal, maxVal], ...
                'Value', initVal, ...
                'Position', [10, yPos + 8, 255, 3], ...
                'MajorTicks', [], ...
                'MajorTickLabels', {}, ...
                'ValueChangedFcn', @(s,~) obj.onSliderChange_(s, valLabel, propName));
            
            obj.SliderControls_.(propName).slider = slider;
            obj.SliderControls_.(propName).label = valLabel;
        end
        
        function onSliderChange_(obj, sliderHandle, labelHandle, propName)
            val = sliderHandle.Value;
            obj.(propName) = val;
            labelHandle.Text = sprintf('%.2f', val);
            obj.redraw_();
        end
        
        function changeProjType_(obj, val)
            obj.ProjType_ = val;
            obj.redraw_();
        end
        
        function R = getRotationMatrix_(obj, rx, ry, rz)
            phi = deg2rad(rx);   % pitch
            theta = deg2rad(ry); % yaw
            psi = deg2rad(rz);   % roll
            
            Rx = [1, 0, 0; 0, cos(phi), -sin(phi); 0, sin(phi), cos(phi)];
            Ry = [cos(theta), 0, sin(theta); 0, 1, 0; -sin(theta), 0, cos(theta)];
            Rz = [cos(psi), -sin(psi), 0; sin(psi), cos(psi), 0; 0, 0, 1];
            
            % ZYX convention matching Szeliski eulerToMatrix
            R = Rz * Ry * Rx;
        end
        
        function redraw_(obj)
            % Clean up existing graphic handles
            if ~isempty(obj.RenderHandles_)
                delete(obj.RenderHandles_);
                obj.RenderHandles_ = [];
            end
            
            % Size parameters
            w = obj.CanvasAx_.Position(3);
            h = obj.CanvasAx_.Position(4);
            cx = w / 2;
            cy = h / 2;
            
            if isempty(obj.Points_)
                return;
            end
            
            % 1. Compute Object Pose Model Matrix
            R_obj = obj.getRotationMatrix_(obj.Obj_rx_, obj.Obj_ry_, obj.Obj_rz_);
            S_obj = obj.Obj_scale_;
            T_obj = [obj.Obj_tx_; obj.Obj_ty_; obj.Obj_tz_];
            
            M_obj = eye(4);
            M_obj(1:3, 1:3) = R_obj * S_obj;
            M_obj(1:3, 4) = T_obj;
            
            % 2. Compute Camera View Matrix (Inverse of camera pose)
            R_cam = obj.getRotationMatrix_(obj.Cam_rx_, obj.Cam_ry_, obj.Cam_rz_);
            T_cam = [obj.Cam_tx_; obj.Cam_ty_; obj.Cam_tz_];
            
            M_view = eye(4);
            M_view(1:3, 1:3) = R_cam';
            M_view(1:3, 4) = -R_cam' * T_cam;
            
            % 3. Transform points to Camera Space
            pts_cam = zeros(4, numel(obj.Points_));
            for k = 1:numel(obj.Points_)
                pts_cam(:, k) = M_view * M_obj * obj.Points_{k};
            end
            
            % 4. Project coordinates to viewport screen space
            pts_screen = zeros(2, numel(obj.Points_));
            in_front = false(1, numel(obj.Points_));
            
            for k = 1:numel(obj.Points_)
                p = pts_cam(:, k);
                px = p(1); py = p(2); pz = p(3);
                
                if strcmp(obj.ProjType_, 'Perspective')
                    if pz > 0.1
                        f = obj.FocalLength_;
                        pts_screen(1, k) = f * px / pz + cx;
                        pts_screen(2, k) = -f * py / pz + cy; % invert y
                        in_front(k) = true;
                    end
                else
                    s = obj.OrthoScale_;
                    pts_screen(1, k) = s * px + cx;
                    pts_screen(2, k) = -s * py + cy;
                    in_front(k) = true;
                end
            end
            
            % --- RENDER STEP 1: Polygons with Hidden Surface Removal (Painter's Algorithm) ---
            % Calculate depth for each polygon in camera coordinates
            nPolys = numel(obj.Polygons_);
            if nPolys > 0
                polyDepths = zeros(nPolys, 1);
                validPolys = false(nPolys, 1);
                
                for k = 1:nPolys
                    idx = obj.Polygons_{k} + 1; % Convert 0-based to 1-based index
                    if all(in_front(idx))
                        validPolys(k) = true;
                        polyDepths(k) = mean(pts_cam(3, idx));
                    end
                end
                
                % Sort valid polygons by depth descending (largest depth = farthest -> draw first)
                validIndices = find(validPolys);
                [~, sortIdx] = sort(polyDepths(validIndices), 'descend');
                sortedPolyIndices = validIndices(sortIdx);
                
                if obj.EnableTexture_
                    % Rasterized perspective-correct texture rendering pipeline
                    canvas_w = floor(w);
                    canvas_h = floor(h);
                    I_canvas = repmat(reshape(obj.COLOR_CANVAS, 1, 1, 3), canvas_h, canvas_w);
                    
                    for k = 1:numel(sortedPolyIndices)
                        polyIdx = sortedPolyIndices(k);
                        v_idxs = obj.Polygons_{polyIdx} + 1;
                        pts_poly_cam = pts_cam(:, v_idxs)'; % [N x 4] - camera space coords
                        pts_poly_screen = pts_screen(:, v_idxs)'; % [N x 2] - screen space coords
                        
                        has_texture = (polyIdx <= numel(obj.PolyTexCoords_)) && ~isempty(obj.PolyTexCoords_{polyIdx});
                        if has_texture
                            tc = obj.PolyTexCoords_{polyIdx};
                            [H_tex, W_tex, ~] = size(obj.TextureImg_);
                            
                            % Scale normalized texture coordinates to pixel space
                            u_norm = tc(:, 1) ./ tc(:, 3);
                            v_norm = tc(:, 2) ./ tc(:, 3);
                            u_tex = u_norm * (W_tex - 1) + 1;
                            v_tex = v_norm * (H_tex - 1) + 1;
                            
                            % Triangulate the N-vertex polygon using a triangle fan
                            nV = numel(v_idxs);
                            for t = 1:(nV - 2)
                                idxA = 1;
                                idxB = t + 1;
                                idxC = t + 2;
                                
                                % Triangle vertices screen coords
                                p1 = pts_poly_screen(idxA, :);
                                p2 = pts_poly_screen(idxB, :);
                                p3 = pts_poly_screen(idxC, :);
                                
                                % Depths Z (camera space Z)
                                Z1 = pts_poly_cam(idxA, 3);
                                Z2 = pts_poly_cam(idxB, 3);
                                Z3 = pts_poly_cam(idxC, 3);
                                
                                % Avoid division by zero/negative depths
                                Z1(abs(Z1) < 1e-6) = 1e-6;
                                Z2(abs(Z2) < 1e-6) = 1e-6;
                                Z3(abs(Z3) < 1e-6) = 1e-6;
                                
                                % Texture coords at vertices
                                ut1 = u_tex(idxA); vt1 = v_tex(idxA);
                                ut2 = u_tex(idxB); vt2 = v_tex(idxB);
                                ut3 = u_tex(idxC); vt3 = v_tex(idxC);
                                
                                % Projective attributes at vertices:
                                % 1. Inverse depth w = 1/Z
                                % 2. Pre-divided texture coordinates u/Z, v/Z
                                w1 = 1 / Z1;  u1_z = ut1 / Z1;  v1_z = vt1 / Z1;
                                w2 = 1 / Z2;  u2_z = ut2 / Z2;  v2_z = vt2 / Z2;
                                w3 = 1 / Z3;  u3_z = ut3 / Z3;  v3_z = vt3 / Z3;
                                
                                % Bounding box of the triangle
                                min_x = max(1, floor(min([p1(1), p2(1), p3(1)])));
                                max_x = min(canvas_w, ceil(max([p1(1), p2(1), p3(1)])));
                                min_y = max(1, floor(min([p1(2), p2(2), p3(2)])));
                                max_y = min(canvas_h, ceil(max([p1(2), p2(2), p3(2)])));
                                
                                if min_x <= max_x && min_y <= max_y
                                    [X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);
                                    
                                    x1 = p1(1); y1 = p1(2);
                                    x2 = p2(1); y2 = p2(2);
                                    x3 = p3(1); y3 = p3(2);
                                    
                                    % Determinant for barycentric coordinates
                                    D = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
                                    if abs(D) > 1e-6
                                        beta = ((X_grid - x1)*(y3 - y1) - (x3 - x1)*(Y_grid - y1)) / D;
                                        gamma = ((x2 - x1)*(Y_grid - y1) - (X_grid - x1)*(y2 - y1)) / D;
                                        alpha = 1 - beta - gamma;
                                        
                                        inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0);
                                        if any(inside(:))
                                            x_canvas = X_grid(inside);
                                            y_canvas = Y_grid(inside);
                                            
                                            a_v = alpha(inside);
                                            b_v = beta(inside);
                                            g_v = gamma(inside);
                                            
                                            % Interpolate w = 1/Z and projective attributes
                                            w_interp = a_v * w1 + b_v * w2 + g_v * w3;
                                            u_z_interp = a_v * u1_z + b_v * u2_z + g_v * u3_z;
                                            v_z_interp = a_v * v1_z + b_v * v2_z + g_v * v3_z;
                                            
                                            % Final perspective divide per fragment
                                            w_interp(abs(w_interp) < 1e-12) = 1e-12;
                                            src_x = u_z_interp ./ w_interp;
                                            src_y = v_z_interp ./ w_interp;
                                            
                                            % Clamp bilinear coordinates to image boundary
                                            valid_tex = (src_x >= 1 & src_x <= W_tex & src_y >= 1 & src_y <= H_tex);
                                            if any(valid_tex)
                                                src_x_v = src_x(valid_tex);
                                                src_y_v = src_y(valid_tex);
                                                xc = x_canvas(valid_tex);
                                                yc = y_canvas(valid_tex);
                                                
                                                tx_f = floor(src_x_v); ty_f = floor(src_y_v);
                                                tx_c = tx_f + 1;       ty_c = ty_f + 1;
                                                
                                                tx_f_c = max(1, min(W_tex, tx_f));
                                                tx_c_c = max(1, min(W_tex, tx_c));
                                                ty_f_c = max(1, min(H_tex, ty_f));
                                                ty_c_c = max(1, min(H_tex, ty_c));
                                                
                                                dx = src_x_v - tx_f;
                                                dy = src_y_v - ty_f;
                                                
                                                w00 = (1 - dx) .* (1 - dy);
                                                w01 = dx .* (1 - dy);
                                                w10 = (1 - dx) .* dy;
                                                w11 = dx .* dy;
                                                
                                                idx00 = ty_f_c + (tx_f_c - 1) * H_tex;
                                                idx01 = ty_f_c + (tx_c_c - 1) * H_tex;
                                                idx10 = ty_c_c + (tx_f_c - 1) * H_tex;
                                                idx11 = ty_c_c + (tx_c_c - 1) * H_tex;
                                                
                                                canvas_idx = yc + (xc - 1) * canvas_h;
                                                for channel = 1:3
                                                    Ic = obj.TextureImg_(:, :, channel);
                                                    val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                                                    I_canvas(canvas_idx + (channel - 1) * canvas_h * canvas_w) = val;
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        else
                            % Fallback: solid color rasterization
                            col = obj.PolyColors_{polyIdx};
                            min_x = max(1, floor(min(pts_poly_screen(:, 1))));
                            max_x = min(canvas_w, ceil(max(pts_poly_screen(:, 1))));
                            min_y = max(1, floor(min(pts_poly_screen(:, 2))));
                            max_y = min(canvas_h, ceil(max(pts_poly_screen(:, 2))));
                            
                            if min_x <= max_x && min_y <= max_y
                                [X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);
                                inside = inpolygon(X_grid, Y_grid, pts_poly_screen(:, 1), pts_poly_screen(:, 2));
                                if any(inside(:))
                                    x_canvas = X_grid(inside);
                                    y_canvas = Y_grid(inside);
                                    canvas_idx = y_canvas + (x_canvas - 1) * canvas_h;
                                    for channel = 1:3
                                        I_canvas(canvas_idx + (channel - 1) * canvas_h * canvas_w) = col(channel);
                                    end
                                end
                            end
                        end
                    end
                    
                    % Update background image handle
                    if isgraphics(obj.Plot_ImgHandle_)
                        set(obj.Plot_ImgHandle_, 'CData', im2uint8(I_canvas));
                        set(obj.Plot_ImgHandle_, 'Visible', 'on');
                    else
                        hold(obj.CanvasAx_, 'on');
                        obj.Plot_ImgHandle_ = imshow(I_canvas, 'Parent', obj.CanvasAx_);
                        uistack(obj.Plot_ImgHandle_, 'bottom');
                        axis(obj.CanvasAx_, 'image'); axis(obj.CanvasAx_, 'off');
                    end
                else
                    % Vector rendering
                    if isgraphics(obj.Plot_ImgHandle_)
                        set(obj.Plot_ImgHandle_, 'Visible', 'off');
                    end
                    for k = 1:numel(sortedPolyIndices)
                        polyIdx = sortedPolyIndices(k);
                        v_idxs = obj.Polygons_{polyIdx} + 1;
                        col = obj.PolyColors_{polyIdx};
                        p = patch(obj.CanvasAx_, ...
                            'XData', pts_screen(1, v_idxs), ...
                            'YData', pts_screen(2, v_idxs), ...
                            'FaceColor', col, ...
                            'FaceAlpha', 0.85, ...
                            'EdgeColor', obj.COLOR_GRID, ...
                            'LineWidth', 0.5);
                        obj.RenderHandles_(end+1) = p;
                    end
                end
            end

            
            % --- RENDER STEP 2: Lines ---
            for k = 1:numel(obj.Lines_)
                idx1 = obj.Lines_{k}(1) + 1;
                idx2 = obj.Lines_{k}(2) + 1;
                
                if in_front(idx1) && in_front(idx2)
                    l = plot(obj.CanvasAx_, ...
                        [pts_screen(1, idx1), pts_screen(1, idx2)], ...
                        [pts_screen(2, idx1), pts_screen(2, idx2)], ...
                        '-', 'Color', obj.COLOR_ACCENT, ...
                        'LineWidth', 1.5);
                    obj.RenderHandles_(end+1) = l;
                end
            end
            
            % --- RENDER STEP 3: Points ---
            for k = 1:numel(obj.Points_)
                if in_front(k)
                    pt = plot(obj.CanvasAx_, pts_screen(1, k), pts_screen(2, k), ...
                        'o', 'MarkerSize', 5, ...
                        'MarkerFaceColor', obj.COLOR_SECONDARY, ...
                        'MarkerEdgeColor', [1 1 1], ...
                        'LineWidth', 0.5);
                    obj.RenderHandles_(end+1) = pt;
                end
            end
        end
        
        % --- Mouse events for camera Orbit rotation ---
        function onMouseDown_(obj, event)
            cp = obj.Fig_.CurrentPoint;
            % Confirm click is within Canvas area
            axPos = obj.CanvasAx_.Position;
            if cp(1) >= axPos(1) && cp(1) <= axPos(1) + axPos(3) && ...
               cp(2) >= axPos(2) && cp(2) <= axPos(2) + axPos(4)
                obj.IsMouseBtnDown_ = true;
                obj.LastMouseX_ = cp(1);
                obj.LastMouseY_ = cp(2);
            end
        end
        
        function onMouseDrag_(obj, event)
            if obj.IsMouseBtnDown_
                cp = obj.Fig_.CurrentPoint;
                dx = cp(1) - obj.LastMouseX_;
                dy = cp(2) - obj.LastMouseY_;
                
                % Update camera Pitch (X) and Yaw (Y) sliders
                new_ry = obj.Cam_ry_ - dx * 0.5;
                new_rx = obj.Cam_rx_ - dy * 0.5;
                
                % Clamp Pitch
                new_rx = max(-90.0, min(90.0, new_rx));
                % Wrap Yaw
                if new_ry > 180, new_ry = new_ry - 360; end
                if new_ry < -180, new_ry = new_ry + 360; end
                
                obj.Cam_ry_ = new_ry;
                obj.Cam_rx_ = new_rx;
                
                % Update UI Slider controls
                obj.SliderControls_.Cam_ry_.slider.Value = new_ry;
                obj.SliderControls_.Cam_ry_.label.Text = sprintf('%.2f', new_ry);
                obj.SliderControls_.Cam_rx_.slider.Value = new_rx;
                obj.SliderControls_.Cam_rx_.label.Text = sprintf('%.2f', new_rx);
                
                obj.LastMouseX_ = cp(1);
                obj.LastMouseY_ = cp(2);
                
                obj.redraw_();
            end
        end
        
        function onMouseUp_(obj, event)
            obj.IsMouseBtnDown_ = false;
        end
        
        % --- Scene Importer parser ---
        function importScene_(obj)
            lines = obj.ImportTextArea_.Value;
            
            new_pts = {};
            new_lines = {};
            new_polys = {};
            
            for k = 1:numel(lines)
                l = strtrim(lines{k});
                if isempty(l) || startsWith(l, '#')
                    continue;
                end
                
                tokens = strsplit(l);
                cmd = upper(tokens{1});
                
                try
                    if strcmp(cmd, 'P')
                        if numel(tokens) < 4
                            error('Point command needs x, y, z.');
                        end
                        x = str2double(tokens{2});
                        y = str2double(tokens{3});
                        z = str2double(tokens{4});
                        new_pts{end+1} = [x; y; z; 1.0];
                        
                    elseif strcmp(cmd, 'L')
                        if numel(tokens) < 3
                            error('Line command needs 2 indices.');
                        end
                        idx1 = str2double(tokens{2});
                        idx2 = str2double(tokens{3});
                        new_lines{end+1} = [idx1, idx2];
                        
                    elseif strcmp(cmd, 'POLY')
                        if numel(tokens) < 4
                            error('Polygon command needs at least 3 indices.');
                        end
                        idxs = [];
                        for t = 2:numel(tokens)
                            idxs(end+1) = str2double(tokens{t});
                        end
                        new_polys{end+1} = idxs;
                    else
                        error('Unknown command: %s', cmd);
                    end
                catch ME
                    uialert(obj.Fig_, sprintf('Parse error at line %d:\n%s', k, ME.message), 'Import Error');
                    return;
                end
            end
            
            % Index validation
            nPts = numel(new_pts);
            for k = 1:numel(new_lines)
                i1 = new_lines{k}(1); i2 = new_lines{k}(2);
                if i1 >= nPts || i2 >= nPts || i1 < 0 || i2 < 0
                    uialert(obj.Fig_, sprintf('Line %d refers to out of bounds indices (total points: %d)', k, nPts), 'Validation Error');
                    return;
                end
            end
            for k = 1:numel(new_polys)
                poly = new_polys{k};
                for j = 1:numel(poly)
                    if poly(j) >= nPts || poly(j) < 0
                        uialert(obj.Fig_, sprintf('Polygon %d refers to out of bounds index %d (total points: %d)', k, poly(j), nPts), 'Validation Error');
                        return;
                    end
                end
            end
            
            % Success
            obj.Points_ = new_pts;
            obj.Lines_ = new_lines;
            obj.Polygons_ = new_polys;
            
            % Generate cyclic colors for imported polygons
            colors = {
                [0.17, 0.24, 0.31], [0.75, 0.22, 0.17], [0.15, 0.68, 0.38], ...
                [0.16, 0.50, 0.73], [0.95, 0.77, 0.06], [0.56, 0.27, 0.68], ...
                [0.09, 0.63, 0.52], [0.83, 0.33, 0.00]
            };
            obj.PolyColors_ = {};
            for k = 1:numel(new_polys)
                obj.PolyColors_{k} = colors{mod(k-1, numel(colors)) + 1};
            end
            
            obj.PolyTexCoords_ = {};
            for k = 1:numel(new_polys)
                nV = numel(new_polys{k});
                if nV == 3
                    obj.PolyTexCoords_{k} = [0, 1, 1; 1, 1, 1; 0.5, 0, 1];
                elseif nV == 4
                    obj.PolyTexCoords_{k} = [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1];
                else
                    theta_vals = linspace(0, 2*pi, nV+1)';
                    theta_vals = theta_vals(1:end-1);
                    obj.PolyTexCoords_{k} = [0.5 + 0.5*cos(theta_vals), 0.5 - 0.5*sin(theta_vals), ones(nV, 1)];
                end
            end
            
            obj.redraw_();
            uialert(obj.Fig_, sprintf('Successfully loaded scene:\n- %d points\n- %d lines\n- %d polygons', nPts, numel(new_lines), numel(new_polys)), 'Import Success', 'Icon', 'info');
        end
        
        function showPresetPopup_(obj)
            preset_win = uifigure('Name', 'Select Preset', 'Position', [200, 200, 300, 150], 'Color', obj.COLOR_PANEL);
            uilabel(preset_win, 'Text', 'Choose a 3D Preset Scene:', 'Position', [10, 95, 280, 25], 'FontColor', obj.COLOR_TEXT, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'center');
            
            uibutton(preset_win, 'Text', 'House', 'Position', [20, 35, 75, 32], 'BackgroundColor', obj.COLOR_ACCENT, 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~,~) obj.applyPreset_('House', preset_win));
            uibutton(preset_win, 'Text', 'Cube', 'Position', [110, 35, 75, 32], 'BackgroundColor', obj.COLOR_ACCENT, 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~,~) obj.applyPreset_('Cube', preset_win));
            uibutton(preset_win, 'Text', 'Pyramid', 'Position', [200, 35, 80, 32], 'BackgroundColor', obj.COLOR_ACCENT, 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~,~) obj.applyPreset_('Pyramid', preset_win));
        end
        
        function applyPreset_(obj, name, winHandle)
            if strcmp(name, 'House')
                obj.loadDefaultHouse_();
            elseif strcmp(name, 'Cube')
                obj.Points_ = {};
                obj.Lines_ = {};
                obj.Polygons_ = {};
                
                % 8 vertices
                vertices = [
                    -1.0, -1.0, -1.0; ...
                     1.0, -1.0, -1.0; ...
                     1.0,  1.0, -1.0; ...
                    -1.0,  1.0, -1.0; ...
                    -1.0, -1.0,  1.0; ...
                     1.0, -1.0,  1.0; ...
                     1.0,  1.0,  1.0; ...
                    -1.0,  1.0,  1.0
                ];
                for k = 1:8
                    obj.Points_{k} = [vertices(k,1); vertices(k,2); vertices(k,3); 1.0];
                end
                
                obj.Lines_ = {
                    [0,1], [1,2], [2,3], [3,0], ...
                    [4,5], [5,6], [6,7], [7,4], ...
                    [0,4], [1,5], [2,6], [3,7]
                };
                
                obj.Polygons_ = {
                    [0,1,2,3], [4,5,6,7], [0,1,5,4], ...
                    [1,2,6,5], [2,3,7,6], [3,0,4,7]
                };
                
                obj.PolyColors_ = {
                    [0.75, 0.22, 0.17], [0.15, 0.68, 0.38], [0.16, 0.50, 0.73], ...
                    [0.95, 0.77, 0.06], [0.56, 0.27, 0.68], [0.90, 0.49, 0.13]
                };
                
                obj.PolyTexCoords_ = {
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ...
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ...
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ...
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ...
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ...
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1]
                };
                
            elseif strcmp(name, 'Pyramid')
                obj.Points_ = {};
                obj.Lines_ = {};
                obj.Polygons_ = {};
                
                % 5 vertices (4 base + 1 apex)
                vertices = [
                    -1.0, -1.0, -1.0; ...
                     1.0, -1.0, -1.0; ...
                     1.0,  1.0, -1.0; ...
                    -1.0,  1.0, -1.0; ...
                     0.0,  0.0,  1.0
                ];
                for k = 1:5
                    obj.Points_{k} = [vertices(k,1); vertices(k,2); vertices(k,3); 1.0];
                end
                
                obj.Lines_ = {
                    [0,1], [1,2], [2,3], [3,0], ...
                    [0,4], [1,4], [2,4], [3,4]
                };
                
                obj.Polygons_ = {
                    [0,1,2,3], [0,1,4], [1,2,4], [2,3,4], [3,0,4]
                };
                
                obj.PolyColors_ = {
                    [0.17, 0.24, 0.31], [0.18, 0.80, 0.44], [0.20, 0.60, 0.86], ...
                    [0.91, 0.30, 0.24], [0.95, 0.77, 0.06]
                };
                
                obj.PolyTexCoords_ = {
                    [0, 1, 1; 1, 1, 1; 1, 0, 1; 0, 0, 1], ... % square base
                    [0, 1, 1; 1, 1, 1; 0.5, 0, 1], ...        % side 1
                    [0, 1, 1; 1, 1, 1; 0.5, 0, 1], ...        % side 2
                    [0, 1, 1; 1, 1, 1; 0.5, 0, 1], ...        % side 3
                    [0, 1, 1; 1, 1, 1; 0.5, 0, 1]              % side 4
                };
            end
            
            % Rebuild import text commands representation
            cmds = sprintf('# 3D %s Preset\n', name);
            for k = 1:numel(obj.Points_)
                p = obj.Points_{k};
                cmds = [cmds, sprintf('P %.2f %.2f %.2f\n', p(1), p(2), p(3))];
            end
            for k = 1:numel(obj.Lines_)
                l = obj.Lines_{k};
                cmds = [cmds, sprintf('L %d %d\n', l(1), l(2))];
            end
            for k = 1:numel(obj.Polygons_)
                p = obj.Polygons_{k};
                p_str = num2str(p);
                cmds = [cmds, sprintf('POLY %s\n', p_str)];
            end
            
            obj.ImportTextArea_.Value = strsplit(cmds, '\n');
            
            obj.redraw_();
            delete(winHandle);
        end

        % ================================================================== %
        % Texture mapping helpers (Ex 3.27)
        % ================================================================== %
        function toggleTexture_(obj, val)
            obj.EnableTexture_ = val;
            obj.redraw_();
        end
        
        function loadCustomTexture_(obj)
            [file, pth] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp', 'Images'});
            if isequal(file, 0), return; end
            img = imread(fullfile(pth, file));
            img_d = double(img) / 255.0;
            if size(img_d, 3) == 1
                img_d = repmat(img_d, 1, 1, 3);
            end
            obj.TextureImg_ = img_d;
            obj.redraw_();
        end

    end
end
