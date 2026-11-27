classdef LightSimulationSandbox < handle
    % LIGHTSIMULATIONSANDBOX Interactive 3D Room Light Simulation Dashboard.
    
    properties (Access = private)
        % Main window handles
        Fig_
        PanControl_
        PanCanvas_
        
        % Canvas Axes
        AxViewport_     % Main viewport axes (handles TPP and FPP)
        
        % Edit Fields: Camera Pose
        EditCam_X_
        EditCam_Y_
        EditCam_Z_
        EditCam_Pitch_
        EditCam_Yaw_
        EditCam_Zoom_
        
        % Edit Fields: Light Pose
        EditLight_X_
        EditLight_Y_
        EditLight_Z_
        EditLight_Power_
        
        % Edit Fields: Material Coefficients
        EditMat_Ka_
        EditMat_Kd_
        EditMat_Ks_
        EditMat_Shininess_
        EditMat_Reflect_
        
        % Dropdowns & Buttons
        Dropdown_Perspective_
        Dropdown_Renderer_
        Dropdown_Shading_
        Dropdown_Strategy_
        Btn_Render_
        Btn_Reset_
        
        % Core Simulation Objects
        Scene_
        Camera_
        AmbientLight_
        PointLight_
        SphereObject_
        FloorObject_
        CeilingObject_
        BackWallObject_
        LeftWallObject_
        RightWallObject_
        
        % Graphic handles in TPP Preview
        Handles3D_Objects_ = []
        Handle3D_CamFrustum_ = []
        Handle3D_LightStar_ = []
        
        % Visualizer theme adapter
        VizCore_
    end
    
    properties (Access = private, Constant)
        % Sleek Dark Mode colors matching Themes.json
        COLOR_BG        = [0.11, 0.11, 0.13]  % gunmetal background
        COLOR_PANEL     = [0.17, 0.19, 0.23]  % sidebar panel
        COLOR_TEXT      = [0.90, 0.90, 0.90]  % primary text
        COLOR_SUBTEXT   = [0.55, 0.58, 0.63]  % secondary text
        COLOR_ACCENT    = [0.20, 0.60, 1.00]  % ocean blue highlight
        COLOR_SECONDARY = [1.00, 0.40, 0.40]  % secondary highlight
        COLOR_MUTED     = [0.35, 0.37, 0.42]
        COLOR_YELLOW    = [1.00, 0.85, 0.20]  % light star color
        
        FIG_WIDTH       = 1000
        FIG_HEIGHT      = 750
        SIDEBAR_W       = 320
    end
    
    methods
        function obj = LightSimulationSandbox()
            % LIGHTSIMULATIONSANDBOX Construct the application and layout.
            obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
            
            % 1. Setup default simulation scene objects in memory
            obj.setupScene_();
            
            % 2. Draw the GUI layout
            obj.buildUI_();
            
            % 3. Sync controls to memory structures
            obj.onControlChanged_();
            
            % 4. Render initial view
            obj.updateSimulation_();
        end
        
        function show(obj)
            % SHOW Make the application window visible.
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end
        
        function delete(obj)
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
        
        function testPerspectiveSwitch(obj, mode)
            obj.Dropdown_Perspective_.Value = mode;
            obj.onPerspectiveModeChanged_();
        end
        
        function testRendererSwitch(obj, renderer)
            obj.Dropdown_Renderer_.Value = renderer;
            obj.onRenderPressed_();
        end
        
        function testReset(obj)
            obj.onResetPressed_();
        end
        
        function debugLighting(obj)
            import OceanMath.Graphics.Shading.BlinnPhongStrategy
            scene = obj.Scene_;
            lights = scene.Lights;
            strategy = BlinnPhongStrategy();
            
            fprintf('Number of objects in scene: %d\n', numel(scene.Objects));
            for i = 1:numel(scene.Objects)
                sceneObj = scene.Objects{i};
                [V, F] = sceneObj.toMesh();
                if isa(sceneObj.Geometry, 'OceanMath.Vision.Geometry.SphereGeometry')
                    geomPoint = sceneObj.Geometry.Center;
                    geomNormal = [0;0;0];
                else
                    geomPoint = sceneObj.Geometry.Point;
                    geomNormal = sceneObj.Geometry.Normal;
                end
                fprintf('Object %d (%s): Point=[%.2f, %.2f, %.2f], Normal=[%.2f, %.2f, %.2f]\n', ...
                    i, class(sceneObj.Geometry), geomPoint(1), geomPoint(2), geomPoint(3), ...
                    geomNormal(1), geomNormal(2), geomNormal(3));
                
                % Compute lighting for first vertex
                v_pos = V(:, 1);
                pt_local = inv(sceneObj.Pose) * [v_pos; 1];
                N_local = sceneObj.Geometry.getNormal(pt_local(1:3));
                R_sub = sceneObj.Pose(1:3, 1:3);
                N_world = (inv(R_sub)') * N_local(:);
                N_world = N_world / norm(N_world);
                
                T_c2w = inv(obj.Camera_.Extrinsics);
                camPos = T_c2w(1:3, 4);
                viewDir = camPos - v_pos;
                viewDir = viewDir / norm(viewDir);
                
                light = lights{2};
                [L_intensity, L_dir, L_dist] = light.getLightAt(v_pos);
                shadowO = v_pos(:) + 1e-4 * N_world;
                [hitObj, t_shadow, ~] = scene.intersect(shadowO, L_dir(:));
                
                isShadowed = false;
                if ~isnan(t_shadow) && t_shadow < L_dist
                    isShadowed = true;
                end
                
                color = strategy.computeLighting(v_pos, N_world, viewDir, sceneObj.Material, lights, scene);
                
                fprintf('  Vertex 1: [%.2f, %.2f, %.2f]\n', v_pos(1), v_pos(2), v_pos(3));
                fprintf('    N_world: [%.2f, %.2f, %.2f]\n', N_world(1), N_world(2), N_world(3));
                fprintf('    L_dir: [%.2f, %.2f, %.2f], L_dist: %.4f\n', L_dir(1), L_dir(2), L_dir(3), L_dist);
                if ~isempty(hitObj)
                    fprintf('    t_shadow: %.4f (Hit object: %s)\n', t_shadow, class(hitObj.Geometry));
                else
                    fprintf('    t_shadow: NaN (Hit object: None)\n');
                end
                fprintf('    isShadowed: %d\n', isShadowed);
                fprintf('    Color: [%.4f, %.4f, %.4f]\n', color(1), color(2), color(3));
            end
        end
    end
    
    methods (Access = private)
        
        function setupScene_(obj)
            import OceanMath.Vision.Geometry.PlaneGeometry
            import OceanMath.Vision.Geometry.SphereGeometry
            import OceanMath.Graphics.Scene.Material
            import OceanMath.Graphics.Scene.SceneObject
            import OceanMath.Graphics.Scene.Scene
            import OceanMath.Graphics.Lighting.AmbientLight
            import OceanMath.Graphics.Lighting.PointLight
            import OceanMath.Geometry.CameraProjection
            import OceanMath.Geometry.Transform3D
            
            obj.Scene_ = Scene();
            
            % Core Materials
            matSphere = Material('Ambient', [0.1; 0.0; 0.0], 'Diffuse', [0.8; 0.0; 0.0], ...
                                 'Specular', [0.7; 0.7; 0.7], 'Shininess', 32.0, 'Reflectivity', [0.2; 0.2; 0.2]);
            matRoom   = Material('Ambient', [0.1; 0.1; 0.1], 'Diffuse', [0.7; 0.7; 0.7], ...
                                 'Specular', [0.1; 0.1; 0.1], 'Shininess', 8.0);
            matGreen  = Material('Ambient', [0.0; 0.1; 0.0], 'Diffuse', [0.0; 0.7; 0.0]);
            matBlue   = Material('Ambient', [0.0; 0.0; 0.1], 'Diffuse', [0.0; 0.0; 0.7]);
            
            % Room Dimensions: Floor, Ceiling, Left, Right, Back walls fitting exactly:
            % Width = 8 (X: -4 to 4), Height = 4 (Y: -2 to 2), Depth = 6 (Z: 0 to 6)
            obj.FloorObject_ = SceneObject(PlaneGeometry('Point', [0; -2; 3], 'Normal', [0; 1; 0], 'IsFinite', true, 'Width', 3, 'Height', 4), matRoom);
            obj.CeilingObject_ = SceneObject(PlaneGeometry('Point', [0; 2; 3], 'Normal', [0; -1; 0], 'IsFinite', true, 'Width', 3, 'Height', 4), matRoom);
            obj.LeftWallObject_ = SceneObject(PlaneGeometry('Point', [-4; 0; 3], 'Normal', [1; 0; 0], 'IsFinite', true, 'Width', 3, 'Height', 2), matGreen);
            obj.RightWallObject_ = SceneObject(PlaneGeometry('Point', [4; 0; 3], 'Normal', [-1; 0; 0], 'IsFinite', true, 'Width', 3, 'Height', 2), matBlue);
            obj.BackWallObject_ = SceneObject(PlaneGeometry('Point', [0; 0; 6], 'Normal', [0; 0; -1], 'IsFinite', true, 'Width', 2, 'Height', 4), matRoom);
            
            % Red Sphere at [0; 0; 4]
            obj.SphereObject_ = SceneObject(SphereGeometry('Center', [0; 0; 0], 'Radius', 1.2), matSphere, ...
                                             'Pose', Transform3D.translation(0, 0, 4));
            
            % Add all objects to scene
            obj.Scene_.addObject(obj.FloorObject_);
            obj.Scene_.addObject(obj.CeilingObject_);
            obj.Scene_.addObject(obj.LeftWallObject_);
            obj.Scene_.addObject(obj.RightWallObject_);
            obj.Scene_.addObject(obj.BackWallObject_);
            obj.Scene_.addObject(obj.SphereObject_);
            
            % Add Lights
            obj.AmbientLight_ = AmbientLight('Intensity', [0.15; 0.15; 0.15]);
            obj.PointLight_   = PointLight('Position', [0; 1.5; 3.0], 'Intensity', [1.0; 1.0; 1.0]);
            obj.Scene_.addLight(obj.AmbientLight_);
            obj.Scene_.addLight(obj.PointLight_);
            
            % Camera definition: 320x240 sensor projection coordinates
            K = [300, 0, 160; 0, 300, 120; 0, 0, 1];
            obj.Camera_ = CameraProjection('Intrinsics', K, 'Extrinsics', eye(4));
        end
        
        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0  = (scr(3) - obj.FIG_WIDTH)  / 2;
            y0  = (scr(4) - obj.FIG_HEIGHT) / 2;
            
            obj.Fig_ = uifigure( ...
                'Name',            '3D Room Light Simulation Sandbox — Ocean Library', ...
                'Position',        [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color',           obj.COLOR_BG, ...
                'Resize',          'off');
            
            % ---------------- Sidebar Controls Panel ----------------
            obj.PanControl_ = uipanel(obj.Fig_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 730], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
            
            % ---------------- Single Viewport Canvas Panel ----------------
            canvasWidth = obj.FIG_WIDTH - obj.SIDEBAR_W - 30;
            obj.PanCanvas_ = uipanel(obj.Fig_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, canvasWidth, 730], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
            
            % ---------------- Left Controls Content ----------------
            yPos = 700;
            
            % Title
            uilabel(obj.PanControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Simulation Settings', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Perspective Dropdown
            yPos = yPos - 35;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 100, 20], 'Text', 'Perspective:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Dropdown_Perspective_ = uidropdown(obj.PanControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Third-Person (TPP)', 'First-Person (FPP)'}, ...
                'Value', 'Third-Person (TPP)', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onPerspectiveModeChanged_());
                
            % Rendering Mode Dropdown
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 100, 20], 'Text', 'Rendering Mode:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Dropdown_Renderer_ = uidropdown(obj.PanControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Vector 2D viewport', 'Ray Trace viewport'}, ...
                'Value', 'Vector 2D viewport', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'Enable', 'off', ...
                'ValueChangedFcn', @(~,~) obj.onRenderPressed_());
                
            % Shading Style Dropdown
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 100, 20], 'Text', 'Shading Style:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Dropdown_Shading_ = uidropdown(obj.PanControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Flat Shading', 'Gouraud Shading'}, ...
                'Value', 'Gouraud Shading', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onRenderPressed_());
                
            % Lighting Strategy Dropdown
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 100, 20], 'Text', 'Lighting strategy:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Dropdown_Strategy_ = uidropdown(obj.PanControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Lambertian', 'Phong', 'Blinn-Phong'}, ...
                'Value', 'Blinn-Phong', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onRenderPressed_());
            
            % Section 1: Camera Pose Edit Fields
            yPos = yPos - 35;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            yPos = yPos - 25;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Camera Settings', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 11, 'FontWeight', 'bold');
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Camera X (left/right):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditCam_X_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.0, 'Limits', [-3, 3], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Camera Y (up/down):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditCam_Y_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.0, 'Limits', [-1.5, 1.5], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Camera Z (depth):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditCam_Z_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.0, 'Limits', [-3, 2], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Camera Pitch (tilt):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditCam_Pitch_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.0, 'Limits', [-30, 30], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Camera Yaw (pan):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditCam_Yaw_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.0, 'Limits', [-45, 45], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Camera Zoom (focal):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditCam_Zoom_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 300.0, 'Limits', [100, 1000], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            % Section 2: Point Light Pose Edit Fields
            yPos = yPos - 35;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            yPos = yPos - 25;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Light Source Settings', 'FontColor', obj.COLOR_YELLOW, 'FontSize', 11, 'FontWeight', 'bold');
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Light X:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditLight_X_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.0, 'Limits', [-3.8, 3.8], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Light Y (height):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditLight_Y_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 1.5, 'Limits', [-1.8, 1.8], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Light Z:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditLight_Z_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 3.0, 'Limits', [0.5, 5.5], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Light Power (RGB):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditLight_Power_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 1.0, 'Limits', [0.0, 2.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            % Section 3: Material Edit Fields
            yPos = yPos - 35;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            yPos = yPos - 25;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Sphere Material Settings', 'FontColor', obj.COLOR_SECONDARY, 'FontSize', 11, 'FontWeight', 'bold');
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Ambient (Ka):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditMat_Ka_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.1, 'Limits', [0.0, 1.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Diffuse (Kd):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditMat_Kd_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.8, 'Limits', [0.0, 1.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Specular (Ks):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditMat_Ks_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.7, 'Limits', [0.0, 1.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Shininess (alpha):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditMat_Shininess_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 32.0, 'Limits', [1.0, 128.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            yPos = yPos - 28;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 120, 20], 'Text', 'Reflectivity (Kr):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.EditMat_Reflect_ = uieditfield(obj.PanControl_, 'numeric', 'Position', [170, yPos, 120, 22], 'Value', 0.2, 'Limits', [0.0, 1.0], 'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT);
            
            % Bottom Control Action Buttons
            yPos = yPos - 35;
            uilabel(obj.PanControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            yPos = yPos - 40;
            obj.Btn_Render_ = uibutton(obj.PanControl_, ...
                'Position', [20, yPos, 270, 30], ...
                'Text', 'Render / Update View', ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'BackgroundColor', obj.COLOR_ACCENT, ...
                'FontColor', [1; 1; 1], ...
                'ButtonPushedFcn', @(~,~) obj.onRenderPressed_());
                
            yPos = yPos - 35;
            obj.Btn_Reset_ = uibutton(obj.PanControl_, ...
                'Position', [20, yPos, 270, 26], ...
                'Text', 'Reset Defaults', ...
                'FontSize', 11, ...
                'FontWeight', 'bold', ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1; 1; 1], ...
                'ButtonPushedFcn', @(~,~) obj.onResetPressed_());
            
            % ---------------- Canvas Axis viewport ----------------
            obj.AxViewport_ = uiaxes(obj.PanCanvas_, ...
                'Position', [0, 10, canvasWidth, 710], ...
                'Color', [0.08, 0.10, 0.14]);
            obj.VizCore_.styleAxes(obj.AxViewport_);
            grid(obj.AxViewport_, 'on');
        end
        
        function onControlChanged_(obj)
            % Read settings from UI inputs into memory only
            camX = obj.EditCam_X_.Value;
            camY = obj.EditCam_Y_.Value;
            camZ = obj.EditCam_Z_.Value;
            camP = obj.EditCam_Pitch_.Value;
            camYw = obj.EditCam_Yaw_.Value;
            camZm = obj.EditCam_Zoom_.Value;
            
            lightX = obj.EditLight_X_.Value;
            lightY = obj.EditLight_Y_.Value;
            lightZ = obj.EditLight_Z_.Value;
            lightPower = obj.EditLight_Power_.Value;
            
            matKa = obj.EditMat_Ka_.Value;
            matKd = obj.EditMat_Kd_.Value;
            matKs = obj.EditMat_Ks_.Value;
            matSh = obj.EditMat_Shininess_.Value;
            matRf = obj.EditMat_Reflect_.Value;
            
            % Update Scene Object state in memory
            obj.PointLight_.Position = [lightX; lightY; lightZ];
            obj.PointLight_.Intensity = [lightPower; lightPower; lightPower];
            
            obj.SphereObject_.Material.Ambient = [matKa; 0.0; 0.0];
            obj.SphereObject_.Material.Diffuse = [matKd; 0.0; 0.0];
            obj.SphereObject_.Material.Specular = [matKs; matKs; matKs];
            obj.SphereObject_.Material.Shininess = matSh;
            obj.SphereObject_.Material.Reflectivity = [matRf; matRf; matRf];
            
            % Update Camera Extrinsics: Position in world + rotations
            import OceanMath.Geometry.Rotation3D
            R_c2w = Rotation3D.eulerToMatrix(0, camP, camYw);
            camPos = [camX; camY; camZ];
            T_w2c = [R_c2w', -R_c2w'*camPos; 0, 0, 0, 1];
            obj.Camera_.Extrinsics = T_w2c;
            
            % Update Intrinsics (focal length zoom)
            obj.Camera_.Intrinsics(1,1) = camZm;
            obj.Camera_.Intrinsics(2,2) = camZm;
        end
        
        function onPerspectiveModeChanged_(obj)
            mode = obj.Dropdown_Perspective_.Value;
            if strcmp(mode, 'Third-Person (TPP)')
                obj.Dropdown_Renderer_.Enable = 'off';
                obj.Dropdown_Shading_.Enable = 'on';
                obj.Dropdown_Strategy_.Enable = 'on';
            else
                obj.Dropdown_Renderer_.Enable = 'on';
                obj.Dropdown_Shading_.Enable = 'on';
                obj.Dropdown_Strategy_.Enable = 'on';
            end
            
            % Immediately redraw the viewport
            obj.onRenderPressed_();
        end
        
        function onRenderPressed_(obj)
            % Commit typed settings to memory
            obj.onControlChanged_();
            
            % Redraw viewport
            obj.updateSimulation_();
        end
        
        function updateSimulation_(obj)
            import OceanMath.Graphics.Renderer.VectorMeshRenderer
            import OceanMath.Graphics.Renderer.RayTraceRenderer
            import OceanMath.Graphics.Shading.LambertianStrategy
            import OceanMath.Graphics.Shading.PhongStrategy
            import OceanMath.Graphics.Shading.BlinnPhongStrategy
            
            % 1. Clean up axes graphics handles
            if ~isempty(obj.Handles3D_Objects_)
                delete(obj.Handles3D_Objects_(ishandle(obj.Handles3D_Objects_)));
                obj.Handles3D_Objects_ = [];
            end
            if ~isempty(obj.Handle3D_CamFrustum_)
                delete(obj.Handle3D_CamFrustum_(ishandle(obj.Handle3D_CamFrustum_)));
                obj.Handle3D_CamFrustum_ = [];
            end
            if ~isempty(obj.Handle3D_LightStar_)
                delete(obj.Handle3D_LightStar_(ishandle(obj.Handle3D_LightStar_)));
                obj.Handle3D_LightStar_ = [];
            end
            
            % Reset axes state
            cla(obj.AxViewport_);
            legend(obj.AxViewport_, 'off');
            
            % 2. Establish Shading strategy
            stratVal = obj.Dropdown_Strategy_.Value;
            if strcmp(stratVal, 'Lambertian')
                strategy = LambertianStrategy();
            elseif strcmp(stratVal, 'Phong')
                strategy = PhongStrategy();
            else
                strategy = BlinnPhongStrategy();
            end
            
            % 3. Check perspective mode
            perspValue = obj.Dropdown_Perspective_.Value;
            
            if strcmp(perspValue, 'Third-Person (TPP)')
                % TPP: Renders the 3D room externally and overlays the camera mesh and light source star
                shMode = "Gouraud";
                if strcmp(obj.Dropdown_Shading_.Value, 'Flat Shading')
                    shMode = "Flat";
                end
                
                % Render 3D scene mesh
                vectorPreview = VectorMeshRenderer('RenderDimension', '3D', 'ShadingMode', shMode);
                obj.Handles3D_Objects_ = vectorPreview.render(obj.Scene_, obj.Camera_, ...
                                                              'Axes', obj.AxViewport_, 'LightingStrategy', strategy);
                
                % Make room walls (objects 1 to 5) translucent (FaceAlpha = 0.15) to see inside clearly
                for k = 1:min(5, numel(obj.Handles3D_Objects_))
                    if ishandle(obj.Handles3D_Objects_(k))
                        set(obj.Handles3D_Objects_(k), 'FaceAlpha', 0.15, 'EdgeColor', obj.COLOR_TEXT * 0.4, 'LineWidth', 1.0);
                    end
                end
                
                % Compute Camera pose in world
                T_c2w = inv(obj.Camera_.Extrinsics);
                cPos = T_c2w(1:3, 4);
                
                % Compute camera frustum corners dynamically based on current Zoom (focal length)
                fx = obj.Camera_.Intrinsics(1,1);
                fy = obj.Camera_.Intrinsics(2,2);
                d = 1.2; % Depth of frustum display plane
                w_c = (160 / fx) * d;
                h_c = (120 / fy) * d;
                corners_cam = [ -w_c,  w_c,  w_c, -w_c; ...
                                -h_c, -h_c,  h_c,  h_c; ...
                                 d,    d,    d,    d ];
                corners_world = T_c2w * [corners_cam; ones(1, 4)];
                
                % Camera body vertices in camera space
                body_cam = [
                    -0.20, -0.15, -0.35; % 1
                     0.20, -0.15, -0.35; % 2
                     0.20,  0.15, -0.35; % 3
                    -0.20,  0.15, -0.35; % 4
                    -0.20, -0.15,  0.00; % 5
                     0.20, -0.15,  0.00; % 6
                     0.20,  0.15,  0.00; % 7
                    -0.20,  0.15,  0.00  % 8
                ]';
                body_faces = [
                    1, 2, 3, 4; % back
                    5, 6, 7, 8; % front
                    1, 2, 6, 5; % bottom
                    2, 3, 7, 6; % right
                    3, 4, 8, 7; % top
                    4, 1, 5, 8  % left
                ];
                
                % Camera lens: cylinder/cone extending from Z=0 to Z=0.25
                theta_lens = linspace(0, 2*pi, 9);
                lens_pts = zeros(3, 16);
                for k = 1:8
                    % Base circle Z=0
                    lens_pts(:, 2*k - 1) = [0.08 * cos(theta_lens(k)); 0.08 * sin(theta_lens(k)); 0.0];
                    % Tip circle Z=0.25
                    lens_pts(:, 2*k)     = [0.12 * cos(theta_lens(k)); 0.12 * sin(theta_lens(k)); 0.25];
                end
                
                lens_faces = zeros(8, 4);
                for k = 1:8
                    idx1 = 2*k - 1;
                    idx2 = 2*k;
                    idx3 = 2*mod(k, 8) + 1;
                    idx4 = 2*mod(k, 8) + 2;
                    lens_faces(k, :) = [idx1, idx2, idx4, idx3];
                end
                
                % Transform coordinates to world space
                body_world = T_c2w * [body_cam; ones(1, size(body_cam, 2))];
                lens_world = T_c2w * [lens_pts; ones(1, size(lens_pts, 2))];
                
                hold(obj.AxViewport_, 'on');
                
                % Draw frustum lines
                hFrust = [];
                hFrust(end+1) = plot3(obj.AxViewport_, [cPos(1), corners_world(1,1)], [cPos(2), corners_world(2,1)], [cPos(3), corners_world(3,1)], 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.5);
                hFrust(end+1) = plot3(obj.AxViewport_, [cPos(1), corners_world(1,2)], [cPos(2), corners_world(2,2)], [cPos(3), corners_world(3,2)], 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.5);
                hFrust(end+1) = plot3(obj.AxViewport_, [cPos(1), corners_world(1,3)], [cPos(2), corners_world(2,3)], [cPos(3), corners_world(3,3)], 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.5);
                hFrust(end+1) = plot3(obj.AxViewport_, [cPos(1), corners_world(1,4)], [cPos(2), corners_world(2,4)], [cPos(3), corners_world(3,4)], 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.5);
                hFrust(end+1) = plot3(obj.AxViewport_, corners_world(1, [1, 2, 3, 4, 1]), corners_world(2, [1, 2, 3, 4, 1]), corners_world(3, [1, 2, 3, 4, 1]), 'Color', obj.COLOR_ACCENT, 'LineWidth', 1.5);
                obj.Handle3D_CamFrustum_ = hFrust;
                
                % Draw solid camera model
                hBody = patch(obj.AxViewport_, 'Vertices', body_world(1:3, :)', 'Faces', body_faces, ...
                              'FaceColor', obj.COLOR_ACCENT * 0.5, 'EdgeColor', obj.COLOR_TEXT * 0.9, 'LineWidth', 1.0);
                hLens = patch(obj.AxViewport_, 'Vertices', lens_world(1:3, :)', 'Faces', lens_faces, ...
                              'FaceColor', [0.12, 0.14, 0.18], 'EdgeColor', obj.COLOR_TEXT * 0.9, 'LineWidth', 1.0);
                obj.Handles3D_Objects_ = [obj.Handles3D_Objects_(:)', hBody, hLens];
                
                % Plot yellow star at light source
                obj.Handle3D_LightStar_ = plot3(obj.AxViewport_, obj.PointLight_.Position(1), obj.PointLight_.Position(2), obj.PointLight_.Position(3), ...
                                                '*', 'Color', obj.COLOR_YELLOW, 'MarkerSize', 10, 'LineWidth', 2.0);
                
                % Configure 3D viewing axes
                view(obj.AxViewport_, [-35, 20]);
                grid(obj.AxViewport_, 'on');
                axis(obj.AxViewport_, 'equal');
                xlim(obj.AxViewport_, [-5, 5]);
                ylim(obj.AxViewport_, [-3, 3]);
                zlim(obj.AxViewport_, [-1, 7]);
                xlabel(obj.AxViewport_, 'X (World)');
                ylabel(obj.AxViewport_, 'Y (World)');
                zlabel(obj.AxViewport_, 'Z (World)');
                title(obj.AxViewport_, 'Third-Person Perspective (TPP) Scene View', 'Color', obj.COLOR_TEXT);
                
            else
                % FPP Mode (First-Person Perspective - Camera's POV)
                view(obj.AxViewport_, 2);
                rendererMode = obj.Dropdown_Renderer_.Value;
                
                if strcmp(rendererMode, 'Vector 2D viewport')
                    shMode = "Gouraud";
                    if strcmp(obj.Dropdown_Shading_.Value, 'Flat Shading')
                        shMode = "Flat";
                    end
                    
                    vector2D = VectorMeshRenderer('RenderDimension', '2D', 'ShadingMode', shMode);
                    vector2D.render(obj.Scene_, obj.Camera_, 'Axes', obj.AxViewport_, 'LightingStrategy', strategy);
                    grid(obj.AxViewport_, 'on');
                    axis(obj.AxViewport_, 'equal');
                    xlim(obj.AxViewport_, [0, obj.Camera_.Intrinsics(1,3) * 2]);
                    ylim(obj.AxViewport_, [0, obj.Camera_.Intrinsics(2,3) * 2]);
                    set(obj.AxViewport_, 'YDir', 'reverse');
                    xlabel(obj.AxViewport_, 'X (Pixels)');
                    ylabel(obj.AxViewport_, 'Y (Pixels)');
                    title(obj.AxViewport_, 'First-Person Perspective (FPP) - Projected Vector Viewport', 'Color', obj.COLOR_TEXT);
                    
                else
                    % Ray-Traced raster
                    resWidth = 160;
                    resHeight = 120;
                    
                    % Disable UI interaction during render
                    obj.Btn_Render_.Text = 'Rendering Ray Trace...';
                    obj.Btn_Render_.Enable = 'off';
                    drawnow();
                    
                    tracer = RayTraceRenderer('Resolution', [resWidth, resHeight], 'MaxDepth', 1);
                    img = tracer.render(obj.Scene_, obj.Camera_, 'LightingStrategy', strategy);
                    
                    image(obj.AxViewport_, img);
                    axis(obj.AxViewport_, 'image');
                    grid(obj.AxViewport_, 'off');
                    set(obj.AxViewport_, 'XTick', [], 'YTick', []);
                    title(obj.AxViewport_, sprintf('First-Person Perspective (FPP) - Ray-Traced viewport (%dx%d)', resWidth, resHeight), 'Color', obj.COLOR_TEXT);
                    
                    obj.Btn_Render_.Text = 'Render / Update View';
                    obj.Btn_Render_.Enable = 'on';
                end
            end
        end
        
        function onResetPressed_(obj)
            % Reset all fields to defaults
            obj.EditCam_X_.Value = 0.0;
            obj.EditCam_Y_.Value = 0.0;
            obj.EditCam_Z_.Value = 0.0;
            obj.EditCam_Pitch_.Value = 0.0;
            obj.EditCam_Yaw_.Value = 0.0;
            obj.EditCam_Zoom_.Value = 300.0;
            
            obj.EditLight_X_.Value = 0.0;
            obj.EditLight_Y_.Value = 1.5;
            obj.EditLight_Z_.Value = 3.0;
            obj.EditLight_Power_.Value = 1.0;
            
            obj.EditMat_Ka_.Value = 0.1;
            obj.EditMat_Kd_.Value = 0.8;
            obj.EditMat_Ks_.Value = 0.7;
            obj.EditMat_Shininess_.Value = 32.0;
            obj.EditMat_Reflect_.Value = 0.2;
            
            obj.Dropdown_Perspective_.Value = 'Third-Person (TPP)';
            obj.Dropdown_Renderer_.Value = 'Vector 2D viewport';
            obj.Dropdown_Shading_.Value = 'Gouraud Shading';
            obj.Dropdown_Strategy_.Value = 'Blinn-Phong';
            
            obj.onPerspectiveModeChanged_();
            obj.onControlChanged_();
            obj.updateSimulation_();
        end
    end
end
