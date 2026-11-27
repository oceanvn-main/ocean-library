classdef ComputerVisionSandbox < handle
    % COMPUTERVISIONSANDBOX Main GUI application for Computer Vision Primitives.
    %
    % This unified programmatic application lets you interactively explore:
    %   1. 2D Transformations (Affine, Homography, Warp)
    %   2. 3D Rotations & Slerp (Euler, Quaternion, Axis-Angle, Exp Map)
    %   3. 3D-to-2D Camera Projections (Perspective, Orthographic, Weak)
    %   4. Lens Distortions (Radial & Tangential aberrations)
    %
    % Built as a plain MATLAB OOP class (no binary .mlapp) for perfect VCS integration.
    %
    % Usage:
    %   app = ComputerVisionSandbox();
    %   app.show();
    
    properties (Access = private)
        % Main window handles
        Fig_
        TabGroup_
        Tab2D_
        Tab3D_
        TabCam_
        TabDist_
        
        % ---- Tab 1: 2D Transformations ----
        % Containers
        Pan2DControl_
        Pan2DCanvas_
        Pan2DInfo_
        
        % Sliders & Input handles
        Slider2D_Tx_
        Slider2D_Ty_
        Slider2D_Rot_
        Slider2D_Sx_
        Slider2D_Sy_
        Slider2D_Shx_
        Slider2D_Shy_
        
        % Labels for readouts
        Lbl2D_Tx_
        Lbl2D_Ty_
        Lbl2D_Rot_
        Lbl2D_Sx_
        Lbl2D_Sy_
        Lbl2D_Shx_
        Lbl2D_Shy_
        
        % Matrix input fields (3x3 grid)
        Grid2D_Matrix_ % cell array of 9 uieditfields
        Chk2D_AutoApply_
        Btn2D_ApplyMatrix_
        Btn2D_Reset_
        
        % Visualizer handles
        Ax2D_
        Grid2D_Ref_      % Line handle for original grid
        Grid2D_Warped_   % Line handle for warped grid
        Shape2D_Ref_     % Patch/Line handle for original shape
        Shape2D_Warped_  % Patch/Line handle for warped shape
        
        % Tab 1 State
        State2D_Tx_ = 0
        State2D_Ty_ = 0
        State2D_Rot_ = 0
        State2D_Sx_ = 1
        State2D_Sy_ = 1
        State2D_Shx_ = 0
        State2D_Shy_ = 0
        State2D_Matrix_ = eye(3)
        
        % Reference data
        RefGrid2DPoints_
        RefShape2DPoints_
        
        % ---- Tab 2: 3D Rotations & Slerp ----
        % Containers
        Pan3DControl_
        Pan3DCanvas_
        Pan3DInfo_
        
        % Formats and panels
        Dropdown3D_Format_
        Pan3D_Euler_
        Pan3D_Quat_
        Pan3D_AxisAngle_
        Pan3D_ExpMap_
        
        % Sliders: Euler
        Slider3D_Roll_
        Slider3D_Pitch_
        Slider3D_Yaw_
        Lbl3D_Roll_
        Lbl3D_Pitch_
        Lbl3D_Yaw_
        
        % Inputs: Quaternions
        Field3D_Qw_
        Field3D_Qx_
        Field3D_Qy_
        Field3D_Qz_
        
        % Inputs: Axis-Angle
        Field3D_Ax_
        Field3D_Ay_
        Field3D_Az_
        Slider3D_Ang_
        Lbl3D_Ang_
        
        % Sliders: Exp Map
        Slider3D_V1_
        Slider3D_V2_
        Slider3D_V3_
        Lbl3D_V1_
        Lbl3D_V2_
        Lbl3D_V3_
        
        % Slerp panel elements
        Btn3D_SetStart_
        Btn3D_SetEnd_
        Slider3D_SlerpT_
        Lbl3D_SlerpT_
        Btn3D_AnimateSlerp_
        Btn3D_Reset_
        
        % Canvas elements
        Ax3D_
        Cube3D_Warped_
        LocalAxisX_
        LocalAxisY_
        LocalAxisZ_
        
        % Right side matrix display & readouts
        Grid3D_Matrix_ % cell array of 9 uilabels
        Lbl3D_Quat_
        Lbl3D_Exp_
        
        % Tab 2 State
        State3D_R_ = eye(3)
        State3D_QStart_ = [1, 0, 0, 0]
        State3D_QEnd_ = [1, 0, 0, 0]
        SlerpTimer_
        SlerpActive_ = false
        RefCubeVertices_
        RefCubeFaces_
        
        % ---- Tab 3: Camera Projection ----
        % Containers
        PanCamControl_
        PanCamCanvas_
        PanCamInfo_
        
        % Sliders: Intrinsics
        SliderCam_Fx_
        SliderCam_Fy_
        SliderCam_Cx_
        SliderCam_Cy_
        SliderCam_Skew_
        LblCam_Fx_
        LblCam_Fy_
        LblCam_Cx_
        LblCam_Cy_
        LblCam_Skew_
        
        % Sliders: Extrinsics (Camera Position / World Pose)
        SliderCam_X_
        SliderCam_Y_
        SliderCam_Z_
        SliderCam_Roll_
        SliderCam_Pitch_
        SliderCam_Yaw_
        LblCam_X_
        LblCam_Y_
        LblCam_Z_
        LblCam_Roll_
        LblCam_Pitch_
        LblCam_Yaw_
        
        DropdownCam_Type_
        BtnCam_Reset_
        
        % Canvas Axes
        AxCam3D_ % Left: 3D world scene view
        AxCam2D_ % Right: 2D Projected View
        
        % Plotted handles (3D world)
        House3D_Line_     % House in 3D
        Frustum3D_Line_   % wireframe pyramid frustum
        CamOrigin3D_Pt_   % dot at camera origin
        
        % Plotted handles (2D sensor projection)
        House2D_Line_     % projected house in 2D
        
        % Tab 3 State
        StateCam_Fx_ = 500
        StateCam_Fy_ = 500
        StateCam_Cx_ = 320
        StateCam_Cy_ = 240
        StateCam_Skew_ = 0
        StateCam_X_ = 0
        StateCam_Y_ = 0
        StateCam_Z_ = -8
        StateCam_Roll_ = 0
        StateCam_Pitch_ = 0
        StateCam_Yaw_ = 0
        
        % Reference 3D House wireframe
        RefHouseVertices_
        RefHouseLines_
        
        % ---- Tab 4: Lens Distortions ----
        % Containers
        PanDistControl_
        PanDistCanvas_
        PanDistInfo_
        
        % Sliders: aberration coefficients
        SliderDist_K1_
        SliderDist_K2_
        SliderDist_K3_
        SliderDist_P1_
        SliderDist_P2_
        LblDist_K1_
        LblDist_K2_
        LblDist_K3_
        LblDist_P1_
        LblDist_P2_
        BtnDist_Reset_
        
        % Canvas elements
        AxDist_
        DistPoints_Ref_    % Small gray dots
        DistPoints_Warped_ % Warped cyan circles
        DistLines_Warped_  % Warped grid lines
        
        % Tab 4 State
        StateDist_K1_ = 0.0
        StateDist_K2_ = 0.0
        StateDist_K3_ = 0.0
        StateDist_P1_ = 0.0
        StateDist_P2_ = 0.0
        RefDistPoints_      % [2 x M] matrix of normalized points
        RefDistLineIndices_ % Connection cell array
        
        % ---- Tab 5: Image Pyramids ----
        % Containers
        TabPyr_
        PanPyrControl_
        PanPyrCanvas_
        PanPyrInfo_
        
        % Controls
        DropdownPyr_Image_
        BtnPyr_LoadCustom_
        DropdownPyr_Mode_
        SliderPyr_Levels_
        LblPyr_Levels_
        DropdownPyr_Kernel_
        SliderPyr_KernelSize_
        LblPyr_KernelSize_
        BtnPyr_Reset_
        
        % Canvas elements
        AxPyr_             % 1x6 cell array of axes
        ImgPyr_            % 1x6 cell array of image plot objects
        
        % State variables
        StatePyr_ImageName_ = 'ocean_img.jpg'
        StatePyr_Image_     % Full loaded original double image
        StatePyr_Mode_ = 'Gaussian'
        StatePyr_Levels_ = 4
        StatePyr_Kernel_ = 'Gaussian'
        StatePyr_KernelSize_ = 5
        StatePyr_Bypass_ = false  % Bypass low-pass filter before decimation
        
        % Bypass filter checkbox (Tab 5)
        ChkPyr_Bypass_
        BtnPyr_Theory_
        
        % Math text labels in the Right Panel
        LblPyr_Info_Title_
        LblPyr_Info_Desc_
        LblPyr_Info_Stats_
        
        % ---- Tab 6: Image Resampling ----
        TabResample_
        PanResampleControl_
        PanResampleCanvas_
        PanResampleInfo_
        
        DropdownResample_Image_
        BtnResample_LoadCustom_
        LblResample_OriginalShape_
        DropdownResample_FilterA_
        DropdownResample_FilterB_
        SliderResample_Scale_
        LblResample_Scale_
        ChkResample_FreqAnalysis_
        BtnResample_Blink_
        BtnResample_Animate_
        BtnResample_Reset_
        
        % Tab 6 bypass bypass filter / theory button
        BtnResample_Theory_
        
        AxResampleRef_
        AxResampleA_
        AxResampleB_
        ImgResampleRef_
        ImgResampleA_
        ImgResampleB_
        
        StateResample_ImageName_ = 'Synthetic Chirp'
        StateResample_Image_
        StateResample_FilterA_ = 'Nearest'
        StateResample_FilterB_ = 'Lanczos'
        StateResample_Scale_ = 1.0
        StateResample_BlinkState_ = false
        StateResample_Animate_ = false
        ResampleTimer_
        ResampleAnimDirection_ = 1
        
        % ---- Tab 7: Image Blending (Szeliski Ex 3.18) ----
        TabBlend_
        PanBlendControl_
        PanBlendCanvas_
        PanBlendInfo_
        
        DropdownBlend_ImageA_
        DropdownBlend_ImageB_
        BtnBlend_LoadA_
        BtnBlend_LoadB_
        BtnBlend_LoadMask_
        DropdownBlend_MaskType_
        SliderBlend_Levels_
        LblBlend_Levels_
        DropdownBlend_Kernel_
        SliderBlend_KernelSize_
        LblBlend_KernelSize_
        ChkBlend_Bypass_
        BtnBlend_Reset_
        BtnBlend_Theory_
        
        AxBlendA_
        AxBlendB_
        AxBlendMask_
        AxBlendLinear_
        AxBlendPyramid_
        
        StateBlend_ImageA_
        StateBlend_ImageB_
        StateBlend_Mask_
        StateBlend_ImageNameA_ = 'ocean_img.jpg'
        StateBlend_ImageNameB_ = 'elon_musk.jpg'
        StateBlend_Levels_ = 6
        StateBlend_Kernel_ = 'Gaussian'
        StateBlend_KernelSize_ = 5
        StateBlend_Bypass_ = false
        LblBlend_Info_Stats_
        
        % ---- Tab 8: Local Warping (Szeliski Ex 3.23) ----
        TabLocalWarp_
        PanLocalWarpControl_
        PanLocalWarpCanvas_
        PanLocalWarpInfo_
        
        DropdownLocalWarp_Image_
        BtnLocalWarp_LoadCustom_
        DropdownLocalWarp_Mode_
        SliderLocalWarp_Grid_
        LblLocalWarp_Grid_
        ChkLocalWarp_ShowGrid_
        BtnLocalWarp_Reset_
        BtnLocalWarp_Theory_
        
        % Paintbrush controls (Part 4)
        SliderLocalWarp_BrushSize_
        LblLocalWarp_BrushSize_
        SliderLocalWarp_BrushStrength_
        LblLocalWarp_BrushStrength_
        BtnLocalWarp_ResetFlow_
        
        AxLocalWarp_
        ImgLocalWarp_
        PlotLocalWarp_ImgHandle_
        PlotLocalWarp_GridLines_ = []
        PlotLocalWarp_Handles_ = []
        
        StateLocalWarp_ImageName_ = 'Synthetic Grid'
        StateLocalWarp_Image_
        StateLocalWarp_Mode_ = 'Piecewise Affine'
        StateLocalWarp_GridRes_ = 4
        StateLocalWarp_ShowGrid_ = true
        StateLocalWarp_PtsSrc_
        StateLocalWarp_PtsTgt_
        StateLocalWarp_ActivePointIdx_ = -1
        LblLocalWarp_Info_Desc_
        LblLocalWarp_Info_Stats_
        
        % Beier-Neely Line morphing state (Part 2)
        StateLocalWarp_LinesSrc_P_
        StateLocalWarp_LinesSrc_Q_
        StateLocalWarp_LinesTgt_P_
        StateLocalWarp_LinesTgt_Q_
        StateLocalWarp_ActiveLineEnd_ = []
        
        % Paintbrush displacement flow buffers (Part 4)
        StateLocalWarp_Dx_
        StateLocalWarp_Dy_
        StateLocalWarp_BrushSize_ = 30
        StateLocalWarp_BrushStrength_ = 0.5
        StateLocalWarp_PrevMousePos_
        
        % ---- Tab 9: Feature Morphing (Szeliski Ex 3.25) ----
        TabMorph_
        PanMorphControl_
        PanMorphCanvas_
        PanMorphInfo_
        DropdownMorph_ImageA_
        DropdownMorph_ImageB_
        BtnMorph_LoadA_
        BtnMorph_LoadB_
        DropdownMorph_Mode_
        SliderMorph_GridRes_
        LblMorph_GridRes_
        ChkMorph_ShowGrid_
        SliderMorph_t_
        LblMorph_t_
        BtnMorph_AddPair_
        BtnMorph_DeletePair_
        BtnMorph_ClearPairs_
        LblMorph_PairCount_
        BtnMorph_Play_
        BtnMorph_SaveGif_
        SliderMorph_RotAngle_
        LblMorph_RotAngle_
        BtnMorph_TestRotation_
        BtnMorph_ResetPts_
        BtnMorph_Theory_
        AxMorph_A_
        AxMorph_B_
        AxMorph_Result_
        PlotMorph_ImgA_
        PlotMorph_ImgB_
        PlotMorph_Result_
        PlotMorph_HandlesA_
        PlotMorph_HandlesB_
        PlotMorph_LabelsA_  = {}
        PlotMorph_LabelsB_  = {}
        PlotMorph_GridA_    = []
        PlotMorph_GridB_    = []
        StateMorph_ImageA_
        StateMorph_ImageB_
        StateMorph_ImageNameA_ = 'ocean_img.jpg'
        StateMorph_ImageNameB_ = 'elon_musk.jpg'
        StateMorph_PtsA_
        StateMorph_PtsB_
        StateMorph_Mode_     = 'Piecewise Affine'
        StateMorph_GridRes_  = 4
        StateMorph_ShowGrid_ = true
        StateMorph_t_        = 0.5
        StateMorph_ActivePtIdx_  = -1
        StateMorph_ActiveSide_   = ''
        StateMorph_AddPairMode_  = false
        StateMorph_PendingPtA_   = []
        StateMorph_RotAngle_     = 45
        MorphTimer_
        MorphTimerT_ = 0
        LblMorph_Info_Desc_
        LblMorph_Info_Stats_
        
        % ---- Tab 10: 2D Collage Editor (Szeliski Ex 3.26) ----
        TabCollage_
        PanCollageControl_
        PanCollageCanvas_
        PanCollageInfo_
        BtnCollage_Import_
        DropdownCollage_Mode_
        ChkCollage_ShowGrid_
        BtnCollage_BringFront_
        BtnCollage_SendBack_
        BtnCollage_Delete_
        BtnCollage_ClearAll_
        BtnCollage_Theory_
        AxCollage_
        PlotCollage_ImgHandle_
        PlotCollage_Wireframe_ = []
        PlotCollage_Handles_ = []
        PlotCollage_Texts_ = {}
        StateCollage_Layers_ = {}
        StateCollage_SelectedIdx_ = -1
        StateCollage_ActivePtIdx_ = -1
        StateCollage_Mode_ = 'Perspective'
        StateCollage_ShowGrid_ = true
        StateCollage_BgCache_
        StateCollage_CanvasSz_ = [600, 800]
        LblCollage_Info_Desc_
        LblCollage_Info_Stats_
        
        % Visualizer instances
        VizCore_
    end
    
    properties (Access = private, Constant)
        % Sleek Dark Mode colors matching Themes.json
        COLOR_BG        = [0.11, 0.11, 0.13]  % gunmetal background
        COLOR_PANEL     = [0.17, 0.19, 0.23]  % sidebar panel
        COLOR_BTN       = [0.22, 0.25, 0.32]  % button background
        COLOR_TEXT      = [0.90, 0.90, 0.90]  % primary text
        COLOR_SUBTEXT   = [0.55, 0.58, 0.63]  % secondary text
        COLOR_ACCENT    = [0.20, 0.60, 1.00]  % ocean blue highlight
        COLOR_SECONDARY = [1.00, 0.40, 0.40]  % secondary highlight
        COLOR_MUTED     = [0.35, 0.37, 0.42]
        
        FIG_WIDTH       = 1200
        FIG_HEIGHT      = 750
        SIDEBAR_W       = 320
        INFO_W          = 260
    end
    
    methods
        function obj = ComputerVisionSandbox()
            % COMPUTERVISIONSANDBOX  Construct app and instantiate layout.
            obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
            
            % Generate reference geometry for all tabs
            obj.build2DGeometry_();
            obj.build3DGeometry_();
            obj.buildCameraGeometry_();
            obj.buildDistortionGeometry_();
            
            % Draw the GUI
            obj.buildUI_();
            
            % Update all tabs initial visualizations
            obj.update2DTransform_();
            obj.update3DTransform_();
            obj.updateCameraProjection_();
            obj.updateLensDistortion_();
            obj.updateTabPyr_();
            obj.updateTabResample_();
            obj.updateTabBlend_();
            obj.buildLocalWarpGeometry_();
            obj.updateTabLocalWarp_();
            obj.buildMorphGeometry_();
            obj.updateTabMorph_();
            obj.buildCollageGeometry_();
            obj.updateTabCollage_();
        end
        
        function show(obj)
            % SHOW  Make the application window visible.
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                figure(obj.Fig_);
            end
        end
        
        function delete(obj)
            % DELETE  Teardown window handles safely.
            if ~isempty(obj.MorphTimer_) && isvalid(obj.MorphTimer_)
                stop(obj.MorphTimer_);
                delete(obj.MorphTimer_);
            end
            if ~isempty(obj.SlerpTimer_) && isvalid(obj.SlerpTimer_)
                stop(obj.SlerpTimer_);
                delete(obj.SlerpTimer_);
            end
            if ~isempty(obj.ResampleTimer_) && isvalid(obj.ResampleTimer_)
                stop(obj.ResampleTimer_);
                delete(obj.ResampleTimer_);
            end
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
    end
    
    methods (Access = private)
        
        % ================================================================== %
        % Reference Geometry Construction
        % ================================================================== %
        function build2DGeometry_(obj)
            % 1. Create a 2D line grid spanning -4 to 4
            xVals = -4:0.5:4;
            yVals = -4:0.5:4;
            pts = [];
            
            % Vertical lines
            for x = xVals
                pts = [pts, [x; -4], [x; 4], [NaN; NaN]];
            end
            % Horizontal lines
            for y = yVals
                pts = [pts, [-4; y], [4; y], [NaN; NaN]];
            end
            obj.RefGrid2DPoints_ = pts;
            
            % 2. Create asymmetric letter 'F' (closed polygon points)
            % Coordinates shifted slightly to center the shape
            obj.RefShape2DPoints_ = [
                -0.5, -0.5,  1.5,  1.5,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -0.5; ...
                -1.5,  1.5,  1.5,  1.0,  1.0,  0.3,  0.3, -0.2, -0.2, -1.5, -1.5 ...
            ];
        end
        
        function build3DGeometry_(obj)
            % Create standard 3D unit cube vertices & faces
            obj.RefCubeVertices_ = [
                -1 -1 -1; ... % v1
                 1 -1 -1; ... % v2
                 1  1 -1; ... % v3
                -1  1 -1; ... % v4
                -1 -1  1; ... % v5
                 1 -1  1; ... % v6
                 1  1  1; ... % v7
                -1  1  1  ... % v8
            ]';
            obj.RefCubeFaces_ = [
                1 2 3 4; ... % bottom
                2 6 7 3; ... % right
                4 3 7 8; ... % top
                1 5 8 4; ... % left
                1 2 6 5; ... % front
                5 6 7 8  ... % back
            ];
        end
        
        function buildCameraGeometry_(obj)
            % Create 3D wireframe house shape centered at origin
            obj.RefHouseVertices_ = [
                -1.5, -1.5, -1.0; ... % 1: floor corners
                 1.5, -1.5, -1.0; ... % 2
                 1.5,  1.5, -1.0; ... % 3
                -1.5,  1.5, -1.0; ... % 4
                -1.5, -1.5,  0.5; ... % 5: wall tops
                 1.5, -1.5,  0.5; ... % 6
                 1.5,  1.5,  0.5; ... % 7
                -1.5,  1.5,  0.5; ... % 8
                 0.0, -1.5,  1.5; ... % 9: roof ridge front
                 0.0,  1.5,  1.5  ... % 10: roof ridge back
            ]';
            
            % Line segment connectivity indices (1-based index pairs)
            obj.RefHouseLines_ = [
                1, 2; 2, 3; 3, 4; 4, 1; ... % floor loop
                5, 6; 6, 7; 7, 8; 8, 5; ... % walls top loop
                1, 5; 2, 6; 3, 7; 4, 8; ... % pillars
                5, 9; 6, 9; 7, 10; 8, 10; 9, 10 ... % roof ridge
            ];
        end
        
        function buildDistortionGeometry_(obj)
            % Generate a structured 2D grid of points in normalized coordinates [-0.8, 0.8]
            xVals = -0.8:0.2:0.8;
            yVals = -0.8:0.2:0.8;
            
            % Create 2D coordinate mesh
            [X, Y] = meshgrid(xVals, yVals);
            obj.RefDistPoints_ = [X(:)'; Y(:)'];
            
            % Build horizontal and vertical grid lines connectivity for distortion overlay
            linesPts = [];
            
            % Vertical grid lines
            for x = xVals
                linesPts = [linesPts, [x; -0.8], [x; 0.8], [NaN; NaN]];
            end
            % Horizontal grid lines
            for y = yVals
                linesPts = [linesPts, [-0.8; y], [0.8; y], [NaN; NaN]];
            end
            obj.RefDistLineIndices_ = linesPts;
        end
        
        % ================================================================== %
        % UI Layout Construction
        % ================================================================== %
        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0  = (scr(3) - obj.FIG_WIDTH)  / 2;
            y0  = (scr(4) - obj.FIG_HEIGHT) / 2;
            
            obj.Fig_ = uifigure( ...
                'Name',            'Computer Vision Primitives Sandbox — Ocean Library', ...
                'Position',        [x0, y0, obj.FIG_WIDTH, obj.FIG_HEIGHT], ...
                'Color',           obj.COLOR_BG, ...
                'Resize',          'off', ...
                'CloseRequestFcn', @(~,~) obj.onClose_());
            
            obj.TabGroup_ = uitabgroup(obj.Fig_, ...
                'Position', [0, 0, obj.FIG_WIDTH, obj.FIG_HEIGHT]);
            
            obj.Tab2D_ = uitab(obj.TabGroup_, ...
                'Title', '1. 2D Transformations', ...
                'BackgroundColor', obj.COLOR_BG);
            
            obj.Tab3D_ = uitab(obj.TabGroup_, ...
                'Title', '2. 3D Rotations & Slerp', ...
                'BackgroundColor', obj.COLOR_BG);
                
            obj.TabCam_ = uitab(obj.TabGroup_, ...
                'Title', '3. Camera Projection', ...
                'BackgroundColor', obj.COLOR_BG);
                
            obj.TabDist_ = uitab(obj.TabGroup_, ...
                'Title', '4. Lens Distortion', ...
                'BackgroundColor', obj.COLOR_BG);
                
            obj.TabPyr_ = uitab(obj.TabGroup_, ...
                'Title', '5. Image Pyramids', ...
                'BackgroundColor', obj.COLOR_BG);
                
            obj.TabResample_ = uitab(obj.TabGroup_, ...
                'Title', '6. Image Resampling', ...
                'BackgroundColor', obj.COLOR_BG);
            
            obj.TabBlend_ = uitab(obj.TabGroup_, ...
                'Title', '7. Image Blending', ...
                'BackgroundColor', obj.COLOR_BG);
                
            obj.TabLocalWarp_ = uitab(obj.TabGroup_, ...
                'Title', '8. Local Warping', ...
                'BackgroundColor', obj.COLOR_BG);
            
            obj.TabMorph_ = uitab(obj.TabGroup_, ...
                'Title', '9. Feature Morphing', ...
                'BackgroundColor', obj.COLOR_BG);
            obj.TabCollage_ = uitab(obj.TabGroup_, ...
                'Title', '10. 2D Collage Editor', ...
                'BackgroundColor', obj.COLOR_BG);
            
            % Build layouts for each tab
            obj.buildTab2D_();
            obj.buildTab3D_();
            obj.buildTabCam_();
            obj.buildTabDist_();
            obj.buildTabPyr_();
            obj.buildTabResample_();
            obj.buildTabBlend_();
            obj.buildTabMorph_();
            obj.buildTabLocalWarp_();
            obj.buildTabCollage_();
        end
        
        % ================================================================== %
        % Tab 1: 2D Transformations Tab Builder
        % ================================================================== %
        function buildTab2D_(obj)
            % Left control sidebar panel
            obj.Pan2DControl_ = uipanel(obj.Tab2D_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Center canvas view panel
            obj.Pan2DCanvas_ = uipanel(obj.Tab2D_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Right explanation/math info panel
            obj.Pan2DInfo_ = uipanel(obj.Tab2D_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % ---------------- Left Controls ----------------
            yPos = 655;
            
            % Title
            uilabel(obj.Pan2DControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Interactive warps', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Sliders: translation tx
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Translation X (tx):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Tx_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Tx_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-4, 4], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Sliders: translation ty
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Translation Y (ty):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Ty_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Ty_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-4, 4], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Sliders: rotation theta
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 150, 20], 'Text', 'Rotation Angle (deg):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Rot_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Rot_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-180, 180], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Sliders: scaling sx
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Scaling X (sx):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Sx_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '1.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Sx_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [0.1, 3.0], 'Value', 1.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Sliders: scaling sy
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Scaling Y (sy):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Sy_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '1.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Sy_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [0.1, 3.0], 'Value', 1.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Sliders: shear shx
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Shear X (shx):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Shx_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Shx_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-2, 2], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Sliders: shear shy
            yPos = yPos - 45;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Shear Y (shy):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl2D_Shy_ = uilabel(obj.Pan2DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider2D_Shy_ = uislider(obj.Pan2DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-2, 2], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on2DSliderChanged_());
                
            % Section Divider
            yPos = yPos - 20;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Homography Matrix Input grid
            yPos = yPos - 30;
            uilabel(obj.Pan2DControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Homography Matrix (H):', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 12, 'FontWeight', 'bold');
            
            obj.Grid2D_Matrix_ = cell(3,3);
            mWidth = 65;
            mHeight = 22;
            xStart = 35;
            for r = 1:3
                yPos = yPos - 28;
                for c = 1:3
                    xPos = xStart + (c-1)*(mWidth+8);
                    initVal = 0.0;
                    if r == c, initVal = 1.0; end
                    obj.Grid2D_Matrix_{r,c} = uieditfield(obj.Pan2DControl_, 'numeric', ...
                        'Position', [xPos, yPos, mWidth, mHeight], ...
                        'Value', initVal, ...
                        'BackgroundColor', obj.COLOR_BG, ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'FontSize', 10, ...
                        'ValueChangedFcn', @(~,~) obj.on2DMatrixChanged_());
                end
            end
            
            % Matrix controls: Apply, AutoApply, Reset
            yPos = yPos - 35;
            obj.Chk2D_AutoApply_ = uicheckbox(obj.Pan2DControl_, ...
                'Position', [20, yPos, 140, 20], ...
                'Text', 'Auto-apply sliders', ...
                'Value', true, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10);
                
            obj.Btn2D_ApplyMatrix_ = uibutton(obj.Pan2DControl_, ...
                'Position', [170, yPos, 100, 22], ...
                'Text', 'Apply Matrix', ...
                'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.on2DApplyMatrixPressed_());
                
            yPos = yPos - 35;
            obj.Btn2D_Reset_ = uibutton(obj.Pan2DControl_, ...
                'Position', [20, yPos, 250, 26], ...
                'Text', 'Reset Homography (Identity)', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.on2DResetPressed_());
                
            % ---------------- Center Canvas ----------------
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.Ax2D_ = uiaxes(obj.Pan2DCanvas_, ...
                'Position', [0, 10, cW, 670], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.Ax2D_, '2D Projection coordinate grid', 'Color', obj.COLOR_TEXT);
            
            % Style via OceanViz Core
            obj.VizCore_.styleAxes(obj.Ax2D_);
            hold(obj.Ax2D_, 'on');
            
            % Draw original reference grid (dashed gray)
            obj.Grid2D_Ref_ = plot(obj.Ax2D_, obj.RefGrid2DPoints_(1, :), obj.RefGrid2DPoints_(2, :), ...
                '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.8);
                
            % Draw warped grid
            obj.Grid2D_Warped_ = plot(obj.Ax2D_, obj.RefGrid2DPoints_(1, :), obj.RefGrid2DPoints_(2, :), ...
                '-', 'Color', obj.COLOR_MUTED, 'LineWidth', 1.0);
                
            % Draw original shape silhouette
            obj.Shape2D_Ref_ = plot(obj.Ax2D_, obj.RefShape2DPoints_(1, :), obj.RefShape2DPoints_(2, :), ...
                '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2);
                
            % Draw warped shape
            obj.Shape2D_Warped_ = fill(obj.Ax2D_, obj.RefShape2DPoints_(1, :), obj.RefShape2DPoints_(2, :), ...
                obj.COLOR_ACCENT, 'FaceAlpha', 0.4, 'EdgeColor', obj.COLOR_ACCENT, 'LineWidth', 2.0);
                
            axis(obj.Ax2D_, 'equal');
            xlim(obj.Ax2D_, [-6, 6]);
            ylim(obj.Ax2D_, [-6, 6]);
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            uilabel(obj.Pan2DInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Projection Insights', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            uilabel(obj.Pan2DInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 90;
            uilabel(obj.Pan2DInfo_, ...
                'Position', [15, iy, 230, 80], ...
                'Text', sprintf('Coordinate Mapping:\nx'' = H * x\n\n[x'']   [h11  h12  h13]   [x]\n[y''] = [h21  h22  h23] * [y]\n[w'']   [h31  h32  h33]   [1]'), ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 11, ...
                'FontName', 'Courier');
                
            iy = iy - 110;
            uilabel(obj.Pan2DInfo_, ...
                'Position', [15, iy, 230, 100], ...
                'Text', sprintf('Linear Block (Rot/Scale/Shear):\n[h11  h12]\n[h21  h22]\n\nTranslation Vector:\n[h13]\n[h23]'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 10);
                
            iy = iy - 100;
            uilabel(obj.Pan2DInfo_, ...
                'Position', [15, iy, 230, 90], ...
                'Text', sprintf('Projective shear terms:\n[h31  h32]\n\nGlobal scaling factor:\nh33 (Usually normalized to 1)'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 10);
        end
        
        % ================================================================== %
        % Tab 2: 3D Rotations & Slerp Tab Builder
        % ================================================================== %
        function buildTab3D_(obj)
            % Sidebar panel
            obj.Pan3DControl_ = uipanel(obj.Tab3D_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Canvas panel
            obj.Pan3DCanvas_ = uipanel(obj.Tab3D_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Right explanation/math info panel
            obj.Pan3DInfo_ = uipanel(obj.Tab3D_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % ---------------- Left Controls ----------------
            yPos = 655;
            
            % Title
            uilabel(obj.Pan3DControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', '3D rotation controls', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            uilabel(obj.Pan3DControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Format Selection Dropdown
            yPos = yPos - 35;
            uilabel(obj.Pan3DControl_, 'Position', [15, yPos+2, 110, 20], 'Text', 'Rotation Mode:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Dropdown3D_Format_ = uidropdown(obj.Pan3DControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Euler Angles', 'Quaternions', 'Axis-Angle', 'Exponential Map'}, ...
                'Value', 'Euler Angles', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.on3DFormatChanged_());
                
            % Sub-Panels for each mode (occupying same space coordinates)
            panelY = yPos - 170;
            panelH = 160;
            
            % ---- Panel A: Euler ----
            obj.Pan3D_Euler_ = uipanel(obj.Pan3DControl_, ...
                'Position', [10, panelY, 290, panelH], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Visible', 'on');
                
            subY = 120;
            uilabel(obj.Pan3D_Euler_, 'Position', [5, subY+18, 90, 20], 'Text', 'Roll (x-axis):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_Roll_ = uilabel(obj.Pan3D_Euler_, 'Position', [190, subY+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_Roll_ = uislider(obj.Pan3D_Euler_, 'Position', [15, subY+10, 240, 3], 'Limits', [-180, 180], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DEulerSliderChanged_());
                
            subY = subY - 45;
            uilabel(obj.Pan3D_Euler_, 'Position', [5, subY+18, 90, 20], 'Text', 'Pitch (y-axis):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_Pitch_ = uilabel(obj.Pan3D_Euler_, 'Position', [190, subY+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_Pitch_ = uislider(obj.Pan3D_Euler_, 'Position', [15, subY+10, 240, 3], 'Limits', [-89, 89], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DEulerSliderChanged_());
                
            subY = subY - 45;
            uilabel(obj.Pan3D_Euler_, 'Position', [5, subY+18, 90, 20], 'Text', 'Yaw (z-axis):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_Yaw_ = uilabel(obj.Pan3D_Euler_, 'Position', [190, subY+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_Yaw_ = uislider(obj.Pan3D_Euler_, 'Position', [15, subY+10, 240, 3], 'Limits', [-180, 180], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DEulerSliderChanged_());
                
            % ---- Panel B: Quaternion ----
            obj.Pan3D_Quat_ = uipanel(obj.Pan3DControl_, ...
                'Position', [10, panelY, 290, panelH], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Visible', 'off');
                
            uilabel(obj.Pan3D_Quat_, 'Position', [10, 100, 270, 40], ...
                'Text', 'Inputs represent unit quaternion elements (w, x, y, z). Value is auto-normalized when any field changes.', ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 9, 'WordWrap', 'on');
                
            subY = 60;
            fW = 50; fH = 22;
            uilabel(obj.Pan3D_Quat_, 'Position', [15, subY+24, 20, 20], 'Text', 'Qw', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Field3D_Qw_ = uieditfield(obj.Pan3D_Quat_, 'numeric', 'Position', [15, subY, fW, fH], 'Value', 1.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DQuatFieldChanged_());
                
            uilabel(obj.Pan3D_Quat_, 'Position', [80, subY+24, 20, 20], 'Text', 'Qx', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Field3D_Qx_ = uieditfield(obj.Pan3D_Quat_, 'numeric', 'Position', [80, subY, fW, fH], 'Value', 0.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DQuatFieldChanged_());
                
            uilabel(obj.Pan3D_Quat_, 'Position', [145, subY+24, 20, 20], 'Text', 'Qy', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Field3D_Qy_ = uieditfield(obj.Pan3D_Quat_, 'numeric', 'Position', [145, subY, fW, fH], 'Value', 0.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DQuatFieldChanged_());
                
            uilabel(obj.Pan3D_Quat_, 'Position', [210, subY+24, 20, 20], 'Text', 'Qz', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Field3D_Qz_ = uieditfield(obj.Pan3D_Quat_, 'numeric', 'Position', [210, subY, fW, fH], 'Value', 0.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DQuatFieldChanged_());
                
            % ---- Panel C: Axis-Angle ----
            obj.Pan3D_AxisAngle_ = uipanel(obj.Pan3DControl_, ...
                'Position', [10, panelY, 290, panelH], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Visible', 'off');
                
            uilabel(obj.Pan3D_AxisAngle_, 'Position', [5, 120, 90, 20], 'Text', 'Unit Axis [u]:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Field3D_Ax_ = uieditfield(obj.Pan3D_AxisAngle_, 'numeric', 'Position', [95, 120, 50, 20], 'Value', 0.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DAxisAngleChanged_());
            obj.Field3D_Ay_ = uieditfield(obj.Pan3D_AxisAngle_, 'numeric', 'Position', [155, 120, 50, 20], 'Value', 0.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DAxisAngleChanged_());
            obj.Field3D_Az_ = uieditfield(obj.Pan3D_AxisAngle_, 'numeric', 'Position', [215, 120, 50, 20], 'Value', 1.0, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, 'ValueChangedFcn', @(~,~) obj.on3DAxisAngleChanged_());
                
            uilabel(obj.Pan3D_AxisAngle_, 'Position', [5, 65, 120, 20], 'Text', 'Rotation theta (deg):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_Ang_ = uilabel(obj.Pan3D_AxisAngle_, 'Position', [190, 65, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_Ang_ = uislider(obj.Pan3D_AxisAngle_, 'Position', [15, 55, 240, 3], 'Limits', [-180, 180], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DAxisAngleChanged_());
                
            % ---- Panel D: Exponential Map ----
            obj.Pan3D_ExpMap_ = uipanel(obj.Pan3DControl_, ...
                'Position', [10, panelY, 290, panelH], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Visible', 'off');
                
            subY = 120;
            uilabel(obj.Pan3D_ExpMap_, 'Position', [5, subY+18, 90, 20], 'Text', 'Element v1 (rx):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_V1_ = uilabel(obj.Pan3D_ExpMap_, 'Position', [190, subY+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_V1_ = uislider(obj.Pan3D_ExpMap_, 'Position', [15, subY+10, 240, 3], 'Limits', [-pi, pi], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DExpMapChanged_());
                
            subY = subY - 45;
            uilabel(obj.Pan3D_ExpMap_, 'Position', [5, subY+18, 90, 20], 'Text', 'Element v2 (ry):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_V2_ = uilabel(obj.Pan3D_ExpMap_, 'Position', [190, subY+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_V2_ = uislider(obj.Pan3D_ExpMap_, 'Position', [15, subY+10, 240, 3], 'Limits', [-pi, pi], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DExpMapChanged_());
                
            subY = subY - 45;
            uilabel(obj.Pan3D_ExpMap_, 'Position', [5, subY+18, 90, 20], 'Text', 'Element v3 (rz):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.Lbl3D_V3_ = uilabel(obj.Pan3D_ExpMap_, 'Position', [190, subY+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_V3_ = uislider(obj.Pan3D_ExpMap_, 'Position', [15, subY+10, 240, 3], 'Limits', [-pi, pi], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.on3DExpMapChanged_());
                
            % Section Divider
            yPos = panelY - 15;
            uilabel(obj.Pan3DControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Slerp panel Title
            yPos = yPos - 30;
            uilabel(obj.Pan3DControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Spherical Linear Interpolation (SLERP):', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 12, 'FontWeight', 'bold');
            
            % Start & End Capture buttons
            yPos = yPos - 30;
            obj.Btn3D_SetStart_ = uibutton(obj.Pan3DControl_, ...
                'Position', [15, yPos, 130, 22], ...
                'Text', 'Capture Start Rot', ...
                'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onSetStartRotation_());
                
            obj.Btn3D_SetEnd_ = uibutton(obj.Pan3DControl_, ...
                'Position', [155, yPos, 130, 22], ...
                'Text', 'Capture End Rot', ...
                'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onSetEndRotation_());
                
            % Slerp t factor slider
            yPos = yPos - 45;
            uilabel(obj.Pan3DControl_, 'Position', [15, yPos+18, 150, 20], 'Text', 'Interpolation Factor (t):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.Lbl3D_SlerpT_ = uilabel(obj.Pan3DControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.Slider3D_SlerpT_ = uislider(obj.Pan3DControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [0.0, 1.0], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onSlerpSliderChanged_());
                
            % Animate button
            yPos = yPos - 32;
            obj.Btn3D_AnimateSlerp_ = uibutton(obj.Pan3DControl_, ...
                'Position', [15, yPos, 270, 24], ...
                'Text', 'Animate Slerp Interpolation', ...
                'FontWeight', 'bold', ...
                'FontSize', 10, ...
                'BackgroundColor', [0.15, 0.50, 0.80], ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onAnimateSlerpPressed_());
                
            % Global Reset button
            yPos = yPos - 38;
            obj.Btn3D_Reset_ = uibutton(obj.Pan3DControl_, ...
                'Position', [15, yPos, 270, 26], ...
                'Text', 'Reset 3D Orientation (Identity)', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.on3DResetPressed_());
                
            % ---------------- Center Canvas ----------------
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.Ax3D_ = uiaxes(obj.Pan3DCanvas_, ...
                'Position', [0, 10, cW, 670], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.Ax3D_, 'Interactive 3D Rotation space', 'Color', obj.COLOR_TEXT);
            
            % Style via OceanViz Core
            obj.VizCore_.styleAxes(obj.Ax3D_);
            hold(obj.Ax3D_, 'on');
            
            % Draw World Reference Axes (thin dotted lines)
            plot3(obj.Ax3D_, [-2, 2], [0, 0], [0, 0], 'w:', 'LineWidth', 0.8);
            plot3(obj.Ax3D_, [0, 0], [-2, 2], [0, 0], 'w:', 'LineWidth', 0.8);
            plot3(obj.Ax3D_, [0, 0], [0, 0], [-2, 2], 'w:', 'LineWidth', 0.8);
            
            % Draw Translucent 3D patch Cube
            obj.Cube3D_Warped_ = patch(obj.Ax3D_, ...
                'Vertices', obj.RefCubeVertices_', ...
                'Faces', obj.RefCubeFaces_, ...
                'FaceColor', obj.COLOR_ACCENT, ...
                'FaceAlpha', 0.25, ...
                'EdgeColor', obj.COLOR_ACCENT, ...
                'LineWidth', 1.5);
                
            % Draw Local coordinate frame axes (Thick solid lines that rotate)
            obj.LocalAxisX_ = plot3(obj.Ax3D_, [0, 1.5], [0, 0], [0, 0], '-', 'Color', obj.COLOR_SECONDARY, 'LineWidth', 2.2);
            obj.LocalAxisY_ = plot3(obj.Ax3D_, [0, 0], [0, 1.5], [0, 0], '-', 'Color', [0.15, 0.65, 0.40], 'LineWidth', 2.2);
            obj.LocalAxisZ_ = plot3(obj.Ax3D_, [0, 0], [0, 0], [0, 1.5], '-', 'Color', obj.COLOR_ACCENT, 'LineWidth', 2.2);
            
            view(obj.Ax3D_, 3);
            xlim(obj.Ax3D_, [-3, 3]);
            ylim(obj.Ax3D_, [-3, 3]);
            zlim(obj.Ax3D_, [-3, 3]);
            grid(obj.Ax3D_, 'on');
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            uilabel(obj.Pan3DInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Rotation Matrix (R)', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            
            uilabel(obj.Pan3DInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % 3x3 Readout Grid labels
            obj.Grid3D_Matrix_ = cell(3,3);
            iy = iy - 15;
            for r = 1:3
                iy = iy - 26;
                for c = 1:3
                    xPos = 15 + (c-1)*76;
                    obj.Grid3D_Matrix_{r,c} = uilabel(obj.Pan3DInfo_, ...
                        'Position', [xPos, iy, 70, 20], ...
                        'Text', '0.000', ...
                        'FontColor', obj.COLOR_TEXT, ...
                        'FontSize', 10, ...
                        'FontName', 'Courier', ...
                        'HorizontalAlignment', 'center');
                end
            end
            
            % Unified orientation feedback
            iy = iy - 35;
            uilabel(obj.Pan3DInfo_, 'Position', [15, iy, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 30;
            uilabel(obj.Pan3DInfo_, 'Position', [15, iy, 230, 20], 'Text', 'Quaternion representation:', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 11, 'FontWeight', 'bold');
            iy = iy - 22;
            obj.Lbl3D_Quat_.Text = 'qw, qx, qy, qz';
            obj.Lbl3D_Quat_ = uilabel(obj.Pan3DInfo_, 'Position', [15, iy, 230, 20], 'Text', '[1.000, 0.000, 0.000, 0.000]', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10, 'FontName', 'Courier');
            
            iy = iy - 30;
            uilabel(obj.Pan3DInfo_, 'Position', [15, iy, 230, 20], 'Text', 'Exponential Map so(3):', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 11, 'FontWeight', 'bold');
            iy = iy - 22;
            obj.Lbl3D_Exp_ = uilabel(obj.Pan3DInfo_, 'Position', [15, iy, 230, 20], 'Text', '[0.000, 0.000, 0.000]', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10, 'FontName', 'Courier');
            
            % Interactive Help block
            iy = iy - 10;
            uilabel(obj.Pan3DInfo_, 'Position', [15, iy, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 165;
            uilabel(obj.Pan3DInfo_, ...
                'Position', [15, iy, 230, 155], ...
                'Text', sprintf('Mathematical insights:\n\nEuler angles suffer from Gimbal Lock poles (singularity at pitch=90 deg).\n\nQuaternions avoid singularities and enable spherical linear interpolation (SLERP) representing the shortest path geodesic on a 4D sphere.\n\nExponential Map represents Lie Algebra so(3) mapped to Lie Group SO(3).'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
        end
        
        % ================================================================== %
        % Tab 3: Camera Projection Tab Builder
        % ================================================================== %
        function buildTabCam_(obj)
            % Sidebar panel (extrinsics/intrinsics controls)
            obj.PanCamControl_ = uipanel(obj.TabCam_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Canvas panel (Left 3D World vs Right 2D Sensor projection)
            obj.PanCamCanvas_ = uipanel(obj.TabCam_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Right explanation panel
            obj.PanCamInfo_ = uipanel(obj.TabCam_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % ---------------- Left Controls ----------------
            yPos = 655;
            uilabel(obj.PanCamControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Camera settings', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanCamControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Model selection
            yPos = yPos - 35;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+2, 100, 20], 'Text', 'Proj. Model:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownCam_Type_ = uidropdown(obj.PanCamControl_, ...
                'Position', [120, yPos, 170, 22], ...
                'Items', {'Perspective', 'Orthographic', 'WeakPerspective'}, ...
                'Value', 'Perspective', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % --- Section: Intrinsics ---
            yPos = yPos - 30;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Intrinsics (K):', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 12, 'FontWeight', 'bold');
            
            % fx
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Focal Length fx:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Fx_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '500', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Fx_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [100, 1000], 'Value', 500, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % fy
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Focal Length fy:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Fy_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '500', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Fy_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [100, 1000], 'Value', 500, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % cx
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Center cx:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Cx_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '320', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Cx_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [100, 500], 'Value', 320, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % cy
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Center cy:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Cy_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '240', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Cy_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [100, 400], 'Value', 240, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Skew s
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Axis Skew (s):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Skew_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Skew_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-50, 50], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % --- Section: Extrinsics ---
            yPos = yPos - 20;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            yPos = yPos - 25;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos, 280, 20], 'Text', 'Extrinsics (Pose in world):', 'FontColor', obj.COLOR_ACCENT, 'FontSize', 12, 'FontWeight', 'bold');
            
            % Cam X
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Camera World X:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_X_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_X_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-5, 5], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Cam Y
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Camera World Y:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Y_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Y_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-5, 5], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Cam Z
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Camera World Z:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Z_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '-8.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Z_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-15, -2], 'Value', -8.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Cam Roll (x)
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Camera Roll (deg):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Roll_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Roll_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-180, 180], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Cam Pitch (y)
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Camera Pitch (deg):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Pitch_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Pitch_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-90, 90], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Cam Yaw (z)
            yPos = yPos - 42;
            uilabel(obj.PanCamControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Camera Yaw (deg):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 10);
            obj.LblCam_Yaw_ = uilabel(obj.PanCamControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderCam_Yaw_ = uislider(obj.PanCamControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-180, 180], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onCameraSliderChanged_());
                
            % Reset Button
            yPos = yPos - 35;
            obj.BtnCam_Reset_ = uibutton(obj.PanCamControl_, ...
                'Position', [20, yPos, 250, 26], ...
                'Text', 'Reset Camera Settings', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onCameraResetPressed_());
                
            % ---------------- Center Canvas (Side by Side) ----------------
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            wHalf = floor(cW / 2) - 10;
            
            % Left axes: 3D world view
            obj.AxCam3D_ = uiaxes(obj.PanCamCanvas_, ...
                'Position', [0, 10, wHalf, 670], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.AxCam3D_, '3D World coordinate space', 'Color', obj.COLOR_TEXT);
            obj.VizCore_.styleAxes(obj.AxCam3D_);
            hold(obj.AxCam3D_, 'on');
            
            % Plot 3D house wireframe
            obj.House3D_Line_ = cell(size(obj.RefHouseLines_, 1), 1);
            for i = 1:size(obj.RefHouseLines_, 1)
                idx = obj.RefHouseLines_(i, :);
                obj.House3D_Line_{i} = plot3(obj.AxCam3D_, ...
                    obj.RefHouseVertices_(1, idx), ...
                    obj.RefHouseVertices_(2, idx), ...
                    obj.RefHouseVertices_(3, idx), ...
                    '-', 'Color', obj.COLOR_TEXT, 'LineWidth', 1.5);
            end
            
            % Pre-build wireframe frustum lines in 3D world
            obj.Frustum3D_Line_ = cell(8, 1);
            for i = 1:8
                obj.Frustum3D_Line_{i} = plot3(obj.AxCam3D_, [0, 0], [0, 0], [0, 0], '-', 'Color', obj.COLOR_SECONDARY, 'LineWidth', 1.8);
            end
            obj.CamOrigin3D_Pt_ = plot3(obj.AxCam3D_, 0, 0, 0, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
            
            view(obj.AxCam3D_, 3);
            xlim(obj.AxCam3D_, [-6, 6]);
            ylim(obj.AxCam3D_, [-6, 6]);
            zlim(obj.AxCam3D_, [-3, 6]);
            grid(obj.AxCam3D_, 'on');
            
            % Right axes: 2D Projection sensor viewport view
            obj.AxCam2D_ = uiaxes(obj.PanCamCanvas_, ...
                'Position', [wHalf + 20, 10, wHalf, 670], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.AxCam2D_, 'Projected Camera Viewport (640x480)', 'Color', obj.COLOR_TEXT);
            obj.VizCore_.styleAxes(obj.AxCam2D_);
            hold(obj.AxCam2D_, 'on');
            
            % Projected 2D lines
            obj.House2D_Line_ = cell(size(obj.RefHouseLines_, 1), 1);
            for i = 1:size(obj.RefHouseLines_, 1)
                obj.House2D_Line_{i} = plot(obj.AxCam2D_, [0, 0], [0, 0], '-', 'Color', obj.COLOR_ACCENT, 'LineWidth', 2.0);
            end
            
            % Fix viewport limits to fit 640x480 camera sensor coordinate boundary
            xlim(obj.AxCam2D_, [0, 640]);
            ylim(obj.AxCam2D_, [0, 480]);
            obj.AxCam2D_.YDir = 'reverse'; % Pinhole camera has standard reversed Y axis in image-space coords
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            uilabel(obj.PanCamInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Mathematical Duality', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanCamInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 110;
            uilabel(obj.PanCamInfo_, ...
                'Position', [15, iy, 230, 100], ...
                'Text', sprintf('Intrinsic Matrix (K):\n[fx  s   cx]\n[0   fy  cy]\n[0   0   1 ]\n\nProjects camera 3D space to image 2D coordinate space.'), ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10);
                
            iy = iy - 120;
            uilabel(obj.PanCamInfo_, ...
                'Position', [15, iy, 230, 110], ...
                'Text', sprintf('Extrinsics [R | t]:\nP_camera = R * P_world + t\n\nT_w2c homogeneous matrix represents camera coordinate transformation in the world space, enabling coordinate shifts from world origin to local camera sensor origin.'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
                
            iy = iy - 120;
            uilabel(obj.PanCamInfo_, ...
                'Position', [15, iy, 230, 110], ...
                'Text', sprintf('Weak-Perspective:\nApproximates perspective projection by using constant depth scaling (1/Z_average) for all scene points. Perfect for orthographic approximations when object depth variation is small relative to total camera distance.'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
        end
        
        % ================================================================== %
        % Tab 4: Lens Distortions Tab Builder
        % ================================================================== %
        function buildTabDist_(obj)
            % Sidebar panel
            obj.PanDistControl_ = uipanel(obj.TabDist_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Canvas panel
            obj.PanDistCanvas_ = uipanel(obj.TabDist_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Right explanation panel
            obj.PanDistInfo_ = uipanel(obj.TabDist_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % ---------------- Left Controls ----------------
            yPos = 655;
            uilabel(obj.PanDistControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Distortion coefficients', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanDistControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Radial K1
            yPos = yPos - 45;
            uilabel(obj.PanDistControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Radial k1:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblDist_K1_ = uilabel(obj.PanDistControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderDist_K1_ = uislider(obj.PanDistControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-1.0, 1.0], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onDistortionSliderChanged_());
                
            % Radial K2
            yPos = yPos - 45;
            uilabel(obj.PanDistControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Radial k2:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblDist_K2_ = uilabel(obj.PanDistControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderDist_K2_ = uislider(obj.PanDistControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-0.5, 0.5], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onDistortionSliderChanged_());
                
            % Radial K3
            yPos = yPos - 45;
            uilabel(obj.PanDistControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Radial k3:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblDist_K3_ = uilabel(obj.PanDistControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderDist_K3_ = uislider(obj.PanDistControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-0.2, 0.2], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onDistortionSliderChanged_());
                
            % Divider
            yPos = yPos - 15;
            uilabel(obj.PanDistControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Tangential P1
            yPos = yPos - 40;
            uilabel(obj.PanDistControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Tangential p1:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblDist_P1_ = uilabel(obj.PanDistControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderDist_P1_ = uislider(obj.PanDistControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-0.2, 0.2], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onDistortionSliderChanged_());
                
            % Tangential P2
            yPos = yPos - 45;
            uilabel(obj.PanDistControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Tangential p2:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblDist_P2_ = uilabel(obj.PanDistControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '0.00', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderDist_P2_ = uislider(obj.PanDistControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [-0.2, 0.2], 'Value', 0.0, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onDistortionSliderChanged_());
                
            % Global Reset
            yPos = yPos - 45;
            obj.BtnDist_Reset_ = uibutton(obj.PanDistControl_, ...
                'Position', [20, yPos, 250, 26], ...
                'Text', 'Reset Distortion (Identity)', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onDistortionResetPressed_());
                
            % ---------------- Center Canvas ----------------
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.AxDist_ = uiaxes(obj.PanDistCanvas_, ...
                'Position', [0, 10, cW, 670], ...
                'Color', [0.10, 0.12, 0.16]);
            title(obj.AxDist_, 'Lens aberration distortion space comparison', 'Color', obj.COLOR_TEXT);
            obj.VizCore_.styleAxes(obj.AxDist_);
            hold(obj.AxDist_, 'on');
            
            % Draw original ideal grid (dashed gray dots)
            obj.DistPoints_Ref_ = plot(obj.AxDist_, obj.RefDistPoints_(1, :), obj.RefDistPoints_(2, :), ...
                'o', 'Color', [0.4, 0.4, 0.4], 'MarkerSize', 4, 'MarkerFaceColor', [0.3, 0.3, 0.3]);
                
            % Draw distorted grid lines
            obj.DistLines_Warped_ = plot(obj.AxDist_, obj.RefDistLineIndices_(1, :), obj.RefDistLineIndices_(2, :), ...
                '-', 'Color', obj.COLOR_MUTED, 'LineWidth', 1.0);
                
            % Draw distorted grid points
            obj.DistPoints_Warped_ = plot(obj.AxDist_, obj.RefDistPoints_(1, :), obj.RefDistPoints_(2, :), ...
                'o', 'Color', obj.COLOR_ACCENT, 'MarkerSize', 6, 'MarkerFaceColor', obj.COLOR_ACCENT);
                
            axis(obj.AxDist_, 'equal');
            xlim(obj.AxDist_, [-1.5, 1.5]);
            ylim(obj.AxDist_, [-1.5, 1.5]);
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            uilabel(obj.PanDistInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Aberration math', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanDistInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 110;
            uilabel(obj.PanDistInfo_, ...
                'Position', [15, iy, 230, 100], ...
                'Text', sprintf('Radial Distortion (Barrel/Pincushion):\nx_d = x * (1 + k1*r^2 + k2*r^4 + k3*r^6)\ny_d = y * (1 + k1*r^2 + k2*r^4 + k3*r^6)\n\nGoverned by the distance from the optical center (r^2 = x^2 + y^2).'), ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10);
                
            iy = iy - 130;
            uilabel(obj.PanDistInfo_, ...
                'Position', [15, iy, 230, 120], ...
                'Text', sprintf('Tangential Distortion:\ndx = 2*p1*x*y + p2*(r^2 + 2*x^2)\ndy = p1*(r^2 + 2*y^2) + 2*p2*x*y\n\nOccurs when the lens elements are not perfectly aligned parallel to the camera sensor (decentering). Adds asymmetric shear warping.'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
        end
        
        % ================================================================== %
        % Dynamic Calculation & View Update Logic (Tab 1)
        % ================================================================== %
        function update2DTransform_(obj)
            % 1. Compute H composite matrix from state
            H = OceanMath.Geometry.Transform2D.affine( ...
                obj.State2D_Tx_, obj.State2D_Ty_, obj.State2D_Rot_, ...
                obj.State2D_Sx_, obj.State2D_Sy_, obj.State2D_Shx_, obj.State2D_Shy_);
                
            obj.State2D_Matrix_ = H;
            
            % 2. Apply homogeneous coordinate warp
            gridWarped = OceanMath.Geometry.Transform2D.transformPoints(H, obj.RefGrid2DPoints_);
            shapeWarped = OceanMath.Geometry.Transform2D.transformPoints(H, obj.RefShape2DPoints_);
            
            % 3. Update Canvas handles
            obj.Grid2D_Warped_.XData = gridWarped(1, :);
            obj.Grid2D_Warped_.YData = gridWarped(2, :);
            
            obj.Shape2D_Warped_.XData = shapeWarped(1, :);
            obj.Shape2D_Warped_.YData = shapeWarped(2, :);
            
            % 4. Sync values to 3x3 Matrix Grid
            for r = 1:3
                for c = 1:3
                    obj.Grid2D_Matrix_{r,c}.Value = H(r,c);
                end
            end
        end
        
        function update2DFromMatrix_(obj, H)
            % Applies a general matrix directly, bypassing sliders
            obj.State2D_Matrix_ = H;
            
            % Apply homogeneous warp
            gridWarped = OceanMath.Geometry.Transform2D.transformPoints(H, obj.RefGrid2DPoints_);
            shapeWarped = OceanMath.Geometry.Transform2D.transformPoints(H, obj.RefShape2DPoints_);
            
            % Update visual handles
            obj.Grid2D_Warped_.XData = gridWarped(1, :);
            obj.Grid2D_Warped_.YData = gridWarped(2, :);
            
            obj.Shape2D_Warped_.XData = shapeWarped(1, :);
            obj.Shape2D_Warped_.YData = shapeWarped(2, :);
        end
        
        % ================================================================== %
        % Dynamic Calculation & View Update Logic (Tab 2)
        % ================================================================== %
        function update3DTransform_(obj)
            % Dynamic calculations of rotated cube and coordinate frame
            R = obj.State3D_R_;
            
            % 1. Rotate Cube vertices
            V_rot = R * obj.RefCubeVertices_;
            obj.Cube3D_Warped_.Vertices = V_rot';
            
            % 2. Rotate local frame axes
            xEnd = R * [1.5; 0; 0];
            yEnd = R * [0; 1.5; 0];
            zEnd = R * [0; 0; 1.5];
            
            obj.LocalAxisX_.XData = [0, xEnd(1)]; obj.LocalAxisX_.YData = [0, xEnd(2)]; obj.LocalAxisX_.ZData = [0, xEnd(3)];
            obj.LocalAxisY_.XData = [0, yEnd(1)]; obj.LocalAxisY_.YData = [0, yEnd(2)]; obj.LocalAxisY_.ZData = [0, yEnd(3)];
            obj.LocalAxisZ_.XData = [0, zEnd(1)]; obj.LocalAxisZ_.YData = [0, zEnd(2)]; obj.LocalAxisZ_.ZData = [0, zEnd(3)];
            
            % 3. Update right side 3x3 labels matrix R
            for r = 1:3
                for c = 1:3
                    obj.Grid3D_Matrix_{r,c}.Text = sprintf('%.3f', R(r,c));
                end
            end
            
            % 4. Compute and update Quaternion & Exp Map readouts
            q = OceanMath.Geometry.Rotation3D.matrixToQuat(R);
            obj.Lbl3D_Quat_.Text = sprintf('[%.3f, %.3f, %.3f, %.3f]', q(1), q(2), q(3), q(4));
            
            v = OceanMath.Geometry.Rotation3D.matrixToExpMap(R);
            obj.Lbl3D_Exp_.Text = sprintf('[%.3f, %.3f, %.3f]', v(1), v(2), v(3));
        end
        
        % ================================================================== %
        % Dynamic Calculation & View Update Logic (Tab 3)
        % ================================================================== %
        function updateCameraProjection_(obj)
            % 1. Construct camera intrinsics matrix K
            K = [obj.StateCam_Fx_, obj.StateCam_Skew_, obj.StateCam_Cx_; ...
                 0,                obj.StateCam_Fy_,   obj.StateCam_Cy_; ...
                 0,                0,                  1];
                 
            % 2. Construct camera world-to-camera extrinsic transform matrix [R | t]
            % Camera rotation matrix in world frame
            R_c2w = OceanMath.Geometry.Rotation3D.eulerToMatrix( ...
                obj.StateCam_Roll_, obj.StateCam_Pitch_, obj.StateCam_Yaw_);
            t_c2w = [obj.StateCam_X_; obj.StateCam_Y_; obj.StateCam_Z_];
            
            % Extrinsic World-to-Camera matrix: T_w2c = inv(T_c2w)
            T_w2c = [R_c2w', -R_c2w'*t_c2w; 0 0 0 1];
            
            % 3. Update Left 3D World Visualizer
            % Move camera wireframe center dot in world space
            obj.CamOrigin3D_Pt_.XData = t_c2w(1);
            obj.CamOrigin3D_Pt_.YData = t_c2w(2);
            obj.CamOrigin3D_Pt_.ZData = t_c2w(3);
            
            % Update local camera sensor frustum wireframe pyramid corners in world space
            localCorners = [ ...
                -0.6, -0.4, 1.2; ... % corner 1
                 0.6, -0.4, 1.2; ... % corner 2
                 0.6,  0.4, 1.2; ... % corner 3
                -0.6,  0.4, 1.2  ... % corner 4
            ]';
            % Transform to world coordinates: c_w = R_c2w * c_c + t_c2w
            worldCorners = R_c2w * localCorners + repmat(t_c2w, 1, 4);
            
            % Connect origin to 4 corners
            obj.Frustum3D_Line_{1}.XData = [t_c2w(1), worldCorners(1,1)]; obj.Frustum3D_Line_{1}.YData = [t_c2w(2), worldCorners(2,1)]; obj.Frustum3D_Line_{1}.ZData = [t_c2w(3), worldCorners(3,1)];
            obj.Frustum3D_Line_{2}.XData = [t_c2w(1), worldCorners(1,2)]; obj.Frustum3D_Line_{2}.YData = [t_c2w(2), worldCorners(2,2)]; obj.Frustum3D_Line_{2}.ZData = [t_c2w(3), worldCorners(3,2)];
            obj.Frustum3D_Line_{3}.XData = [t_c2w(1), worldCorners(1,3)]; obj.Frustum3D_Line_{3}.YData = [t_c2w(2), worldCorners(2,3)]; obj.Frustum3D_Line_{3}.ZData = [t_c2w(3), worldCorners(3,3)];
            obj.Frustum3D_Line_{4}.XData = [t_c2w(1), worldCorners(1,4)]; obj.Frustum3D_Line_{4}.YData = [t_c2w(2), worldCorners(2,4)]; obj.Frustum3D_Line_{4}.ZData = [t_c2w(3), worldCorners(3,4)];
            
            % Connect 4 corners loop
            obj.Frustum3D_Line_{5}.XData = [worldCorners(1,1), worldCorners(1,2)]; obj.Frustum3D_Line_{5}.YData = [worldCorners(2,1), worldCorners(2,2)]; obj.Frustum3D_Line_{5}.ZData = [worldCorners(3,1), worldCorners(3,2)];
            obj.Frustum3D_Line_{6}.XData = [worldCorners(1,2), worldCorners(1,3)]; obj.Frustum3D_Line_{6}.YData = [worldCorners(2,2), worldCorners(2,3)]; obj.Frustum3D_Line_{6}.ZData = [worldCorners(3,2), worldCorners(3,3)];
            obj.Frustum3D_Line_{7}.XData = [worldCorners(1,3), worldCorners(1,4)]; obj.Frustum3D_Line_{7}.YData = [worldCorners(2,3), worldCorners(2,4)]; obj.Frustum3D_Line_{7}.ZData = [worldCorners(3,3), worldCorners(3,4)];
            obj.Frustum3D_Line_{8}.XData = [worldCorners(1,4), worldCorners(1,1)]; obj.Frustum3D_Line_{8}.YData = [worldCorners(2,4), worldCorners(2,1)]; obj.Frustum3D_Line_{8}.ZData = [worldCorners(3,4), worldCorners(3,1)];
            
            % 4. Call mathematical CameraProjection module to compute viewport 2D coordinates
            proj = OceanMath.Geometry.CameraProjection( ...
                'Intrinsics', K, ...
                'Extrinsics', T_w2c, ...
                'ProjectionType', obj.DropdownCam_Type_.Value);
                
            [pts2D, inFront] = proj.project(obj.RefHouseVertices_);
            
            % Update Right 2D Projected View lines
            for i = 1:size(obj.RefHouseLines_, 1)
                idx = obj.RefHouseLines_(i, :);
                
                % Only render wireframe line segment if BOTH vertices are in front of camera sensor
                if inFront(idx(1)) && inFront(idx(2))
                    obj.House2D_Line_{i}.Visible = 'on';
                    obj.House2D_Line_{i}.XData = pts2D(1, idx);
                    obj.House2D_Line_{i}.YData = pts2D(2, idx);
                else
                    obj.House2D_Line_{i}.Visible = 'off';
                end
            end
        end
        
        % ================================================================== %
        % Dynamic Calculation & View Update Logic (Tab 4)
        % ================================================================== %
        function updateLensDistortion_(obj)
            % 1. Instantiate mathematical LensDistortion model
            distortion = OceanMath.Geometry.LensDistortion( ...
                'Radial', [obj.StateDist_K1_, obj.StateDist_K2_, obj.StateDist_K3_], ...
                'Tangential', [obj.StateDist_P1_, obj.StateDist_P2_]);
                
            % 2. Warp mesh points and line coordinates
            warpedPoints = distortion.distort(obj.RefDistPoints_);
            warpedLines = distortion.distort(obj.RefDistLineIndices_);
            
            % 3. Update canvas plot objects
            obj.DistPoints_Warped_.XData = warpedPoints(1, :);
            obj.DistPoints_Warped_.YData = warpedPoints(2, :);
            
            obj.DistLines_Warped_.XData = warpedLines(1, :);
            obj.DistLines_Warped_.YData = warpedLines(2, :);
        end
        
        % ================================================================== %
        % Event Callbacks (Tab 1)
        % ================================================================== %
        function on2DSliderChanged_(obj)
            % Sync values from sliders to state
            obj.State2D_Tx_  = obj.Slider2D_Tx_.Value;
            obj.State2D_Ty_  = obj.Slider2D_Ty_.Value;
            obj.State2D_Rot_ = obj.Slider2D_Rot_.Value;
            obj.State2D_Sx_  = obj.Slider2D_Sx_.Value;
            obj.State2D_Sy_  = obj.Slider2D_Sy_.Value;
            obj.State2D_Shx_ = obj.Slider2D_Shx_.Value;
            obj.State2D_Shy_ = obj.Slider2D_Shy_.Value;
            
            % Update readouts
            obj.Lbl2D_Tx_.Text  = sprintf('%.2f', obj.State2D_Tx_);
            obj.Lbl2D_Ty_.Text  = sprintf('%.2f', obj.State2D_Ty_);
            obj.Lbl2D_Rot_.Text = sprintf('%.1f', obj.State2D_Rot_);
            obj.Lbl2D_Sx_.Text  = sprintf('%.2f', obj.State2D_Sx_);
            obj.Lbl2D_Sy_.Text  = sprintf('%.2f', obj.State2D_Sy_);
            obj.Lbl2D_Shx_.Text = sprintf('%.2f', obj.State2D_Shx_);
            obj.Lbl2D_Shy_.Text = sprintf('%.2f', obj.State2D_Shy_);
            
            % Re-compute and render if AutoApply is checked
            if obj.Chk2D_AutoApply_.Value
                obj.update2DTransform_();
            end
        end
        
        function on2DMatrixChanged_(obj)
            % Custom matrix edited, disable auto sliders apply highlight
            obj.Chk2D_AutoApply_.Value = false;
        end
        
        function on2DApplyMatrixPressed_(obj)
            % Pull matrix directly from the uieditfields
            H = zeros(3,3);
            for r = 1:3
                for c = 1:3
                    H(r,c) = obj.Grid2D_Matrix_{r,c}.Value;
                end
            end
            
            % Apply H directly
            obj.update2DFromMatrix_(H);
        end
        
        function on2DResetPressed_(obj)
            % Reset State to defaults
            obj.State2D_Tx_  = 0;
            obj.State2D_Ty_  = 0;
            obj.State2D_Rot_ = 0;
            obj.State2D_Sx_  = 1.0;
            obj.State2D_Sy_  = 1.0;
            obj.State2D_Shx_ = 0;
            obj.State2D_Shy_ = 0;
            obj.State2D_Matrix_ = eye(3);
            
            % Sync sliders
            obj.Slider2D_Tx_.Value  = 0;
            obj.Slider2D_Ty_.Value  = 0;
            obj.Slider2D_Rot_.Value = 0;
            obj.Slider2D_Sx_.Value  = 1.0;
            obj.Slider2D_Sy_.Value  = 1.0;
            obj.Slider2D_Shx_.Value = 0;
            obj.Slider2D_Shy_.Value = 0;
            
            % Sync readouts
            obj.Lbl2D_Tx_.Text  = '0.00';
            obj.Lbl2D_Ty_.Text  = '0.00';
            obj.Lbl2D_Rot_.Text = '0.0';
            obj.Lbl2D_Sx_.Text  = '1.00';
            obj.Lbl2D_Sy_.Text  = '1.00';
            obj.Lbl2D_Shx_.Text = '0.00';
            obj.Lbl2D_Shy_.Text = '0.00';
            
            obj.Chk2D_AutoApply_.Value = true;
            
            % Update
            obj.update2DTransform_();
        end
        
        % ================================================================== %
        % Event Callbacks (Tab 2)
        % ================================================================== %
        function on3DFormatChanged_(obj)
            % Hide all panels
            obj.Pan3D_Euler_.Visible = 'off';
            obj.Pan3D_Quat_.Visible = 'off';
            obj.Pan3D_AxisAngle_.Visible = 'off';
            obj.Pan3D_ExpMap_.Visible = 'off';
            
            % Show the active selection
            val = obj.Dropdown3D_Format_.Value;
            switch val
                case 'Euler Angles'
                    obj.Pan3D_Euler_.Visible = 'on';
                    % Force sync sliders from current matrix state
                    [r, p, y] = OceanMath.Geometry.Rotation3D.matrixToEuler(obj.State3D_R_);
                    obj.Slider3D_Roll_.Value  = r;
                    obj.Slider3D_Pitch_.Value = p;
                    obj.Slider3D_Yaw_.Value   = y;
                    obj.Lbl3D_Roll_.Text  = sprintf('%.1f', r);
                    obj.Lbl3D_Pitch_.Text = sprintf('%.1f', p);
                    obj.Lbl3D_Yaw_.Text   = sprintf('%.1f', y);
                    
                case 'Quaternions'
                    obj.Pan3D_Quat_.Visible = 'on';
                    q = OceanMath.Geometry.Rotation3D.matrixToQuat(obj.State3D_R_);
                    obj.Field3D_Qw_.Value = q(1);
                    obj.Field3D_Qx_.Value = q(2);
                    obj.Field3D_Qy_.Value = q(3);
                    obj.Field3D_Qz_.Value = q(4);
                    
                case 'Axis-Angle'
                    obj.Pan3D_AxisAngle_.Visible = 'on';
                    [ax, th] = OceanMath.Geometry.Rotation3D.matrixToAxisAngle(obj.State3D_R_);
                    obj.Field3D_Ax_.Value = ax(1);
                    obj.Field3D_Ay_.Value = ax(2);
                    obj.Field3D_Az_.Value = ax(3);
                    obj.Slider3D_Ang_.Value = th;
                    obj.Lbl3D_Ang_.Text = sprintf('%.1f', th);
                    
                case 'Exponential Map'
                    obj.Pan3D_ExpMap_.Visible = 'on';
                    v = OceanMath.Geometry.Rotation3D.matrixToExpMap(obj.State3D_R_);
                    obj.Slider3D_V1_.Value = v(1);
                    obj.Slider3D_V2_.Value = v(2);
                    obj.Slider3D_V3_.Value = v(3);
                    obj.Lbl3D_V1_.Text = sprintf('%.2f', v(1));
                    obj.Lbl3D_V2_.Text = sprintf('%.2f', v(2));
                    obj.Lbl3D_V3_.Text = sprintf('%.2f', v(3));
            end
        end
        
        function on3DEulerSliderChanged_(obj)
            r = obj.Slider3D_Roll_.Value;
            p = obj.Slider3D_Pitch_.Value;
            y = obj.Slider3D_Yaw_.Value;
            
            obj.Lbl3D_Roll_.Text  = sprintf('%.1f', r);
            obj.Lbl3D_Pitch_.Text = sprintf('%.1f', p);
            obj.Lbl3D_Yaw_.Text   = sprintf('%.1f', y);
            
            % Compute matrix
            obj.State3D_R_ = OceanMath.Geometry.Rotation3D.eulerToMatrix(r, p, y);
            obj.update3DTransform_();
        end
        
        function on3DQuatFieldChanged_(obj)
            qw = obj.Field3D_Qw_.Value;
            qx = obj.Field3D_Qx_.Value;
            qy = obj.Field3D_Qy_.Value;
            qz = obj.Field3D_Qz_.Value;
            
            % Normalize safely
            q = [qw, qx, qy, qz];
            n = norm(q);
            if n < 1e-12
                q = [1, 0, 0, 0];
            else
                q = q / n;
            end
            
            % Update fields to normalized values
            obj.Field3D_Qw_.Value = q(1);
            obj.Field3D_Qx_.Value = q(2);
            obj.Field3D_Qy_.Value = q(3);
            obj.Field3D_Qz_.Value = q(4);
            
            % Compute matrix
            obj.State3D_R_ = OceanMath.Geometry.Rotation3D.quatToMatrix(q);
            obj.update3DTransform_();
        end
        
        function on3DAxisAngleChanged_(obj)
            ax = obj.Field3D_Ax_.Value;
            ay = obj.Field3D_Ay_.Value;
            az = obj.Field3D_Az_.Value;
            theta = obj.Slider3D_Ang_.Value;
            
            obj.Lbl3D_Ang_.Text = sprintf('%.1f', theta);
            
            % Force axis normalization
            v = [ax, ay, az];
            n = norm(v);
            if n < 1e-12
                v = [0, 0, 1];
            else
                v = v / n;
            end
            
            obj.Field3D_Ax_.Value = v(1);
            obj.Field3D_Ay_.Value = v(2);
            obj.Field3D_Az_.Value = v(3);
            
            % Compute R
            obj.State3D_R_ = OceanMath.Geometry.Rotation3D.axisAngleToMatrix(v, theta);
            obj.update3DTransform_();
        end
        
        function on3DExpMapChanged_(obj)
            v1 = obj.Slider3D_V1_.Value;
            v2 = obj.Slider3D_V2_.Value;
            v3 = obj.Slider3D_V3_.Value;
            
            obj.Lbl3D_V1_.Text = sprintf('%.2f', v1);
            obj.Lbl3D_V2_.Text = sprintf('%.2f', v2);
            obj.Lbl3D_V3_.Text = sprintf('%.2f', v3);
            
            % Compute matrix
            obj.State3D_R_ = OceanMath.Geometry.Rotation3D.expMapToMatrix([v1; v2; v3]);
            obj.update3DTransform_();
        end
        
        function onSetStartRotation_(obj)
            obj.State3D_QStart_ = OceanMath.Geometry.Rotation3D.matrixToQuat(obj.State3D_R_);
            obj.Btn3D_SetStart_.Text = 'Start Rot Captured';
            obj.Btn3D_SetStart_.FontColor = obj.COLOR_SECONDARY;
        end
        
        function onSetEndRotation_(obj)
            obj.State3D_QEnd_ = OceanMath.Geometry.Rotation3D.matrixToQuat(obj.State3D_R_);
            obj.Btn3D_SetEnd_.Text = 'End Rot Captured';
            obj.Btn3D_SetEnd_.FontColor = obj.COLOR_SECONDARY;
        end
        
        function onSlerpSliderChanged_(obj)
            t = obj.Slider3D_SlerpT_.Value;
            obj.Lbl3D_SlerpT_.Text = sprintf('%.2f', t);
            
            % Slerp interpolate
            q = OceanMath.Geometry.Rotation3D.slerp(obj.State3D_QStart_, obj.State3D_QEnd_, t);
            obj.State3D_R_ = OceanMath.Geometry.Rotation3D.quatToMatrix(q);
            obj.update3DTransform_();
        end
        
        function onAnimateSlerpPressed_(obj)
            if obj.SlerpActive_
                return;
            end
            obj.SlerpActive_ = true;
            obj.Btn3D_AnimateSlerp_.Text = 'Animating...';
            obj.Btn3D_AnimateSlerp_.BackgroundColor = obj.COLOR_MUTED;
            
            % Start at t = 0
            obj.Slider3D_SlerpT_.Value = 0.0;
            obj.Lbl3D_SlerpT_.Text = '0.00';
            
            % Build high-precision timer to run animation in 30 steps
            obj.SlerpTimer_ = timer( ...
                'ExecutionMode', 'fixedRate', ...
                'Period', 0.05, ...
                'TasksToExecute', 21, ...
                'TimerFcn', @(src, ~) obj.slerpStep_(src.TasksExecuted / 20));
            start(obj.SlerpTimer_);
        end
        
        function slerpStep_(obj, t)
            % Clip t safely to [0, 1]
            t = min(1.0, max(0.0, t));
            
            % Update slider visual
            obj.Slider3D_SlerpT_.Value = t;
            obj.Lbl3D_SlerpT_.Text = sprintf('%.2f', t);
            
            % Interpolate and update visualizer
            q = OceanMath.Geometry.Rotation3D.slerp(obj.State3D_QStart_, obj.State3D_QEnd_, t);
            obj.State3D_R_ = OceanMath.Geometry.Rotation3D.quatToMatrix(q);
            obj.update3DTransform_();
            drawnow;
            
            % Teardown timer when animation ends
            if t >= 1.0
                obj.SlerpActive_ = false;
                obj.Btn3D_AnimateSlerp_.Text = 'Animate Slerp Interpolation';
                obj.Btn3D_AnimateSlerp_.BackgroundColor = [0.15, 0.50, 0.80];
                delete(obj.SlerpTimer_);
            end
        end
        
        function on3DResetPressed_(obj)
            obj.State3D_R_ = eye(3);
            
            % Reset capture labels
            obj.Btn3D_SetStart_.Text = 'Capture Start Rot';
            obj.Btn3D_SetStart_.FontColor = obj.COLOR_TEXT;
            obj.Btn3D_SetEnd_.Text = 'Capture End Rot';
            obj.Btn3D_SetEnd_.FontColor = obj.COLOR_TEXT;
            
            obj.Slider3D_SlerpT_.Value = 0.0;
            obj.Lbl3D_SlerpT_.Text = '0.00';
            
            % Force reload format layout values
            obj.on3DFormatChanged_();
            
            % Update 3D
            obj.update3DTransform_();
        end
        
        % ================================================================== %
        % Event Callbacks (Tab 3)
        % ================================================================== %
        function onCameraSliderChanged_(obj)
            obj.StateCam_Fx_    = obj.SliderCam_Fx_.Value;
            obj.StateCam_Fy_    = obj.SliderCam_Fy_.Value;
            obj.StateCam_Cx_    = obj.SliderCam_Cx_.Value;
            obj.StateCam_Cy_    = obj.SliderCam_Cy_.Value;
            obj.StateCam_Skew_  = obj.SliderCam_Skew_.Value;
            obj.StateCam_X_     = obj.SliderCam_X_.Value;
            obj.StateCam_Y_     = obj.SliderCam_Y_.Value;
            obj.StateCam_Z_     = obj.SliderCam_Z_.Value;
            obj.StateCam_Roll_  = obj.SliderCam_Roll_.Value;
            obj.StateCam_Pitch_ = obj.SliderCam_Pitch_.Value;
            obj.StateCam_Yaw_   = obj.SliderCam_Yaw_.Value;
            
            % Sync readouts
            obj.LblCam_Fx_.Text    = sprintf('%.0f', obj.StateCam_Fx_);
            obj.LblCam_Fy_.Text    = sprintf('%.0f', obj.StateCam_Fy_);
            obj.LblCam_Cx_.Text    = sprintf('%.0f', obj.StateCam_Cx_);
            obj.LblCam_Cy_.Text    = sprintf('%.0f', obj.StateCam_Cy_);
            obj.LblCam_Skew_.Text  = sprintf('%.0f', obj.StateCam_Skew_);
            obj.LblCam_X_.Text     = sprintf('%.1f', obj.StateCam_X_);
            obj.LblCam_Y_.Text     = sprintf('%.1f', obj.StateCam_Y_);
            obj.LblCam_Z_.Text     = sprintf('%.1f', obj.StateCam_Z_);
            obj.LblCam_Roll_.Text  = sprintf('%.1f', obj.StateCam_Roll_);
            obj.LblCam_Pitch_.Text = sprintf('%.1f', obj.StateCam_Pitch_);
            obj.LblCam_Yaw_.Text   = sprintf('%.1f', obj.StateCam_Yaw_);
            
            obj.updateCameraProjection_();
        end
        
        function onCameraResetPressed_(obj)
            obj.StateCam_Fx_ = 500;
            obj.StateCam_Fy_ = 500;
            obj.StateCam_Cx_ = 320;
            obj.StateCam_Cy_ = 240;
            obj.StateCam_Skew_ = 0;
            obj.StateCam_X_ = 0;
            obj.StateCam_Y_ = 0;
            obj.StateCam_Z_ = -8.0;
            obj.StateCam_Roll_ = 0;
            obj.StateCam_Pitch_ = 0;
            obj.StateCam_Yaw_ = 0;
            
            % Sync sliders
            obj.SliderCam_Fx_.Value = 500;
            obj.SliderCam_Fy_.Value = 500;
            obj.SliderCam_Cx_.Value = 320;
            obj.SliderCam_Cy_.Value = 240;
            obj.SliderCam_Skew_.Value = 0;
            obj.SliderCam_X_.Value = 0;
            obj.SliderCam_Y_.Value = 0;
            obj.SliderCam_Z_.Value = -8.0;
            obj.SliderCam_Roll_.Value = 0;
            obj.SliderCam_Pitch_.Value = 0;
            obj.SliderCam_Yaw_.Value = 0;
            
            obj.DropdownCam_Type_.Value = 'Perspective';
            
            obj.onCameraSliderChanged_();
        end
        
        % ================================================================== %
        % Event Callbacks (Tab 4)
        % ================================================================== %
        function onDistortionSliderChanged_(obj)
            obj.StateDist_K1_ = obj.SliderDist_K1_.Value;
            obj.StateDist_K2_ = obj.SliderDist_K2_.Value;
            obj.StateDist_K3_ = obj.SliderDist_K3_.Value;
            obj.StateDist_P1_ = obj.SliderDist_P1_.Value;
            obj.StateDist_P2_ = obj.SliderDist_P2_.Value;
            
            % Update readouts
            obj.LblDist_K1_.Text = sprintf('%.2f', obj.StateDist_K1_);
            obj.LblDist_K2_.Text = sprintf('%.2f', obj.StateDist_K2_);
            obj.LblDist_K3_.Text = sprintf('%.2f', obj.StateDist_K3_);
            obj.LblDist_P1_.Text = sprintf('%.2f', obj.StateDist_P1_);
            obj.LblDist_P2_.Text = sprintf('%.2f', obj.StateDist_P2_);
            
            obj.updateLensDistortion_();
        end
        
        function onDistortionResetPressed_(obj)
            obj.StateDist_K1_ = 0.0;
            obj.StateDist_K2_ = 0.0;
            obj.StateDist_K3_ = 0.0;
            obj.StateDist_P1_ = 0.0;
            obj.StateDist_P2_ = 0.0;
            
            % Sync sliders
            obj.SliderDist_K1_.Value = 0.0;
            obj.SliderDist_K2_.Value = 0.0;
            obj.SliderDist_K3_.Value = 0.0;
            obj.SliderDist_P1_.Value = 0.0;
            obj.SliderDist_P2_.Value = 0.0;
            
            obj.onDistortionSliderChanged_();
        end
        
        function onClose_(obj)
            delete(obj);
        end
        
        % ================================================================== %
        % Tab 5: Image Pyramids Tab Builder
        % ================================================================== %
        function buildTabPyr_(obj)
            % Sidebar panel
            obj.PanPyrControl_ = uipanel(obj.TabPyr_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Canvas panel
            obj.PanPyrCanvas_ = uipanel(obj.TabPyr_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Right explanation panel
            obj.PanPyrInfo_ = uipanel(obj.TabPyr_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % ---------------- Left Controls ----------------
            yPos = 655;
            uilabel(obj.PanPyrControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Image Source', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanPyrControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            yPos = yPos - 40;
            obj.DropdownPyr_Image_ = uidropdown(obj.PanPyrControl_, ...
                'Items', {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg', 'Synthetic Chirp', 'Synthetic Grid'}, ...
                'Value', obj.StatePyr_ImageName_, ...
                'Position', [15, yPos, 150, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onPyrImageChanged_());
                
            obj.BtnPyr_LoadCustom_ = uibutton(obj.PanPyrControl_, ...
                'Text', 'Load Custom...', ...
                'Position', [175, yPos, 130, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onPyrLoadCustomPressed_());
                
            yPos = yPos - 50;
            uilabel(obj.PanPyrControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Pyramid Parameters', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanPyrControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Mode: Gaussian vs Laplacian vs Local Laplacian vs MIP-Map Resampling
            yPos = yPos - 40;
            uilabel(obj.PanPyrControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Pyramid Mode:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            modeVal = [obj.StatePyr_Mode_ ' Pyramid'];
            if strcmp(obj.StatePyr_Mode_, 'Local Laplacian')
                modeVal = 'Local Laplacian';
            elseif strcmp(obj.StatePyr_Mode_, 'MIP-Map')
                modeVal = 'MIP-Map Resampling';
            end
            obj.DropdownPyr_Mode_ = uidropdown(obj.PanPyrControl_, ...
                'Items', {'Gaussian Pyramid', 'Laplacian Pyramid', 'Local Laplacian', 'MIP-Map Resampling'}, ...
                'Value', modeVal, ...
                'Position', [15, yPos-10, 280, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onPyrModeChanged_());
                
            % Levels (2 to 6)
            yPos = yPos - 55;
            uilabel(obj.PanPyrControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Pyramid Levels:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblPyr_Levels_ = uilabel(obj.PanPyrControl_, 'Position', [200, yPos+18, 70, 20], 'Text', num2str(obj.StatePyr_Levels_), 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderPyr_Levels_ = uislider(obj.PanPyrControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [2, 6], 'Value', obj.StatePyr_Levels_, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onPyrLevelsChanged_());
                
            % Kernel type dropdown
            yPos = yPos - 45;
            uilabel(obj.PanPyrControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Smoothing Kernel:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownPyr_Kernel_ = uidropdown(obj.PanPyrControl_, ...
                'Items', {'Gaussian', 'Triangle', 'Box', '2x2 Block', 'Windowed Sinc'}, ...
                'Value', obj.StatePyr_Kernel_, ...
                'Position', [15, yPos-10, 280, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onPyrKernelChanged_());
                
            % Kernel size (3 to 9)
            yPos = yPos - 55;
            uilabel(obj.PanPyrControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Kernel Size (Odd):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblPyr_KernelSize_ = uilabel(obj.PanPyrControl_, 'Position', [200, yPos+18, 70, 20], 'Text', num2str(obj.StatePyr_KernelSize_), 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderPyr_KernelSize_ = uislider(obj.PanPyrControl_, 'Position', [25, yPos+10, 240, 3], 'Limits', [3, 9], 'Value', obj.StatePyr_KernelSize_, ...
                'MajorTicks', [], 'MinorTicks', [], 'ValueChangedFcn', @(~,~) obj.onPyrKernelSizeChanged_());
                
            % Global Reset
            yPos = yPos - 50;
            obj.BtnPyr_Reset_ = uibutton(obj.PanPyrControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', 'Reset Parameters', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onPyrResetPressed_());
                
            % Bypass Filter checkbox
            yPos = yPos - 35;
            obj.ChkPyr_Bypass_ = uicheckbox(obj.PanPyrControl_, ...
                'Position', [20, yPos, 250, 22], ...
                'Text', 'Bypass Low-Pass Filter (Show Aliasing)', ...
                'Value', false, ...
                'FontColor', [1.0, 0.75, 0.20], ...
                'FontSize', 10, ...
                'ValueChangedFcn', @(~,~) obj.onPyrBypassChanged_());
                
            % Show Theory button
            yPos = yPos - 38;
            obj.BtnPyr_Theory_ = uibutton(obj.PanPyrControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', '[ ? ] Learn: What is a Pyramid?', ...
                'FontWeight', 'bold', ...
                'FontSize', 10, ...
                'BackgroundColor', [0.13, 0.25, 0.45], ...
                'FontColor', [0.50, 0.85, 1.00], ...
                'ButtonPushedFcn', @(~,~) obj.onPyrShowTheory_());
                
            % ---------------- Center Canvas ----------------
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            
            obj.AxPyr_ = cell(1, 6);
            obj.ImgPyr_ = cell(1, 6);
            margin_x = 15;
            margin_y = 15;
            gap_x = 15;
            gap_y = 45;
            ax_w = (cW - 2 * margin_x - 2 * gap_x) / 3;
            ax_h = (670 - 2 * margin_y - gap_y) / 2;
            
            for r = 1:2
                for c = 1:3
                    idx = (r-1)*3 + c;
                    ax_x = margin_x + (c-1) * (ax_w + gap_x);
                    ax_y = 670 - margin_y - r * ax_h - (r-1) * gap_y;
                    
                    obj.AxPyr_{idx} = uiaxes(obj.PanPyrCanvas_, ...
                        'Position', [ax_x, ax_y, ax_w, ax_h], ...
                        'Color', [0.10, 0.12, 0.16]);
                    obj.VizCore_.styleAxes(obj.AxPyr_{idx});
                    disableDefaultInteractivity(obj.AxPyr_{idx});
                    hold(obj.AxPyr_{idx}, 'on');
                    
                    obj.ImgPyr_{idx} = [];
                end
            end
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            obj.LblPyr_Info_Title_ = uilabel(obj.PanPyrInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Multi-Resolution Math', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanPyrInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 190;
            obj.LblPyr_Info_Desc_ = uilabel(obj.PanPyrInfo_, ...
                'Position', [15, iy, 230, 180], ...
                'Text', '', ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'WordWrap', 'on');
                
            iy = iy - 200;
            obj.LblPyr_Info_Stats_ = uilabel(obj.PanPyrInfo_, ...
                'Position', [15, iy, 230, 185], ...
                'Text', '', ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 10, ...
                'WordWrap', 'on');
        end
        
        % ================================================================== %
        % Dynamic Calculation & View Update Logic (Tab 5)
        % ================================================================== %
        function updateTabPyr_(obj)
            % 1. Ensure image is loaded
            if isempty(obj.StatePyr_Image_)
                if strcmp(obj.StatePyr_ImageName_, 'Synthetic Chirp')
                    gridSize = 512;
                    [x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
                    r = sqrt(x.^2 + y.^2);
                    obj.StatePyr_Image_ = 0.5 + 0.5 * sin(120 * r.^2);
                elseif strcmp(obj.StatePyr_ImageName_, 'Synthetic Grid')
                    gridSize = 512;
                    [x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
                    obj.StatePyr_Image_ = 0.5 + 0.5 * sign(sin(40 * pi * x) .* sin(40 * pi * y));
                else
                    script_dir = fileparts(mfilename('fullpath'));
                    project_root = fullfile(script_dir, '..', '..', '..');
                    image_path = fullfile(project_root, 'Data', 'images', obj.StatePyr_ImageName_);
                    if exist(image_path, 'file')
                        img = imread(image_path);
                        if size(img, 3) > 1 && strcmp(obj.StatePyr_ImageName_, 'ocean_img_gray.jpg')
                            img = rgb2gray(img);
                        end
                        obj.StatePyr_Image_ = double(img) / 255.0;
                    else
                        obj.StatePyr_Image_ = zeros(256, 256, 3);
                    end
                end
            end
            
            % 2. Resolve smoothing kernel
            try
                if strcmp(obj.StatePyr_Kernel_, 'Gaussian')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('gaussian', 'Size', obj.StatePyr_KernelSize_);
                elseif strcmp(obj.StatePyr_Kernel_, 'Triangle')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('triangle', obj.StatePyr_KernelSize_);
                elseif strcmp(obj.StatePyr_Kernel_, '2x2 Block')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('box', 2);
                elseif strcmp(obj.StatePyr_Kernel_, 'Windowed Sinc')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('sinc', 'Size', obj.StatePyr_KernelSize_);
                else
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('box', obj.StatePyr_KernelSize_);
                end
            catch ME
                kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', obj.StatePyr_KernelSize_);
            end
            
            numLevels = obj.StatePyr_Levels_;
            if strcmp(obj.StatePyr_Mode_, 'Gaussian')
                pyramid = OceanMath.Image.MultiResolution.GaussianPyramid('Kernel', kernel, 'BypassFilter', obj.StatePyr_Bypass_);
                pyramid.build(obj.StatePyr_Image_, numLevels);
                
                I_rec = pyramid.reconstruct();
                max_err = 0.0;
            elseif strcmp(obj.StatePyr_Mode_, 'Laplacian')
                pyramid = OceanMath.Image.MultiResolution.LaplacianPyramid('Kernel', kernel, 'BypassFilter', obj.StatePyr_Bypass_);
                pyramid.build(obj.StatePyr_Image_, numLevels);
                
                I_rec = pyramid.reconstruct();
                max_err = max(abs(obj.StatePyr_Image_(:) - I_rec(:)));
            elseif strcmp(obj.StatePyr_Mode_, 'MIP-Map')
                % MIP-map level selection and anisotropic filtering comparison
                % 1. Build the MIP-map of the loaded image
                mipmap = OceanMath.Image.MultiResolution.MipMap('Kernel', kernel);
                mipmap.build(obj.StatePyr_Image_, numLevels);
                
                % 2. Uniform downsampling by 0.35x
                scale_factor = 0.35;
                [H_orig, W_orig, C_orig] = size(obj.StatePyr_Image_);
                W_out = round(W_orig * scale_factor);
                H_out = round(H_orig * scale_factor);
                [X_out, Y_out] = meshgrid(1:W_out, 1:H_out);
                U_tex = X_out / scale_factor;
                V_tex = Y_out / scale_factor;
                
                J_uniform = [1/scale_factor, 0; 0, 1/scale_factor];
                d_level = mipmap.computeLevel(J_uniform);
                
                I_finer = zeros(H_out, W_out, C_orig);
                I_coarser = zeros(H_out, W_out, C_orig);
                I_trilinear = zeros(H_out, W_out, C_orig);
                
                % Sample for uniform downsampling
                for c = 1:C_orig
                    % Build temporary single-channel mipmap
                    mip_c = OceanMath.Image.MultiResolution.MipMap('Kernel', kernel);
                    mip_c.Levels = cellfun(@(x) x(:,:,c), mipmap.Levels, 'UniformOutput', false);
                    mip_c.NumLevels = mipmap.NumLevels;
                    
                    for r = 1:H_out
                        for col = 1:W_out
                            I_finer(r, col, c) = mip_c.sample(U_tex(r,col), V_tex(r,col), J_uniform, 'Floor');
                            I_coarser(r, col, c) = mip_c.sample(U_tex(r,col), V_tex(r,col), J_uniform, 'Ceil');
                            I_trilinear(r, col, c) = mip_c.sample(U_tex(r,col), V_tex(r,col), J_uniform, 'Trilinear');
                        end
                    end
                end
                
                % 3. Perspective floor rendering for comparison (128x128 resolution for real-time responsiveness)
                W_floor = 128; H_floor = 128;
                I_floor_mip = zeros(H_floor, W_floor, C_orig);
                I_floor_ewa = zeros(H_floor, W_floor, C_orig);
                
                ewa_sampler = OceanMath.Image.Resampling.EWASampler('RadiusSquared', 3.0);
                
                % Perspective params adjusted for 128x128 canvas mapping to original image
                xc = W_floor / 2;
                yh = 35; % Horizon line
                su = 50; sv = 400;
                uc = W_orig / 2; vc = 5;
                
                for y = 1:H_floor
                    if y <= yh
                        % Sky/background (Dark theme background)
                        for c = 1:C_orig
                            I_floor_mip(y, :, c) = 0.08;
                            I_floor_ewa(y, :, c) = 0.08;
                        end
                        continue;
                    end
                    
                    dy = y - yh;
                    for x = 1:W_floor
                        u = uc + su * (x - xc) / dy;
                        v = vc + sv / dy;
                        
                        % Analytical Jacobian
                        dudx = su / dy;
                        dudy = -su * (x - xc) / (dy^2);
                        dvdx = 0;
                        dvdy = -sv / (dy^2);
                        J = [dudx, dudy; dvdx, dvdy];
                        
                        % Check bounds
                        if u < 1 || u > W_orig || v < 1 || v > H_orig
                            for c = 1:C_orig
                                I_floor_mip(y, x, c) = 0.15;
                                I_floor_ewa(y, x, c) = 0.15;
                            end
                            continue;
                        end
                        
                        % Sample MIP (Trilinear)
                        for c = 1:C_orig
                            mip_c = OceanMath.Image.MultiResolution.MipMap('Kernel', kernel);
                            mip_c.Levels = cellfun(@(x) x(:,:,c), mipmap.Levels, 'UniformOutput', false);
                            mip_c.NumLevels = mipmap.NumLevels;
                            I_floor_mip(y, x, c) = mip_c.sample(u, v, J, 'Trilinear');
                        end
                        
                        % Sample EWA
                        I_floor_ewa(y, x, :) = ewa_sampler.sample(obj.StatePyr_Image_, u, v, J);
                    end
                end
                
                I_finer = min(max(I_finer, 0), 1);
                I_coarser = min(max(I_coarser, 0), 1);
                I_trilinear = min(max(I_trilinear, 0), 1);
                I_floor_mip = min(max(I_floor_mip, 0), 1);
                I_floor_ewa = min(max(I_floor_ewa, 0), 1);
                
                I_rec = I_trilinear;
                max_err = 0.0;
            else
                % Local Laplacian detail enhancement mode
                llf = OceanMath.Image.MultiResolution.LocalLaplacianFilter(...
                    'NumLevels', numLevels, ...
                    'NumBins', 8, ...
                    'Sigma', 0.1, ...
                    'Alpha', 0.5, ...
                    'Beta', 1.0);
                I_rec = llf.filter(obj.StatePyr_Image_);
                max_err = 0.0;
            end
            
            % 4. Render levels to the axes
            for idx = 1:6
                cla(obj.AxPyr_{idx});
                title(obj.AxPyr_{idx}, '');
                set(obj.AxPyr_{idx}, 'Visible', 'off');
            end
            
            if strcmp(obj.StatePyr_Mode_, 'Gaussian')
                for idx = 1:numLevels
                    ax = obj.AxPyr_{idx};
                    set(ax, 'Visible', 'on');
                    
                    I_level = pyramid.Levels{idx};
                    I_level = min(max(I_level, 0), 1);
                    
                    imshow(I_level, 'Parent', ax);
                    [h, w, ~] = size(I_level);
                    title(ax, sprintf('Gaussian Level %d\n(%dx%d)', idx, w, h), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                end
            elseif strcmp(obj.StatePyr_Mode_, 'Laplacian')
                for idx = 1:(numLevels-1)
                    ax = obj.AxPyr_{idx};
                    set(ax, 'Visible', 'on');
                    
                    I_level = pyramid.Levels{idx};
                    I_vis = I_level * 4.0 + 0.5;
                    I_vis = min(max(I_vis, 0), 1);
                    
                    imshow(I_vis, 'Parent', ax);
                    [h, w, ~] = size(I_level);
                    title(ax, sprintf('Laplacian Level %d\n(%dx%d)', idx, w, h), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                end
                
                ax = obj.AxPyr_{numLevels};
                set(ax, 'Visible', 'on');
                I_top = pyramid.Levels{numLevels};
                I_top = min(max(I_top, 0), 1);
                imshow(I_top, 'Parent', ax);
                [h, w, ~] = size(I_top);
                title(ax, sprintf('Gaussian Residual L%d\n(%dx%d)', numLevels, w, h), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                
                if numLevels + 1 <= 6
                    ax = obj.AxPyr_{numLevels+1};
                    set(ax, 'Visible', 'on');
                    I_rec_clamp = min(max(I_rec, 0), 1);
                    imshow(I_rec_clamp, 'Parent', ax);
                    [h, w, ~] = size(I_rec_clamp);
                    title(ax, sprintf('Reconstructed\n(%dx%d)', w, h), 'Color', obj.COLOR_ACCENT, 'FontSize', 9);
                end
                
                if numLevels + 2 <= 6
                    ax = obj.AxPyr_{numLevels+2};
                    set(ax, 'Visible', 'on');
                    I_err = abs(obj.StatePyr_Image_ - I_rec);
                    I_err_vis = I_err * 1e5;
                    I_err_vis = min(max(I_err_vis, 0), 1);
                    imshow(I_err_vis, 'Parent', ax);
                    [h, w, ~] = size(I_err_vis);
                    title(ax, sprintf('Error x10^5\n(%dx%d)', w, h), 'Color', obj.COLOR_SECONDARY, 'FontSize', 9);
                end
            elseif strcmp(obj.StatePyr_Mode_, 'MIP-Map')
                % MIP-map and EWA comparison render
                set(obj.AxPyr_{1}, 'Visible', 'on');
                imshow(obj.StatePyr_Image_, 'Parent', obj.AxPyr_{1});
                title(obj.AxPyr_{1}, 'Original Image (Scale 1.0)', 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                
                set(obj.AxPyr_{2}, 'Visible', 'on');
                imshow(I_finer, 'Parent', obj.AxPyr_{2});
                title(obj.AxPyr_{2}, sprintf('Finer Level %d (Aliases)', floor(d_level)+1), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                
                set(obj.AxPyr_{3}, 'Visible', 'on');
                imshow(I_coarser, 'Parent', obj.AxPyr_{3});
                title(obj.AxPyr_{3}, sprintf('Coarser Level %d (Blurry)', ceil(d_level)+1), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                
                set(obj.AxPyr_{4}, 'Visible', 'on');
                imshow(I_trilinear, 'Parent', obj.AxPyr_{4});
                title(obj.AxPyr_{4}, 'Trilinear MIP (Clean Uniform)', 'Color', obj.COLOR_ACCENT, 'FontSize', 9);
                
                set(obj.AxPyr_{5}, 'Visible', 'on');
                imshow(I_floor_mip, 'Parent', obj.AxPyr_{5});
                title(obj.AxPyr_{5}, 'Isotropic MIP (Blurry Horizon)', 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                
                set(obj.AxPyr_{6}, 'Visible', 'on');
                imshow(I_floor_ewa, 'Parent', obj.AxPyr_{6});
                title(obj.AxPyr_{6}, 'EWA Anisotropic (Sharp Horizon)', 'Color', obj.COLOR_ACCENT, 'FontSize', 9);
            else
                % Local Laplacian Detail Comparison View
                % Show original on Ax1
                imshow(obj.StatePyr_Image_, 'Parent', obj.AxPyr_{1});
                title(obj.AxPyr_{1}, 'Original Image', 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                set(obj.AxPyr_{1}, 'Visible', 'on');
                
                % Show filtered on Ax2
                imshow(I_rec, 'Parent', obj.AxPyr_{2});
                title(obj.AxPyr_{2}, 'Filtered (Detail Enh)', 'Color', obj.COLOR_ACCENT, 'FontSize', 9);
                set(obj.AxPyr_{2}, 'Visible', 'on');
                
                % Show input detail levels vs output detail levels on remaining axes
                for idx = 1:min(2, numLevels-1)
                    % Input Laplacian Level idx on Ax(2*idx + 1)
                    ax_in = obj.AxPyr_{2*idx + 1};
                    set(ax_in, 'Visible', 'on');
                    L_in = llf.InputLaplacianPyramid.Levels{idx};
                    L_in_vis = min(max(L_in * 4.0 + 0.5, 0), 1);
                    imshow(L_in_vis, 'Parent', ax_in);
                    [h_in, w_in, ~] = size(L_in);
                    title(ax_in, sprintf('Input LP Lvl %d\n(%dx%d)', idx, w_in, h_in), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
                    
                    % Output Laplacian Level idx on Ax(2*idx + 2)
                    ax_out = obj.AxPyr_{2*idx + 2};
                    set(ax_out, 'Visible', 'on');
                    L_out = llf.OutputLaplacianPyramid.Levels{idx};
                    L_out_vis = min(max(L_out * 4.0 + 0.5, 0), 1);
                    imshow(L_out_vis, 'Parent', ax_out);
                    [h_out, w_out, ~] = size(L_out);
                    title(ax_out, sprintf('Output LP Lvl %d\n(%dx%d)', idx, w_out, h_out), 'Color', obj.COLOR_ACCENT, 'FontSize', 9);
                end
            end
            
            % 5. Update right-hand text labels
            if strcmp(obj.StatePyr_Mode_, 'Gaussian')
                descText = sprintf([...
                    'Gaussian Pyramid:\n', ...
                    'This represents a sequence of low-pass filtered, decimated versions of the image.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Blur the image G_(i-1) with the smoothing filter kernel (low-pass filtering).\n', ...
                    '2. Decimate (downsample) the blurred image by a factor of 2 in both dimensions.\n\n', ...
                    'The decimation uses strided convolution to compute downsampled pixels directly from scratch.']);
            elseif strcmp(obj.StatePyr_Mode_, 'Laplacian')
                descText = sprintf([...
                    'Laplacian Pyramid:\n', ...
                    'This represents a sequence of band-pass (detail) bands. It stores the details lost during decimation.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Compute G_i by reducing G_(i-1).\n', ...
                    '2. Expand G_i back to the resolution of G_(i-1) using interpolation and upsampling.\n', ...
                    '3. L_(i-1) = G_(i-1) - expand(G_i).\n\n', ...
                    'Reconstruction recovers G_(i-1) by adding the details: G_(i-1) = L_(i-1) + expand(G_i).']);
            elseif strcmp(obj.StatePyr_Mode_, 'MIP-Map')
                descText = sprintf([...
                    'MIP-Map & Anisotropic Resampling:\n', ...
                    'Demonstrates uniform scale selection and perspective floor texture mapping.\n\n', ...
                    'Uniform Selection (0.35x):\n', ...
                    '- Finer level aliases due to insufficient pre-filtering.\n', ...
                    '- Coarser level blurs due to excessive pre-filtering.\n', ...
                    '- Trilinear blending averages both levels.\n\n', ...
                    'Perspective Floor (Anisotropic):\n', ...
                    '- Isotropic MIP filters on longest axis -> horizon blurs.\n', ...
                    '- EWA filters along the ellipse -> horizon stays sharp.']);
            else
                descText = sprintf([...
                    'Local Laplacian Filter:\n', ...
                    'An edge-preserving multiresolution filter for local contrast manipulation without halos.\n\n', ...
                    'Methodology:\n', ...
                    '1. Precomputes Laplacian pyramids for multiple discretized intensity bins.\n', ...
                    '2. Performs level-by-level coefficient interpolation based on local Gaussian pyramid values.\n', ...
                    '3. Reconstructs (collapses) the output pyramid to produce the final filtered image.\n\n', ...
                    '- Detail Enhancement: Alpha = 0.5\n', ...
                    '- Edge Control: Beta = 1.0\n', ...
                    '- Threshold: Sigma = 0.1\n', ...
                    '- Reference Bins: M = 8']);
            end
            obj.LblPyr_Info_Desc_.Text = descText;
            
            if strcmp(obj.StatePyr_Mode_, 'Gaussian')
                rec_status = 'N/A';
            elseif strcmp(obj.StatePyr_Mode_, 'Laplacian')
                rec_status = 'Perfect';
            else
                rec_status = 'N/A';
            end
            
            [kh, kw] = size(kernel.getMatrix());
            if strcmp(obj.StatePyr_Mode_, 'Local Laplacian')
                statsText = sprintf([...
                    'Filter Statistics:\n', ...
                    '----------------------------\n', ...
                    'Original Size: %d x %d\n', ...
                    'Pyramid Levels: %d\n', ...
                    'Intensity Bins: 8\n', ...
                    'Edge Threshold (Sigma): 0.10\n', ...
                    'Detail Scale (Alpha): 0.50\n', ...
                    'Edge Scale (Beta): 1.00\n', ...
                    'Luminance Processing: CIE L*a*b*'], ...
                    size(obj.StatePyr_Image_, 2), size(obj.StatePyr_Image_, 1), numLevels);
            elseif strcmp(obj.StatePyr_Mode_, 'MIP-Map')
                statsText = sprintf([...
                    'MipMap / EWA Statistics:\n', ...
                    '----------------------------\n', ...
                    'Original Size: %d x %d\n', ...
                    'Pyramid Levels: %d\n', ...
                    'MIP Level d: %.3f\n', ...
                    'Floor Level (Finer): %d\n', ...
                    'Ceil Level (Coarser): %d\n', ...
                    'EWA Radius Squared: 3.00\n', ...
                    'Jacobian: Analytical d(u,v)/d(x,y)'], ...
                    size(obj.StatePyr_Image_, 2), size(obj.StatePyr_Image_, 1), numLevels, ...
                    d_level, floor(d_level)+1, ceil(d_level)+1);
            else
                statsText = sprintf([...
                    'Pyramid Run Statistics:\n', ...
                    '----------------------------\n', ...
                    'Original Size: %d x %d\n', ...
                    'Levels Count: %d\n', ...
                    'Kernel Type: %s (%d x %d)\n', ...
                    'Boundary Mode: %s\n', ...
                    'Reconstruction: %s\n', ...
                    'Max Abs Error: %e'], ...
                    size(obj.StatePyr_Image_, 2), size(obj.StatePyr_Image_, 1), ...
                    numLevels, obj.StatePyr_Kernel_, kh, kw, ...
                    'replicate', rec_status, max_err);
            end
            obj.LblPyr_Info_Stats_.Text = statsText;
        end
        
        % ================================================================== %
        % Event Callbacks (Tab 5)
        % ================================================================== %
        function onPyrImageChanged_(obj)
            obj.StatePyr_ImageName_ = obj.DropdownPyr_Image_.Value;
            obj.StatePyr_Image_ = []; % trigger reload
            obj.updateTabPyr_();
        end
        
        function onPyrLoadCustomPressed_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files (*.jpg, *.png, *.bmp, *.tiff)'}, ...
                                     'Load Image');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            try
                img = imread(fullfile(path, file));
                obj.StatePyr_Image_ = double(img) / 255.0;
                obj.StatePyr_ImageName_ = file;
                obj.DropdownPyr_Image_.Items = [obj.DropdownPyr_Image_.Items, {file}];
                obj.DropdownPyr_Image_.Value = file;
                obj.updateTabPyr_();
            catch ME
                uialert(obj.Fig_, sprintf('Failed to load image:\n%s', ME.message), 'Error');
            end
        end
        
        function onPyrModeChanged_(obj)
            val = obj.DropdownPyr_Mode_.Value;
            if contains(val, 'Gaussian')
                obj.StatePyr_Mode_ = 'Gaussian';
            elseif contains(val, 'Laplacian')
                obj.StatePyr_Mode_ = 'Laplacian';
            elseif contains(val, 'Local Laplacian')
                obj.StatePyr_Mode_ = 'Local Laplacian';
            else
                obj.StatePyr_Mode_ = 'MIP-Map';
            end
            obj.updateTabPyr_();
        end
        
        function onPyrLevelsChanged_(obj)
            obj.StatePyr_Levels_ = round(obj.SliderPyr_Levels_.Value);
            obj.SliderPyr_Levels_.Value = obj.StatePyr_Levels_;
            obj.LblPyr_Levels_.Text = num2str(obj.StatePyr_Levels_);
            obj.updateTabPyr_();
        end
        
        function onPyrKernelChanged_(obj)
            obj.StatePyr_Kernel_ = obj.DropdownPyr_Kernel_.Value;
            obj.updateTabPyr_();
        end
        
        function onPyrKernelSizeChanged_(obj)
            val = round(obj.SliderPyr_KernelSize_.Value);
            if mod(val, 2) == 0
                if val > obj.SliderPyr_KernelSize_.Value
                    val = val - 1;
                else
                    val = val + 1;
                end
            end
            val = max(3, min(9, val));
            obj.StatePyr_KernelSize_ = val;
            obj.SliderPyr_KernelSize_.Value = val;
            obj.LblPyr_KernelSize_.Text = num2str(val);
            obj.updateTabPyr_();
        end
        
        function onPyrResetPressed_(obj)
            obj.StatePyr_ImageName_ = 'ocean_img.jpg';
            obj.StatePyr_Image_ = [];
            obj.StatePyr_Mode_ = 'Gaussian';
            obj.StatePyr_Levels_ = 4;
            obj.StatePyr_Kernel_ = 'Gaussian';
            obj.StatePyr_KernelSize_ = 5;
            obj.StatePyr_Bypass_ = false;
            
            obj.DropdownPyr_Image_.Value = 'ocean_img.jpg';
            obj.DropdownPyr_Mode_.Value = 'Gaussian Pyramid';
            obj.SliderPyr_Levels_.Value = 4;
            obj.LblPyr_Levels_.Text = '4';
            obj.DropdownPyr_Kernel_.Value = 'Gaussian';
            obj.SliderPyr_KernelSize_.Value = 5;
            obj.LblPyr_KernelSize_.Text = '5';
            obj.ChkPyr_Bypass_.Value = false;
            
            obj.updateTabPyr_();
        end
        
        function onPyrBypassChanged_(obj)
            obj.StatePyr_Bypass_ = obj.ChkPyr_Bypass_.Value;
            obj.updateTabPyr_();
        end
        
        function onPyrShowTheory_(obj)
            msg = sprintf(['IMAGE PYRAMIDS - Theory & History\n' ...
                '=========================================\n\n' ...
                'What is an Image Pyramid?\n' ...
                'An image pyramid is a multi-resolution representation of an image,\n' ...
                'stored as a collection of progressively downsampled images (levels).\n\n' ...
                'Gaussian Pyramid (Burt & Adelson, 1983):\n' ...
                '  Each level is produced by convolving the previous level with a\n' ...
                '  low-pass (blurring) kernel, then subsampling by 2 in both directions.\n' ...
                '  Critical insight: blurring BEFORE downsampling prevents aliasing.\n' ...
                '  Bypass Filter = direct subsampling, shows Moire artifacts.\n\n' ...
                'Laplacian Pyramid:\n' ...
                '  Stores the DIFFERENCE between successive Gaussian levels.\n' ...
                '  Each level captures a narrow frequency band of the image.\n' ...
                '  Supports PERFECT reconstruction: G_i = L_i + expand(G_{i+1}).\n\n' ...
                'Key Applications:\n' ...
                '  - Image compression  - Texture synthesis\n' ...
                '  - Seamless blending (Exercise 3.18)\n' ...
                '  - Object detection (HOG, SIFT scale-space)\n' ...
                '  - HDR exposure fusion\n\n' ...
                'Kernel Choice (Exercise 3.17):\n' ...
                '  2x2 Block: minimal averaging, heavy aliasing at higher levels.\n' ...
                '  Gaussian 5-tap: classic; good balance of sharpness vs aliasing.\n' ...
                '  Windowed Sinc (7-9 tap): near-ideal low-pass; sharpest pyramid.\n\n' ...
                'Reference: Szeliski "Computer Vision: Algorithms & Applications" Sec 3.5']);
            uialert(obj.Fig_, msg, 'Theory: Image Pyramids', 'Icon', 'info');
        end
        
        % ================================================================== %
        % Tab 6: Image Resampling (Szeliski Ex 3.16)
        % ================================================================== %
        function buildTabResample_(obj)
            % Sidebar
            obj.PanResampleControl_ = uipanel(obj.TabResample_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Canvas
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.PanResampleCanvas_ = uipanel(obj.TabResample_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, cW, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Info
            obj.PanResampleInfo_ = uipanel(obj.TabResample_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % Controls - Left panel
            yPos = 655;
            uilabel(obj.PanResampleControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Resampling Controls', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanResampleControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Image Selection
            yPos = yPos - 35;
            uilabel(obj.PanResampleControl_, 'Position', [15, yPos+2, 110, 20], 'Text', 'Test Image:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownResample_Image_ = uidropdown(obj.PanResampleControl_, ...
                'Position', [130, yPos, 105, 22], ...
                'Items', {'Synthetic Chirp', 'Natural Ocean'}, ...
                'Value', 'Synthetic Chirp', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onResampleImageChanged_());
                
            obj.BtnResample_LoadCustom_ = uibutton(obj.PanResampleControl_, ...
                'Position', [240, yPos, 50, 22], ...
                'Text', '...', ...
                'FontSize', 11, ...
                'FontWeight', 'bold', ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onResampleLoadCustomPressed_());
                
            % Original Shape readout label
            yPos = yPos - 22;
            obj.LblResample_OriginalShape_ = uilabel(obj.PanResampleControl_, ...
                'Position', [15, yPos, 280, 20], ...
                'Text', 'Original Shape: 256x256', ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 10);
                
            % Filter A
            yPos = yPos - 35;
            uilabel(obj.PanResampleControl_, 'Position', [15, yPos+2, 110, 20], 'Text', 'Filter A (Left):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownResample_FilterA_ = uidropdown(obj.PanResampleControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Nearest', 'Bilinear', 'Bicubic', 'Lanczos', 'WindowedSinc'}, ...
                'Value', 'Nearest', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onResampleFilterChanged_());
                
            % Filter B
            yPos = yPos - 35;
            uilabel(obj.PanResampleControl_, 'Position', [15, yPos+2, 110, 20], 'Text', 'Filter B (Right):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownResample_FilterB_ = uidropdown(obj.PanResampleControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Nearest', 'Bilinear', 'Bicubic', 'Lanczos', 'WindowedSinc'}, ...
                'Value', 'Lanczos', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onResampleFilterChanged_());
                
            % Scaling Factor
            yPos = yPos - 45;
            uilabel(obj.PanResampleControl_, 'Position', [15, yPos+18, 150, 20], 'Text', 'Scaling Factor:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblResample_Scale_ = uilabel(obj.PanResampleControl_, 'Position', [200, yPos+18, 70, 20], 'Text', '1.00x', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderResample_Scale_ = uislider(obj.PanResampleControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [0.1, 4.0], ...
                'Value', 1.0, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onResampleScaleSliderChanged_());
                
            % Frequency Analysis Checkbox
            yPos = yPos - 30;
            obj.ChkResample_FreqAnalysis_ = uicheckbox(obj.PanResampleControl_, ...
                'Position', [20, yPos, 250, 20], ...
                'Text', 'Show Frequency Spectrum (FFT)', ...
                'Value', false, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'ValueChangedFcn', @(~,~) obj.updateTabResample_());
                
            % Section Divider
            yPos = yPos - 20;
            uilabel(obj.PanResampleControl_, 'Position', [15, yPos, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Buttons
            yPos = yPos - 40;
            obj.BtnResample_Blink_ = uibutton(obj.PanResampleControl_, ...
                'Position', [20, yPos, 250, 26], ...
                'Text', 'Start Blink Compare', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onResampleBlinkPressed_());
                
            yPos = yPos - 35;
            obj.BtnResample_Animate_ = uibutton(obj.PanResampleControl_, ...
                'Position', [20, yPos, 250, 26], ...
                'Text', 'Start Zoom Animation', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onResampleAnimatePressed_());
                
            yPos = yPos - 35;
            obj.BtnResample_Reset_ = uibutton(obj.PanResampleControl_, ...
                'Position', [20, yPos, 250, 26], ...
                'Text', 'Reset Parameters', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onResampleResetPressed_());
                
            % ---------------- Canvas Panel ----------------
            % Three axes: Reference at the top, comparison at the bottom
            aW = floor((cW - 30) / 2);
            
            % Original Reference Image Axes (Top Centered)
            obj.AxResampleRef_ = uiaxes(obj.PanResampleCanvas_, ...
                'Position', [10, 360, cW - 20, 320], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxResampleRef_);
            
            % Filter A & B Comparison Axes (Bottom Side-by-Side)
            obj.AxResampleA_ = uiaxes(obj.PanResampleCanvas_, ...
                'Position', [10, 30, aW, 310], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxResampleA_);
            
            obj.AxResampleB_ = uiaxes(obj.PanResampleCanvas_, ...
                'Position', [10 + aW + 10, 30, aW, 310], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxResampleB_);
            
            % Setup axes
            % (updateTabResample_ will render the initial images on startup)
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            uilabel(obj.PanResampleInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Resampling Analysis', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanResampleInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 120;
            uilabel(obj.PanResampleInfo_, ...
                'Position', [15, iy, 230, 110], ...
                'Text', sprintf('Nearest Neighbor:\nZero-order interpolation. Fast but causes blocky structures and heavy aliasing.\n\nBilinear:\nFirst-order linear triangle filter. Smooths out noise but blurs high-frequencies.'), ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'WordWrap', 'on');
                
            iy = iy - 120;
            uilabel(obj.PanResampleInfo_, ...
                'Position', [15, iy, 230, 110], ...
                'Text', sprintf('Bicubic spline:\nThird-order Keys convolution spline. Achieves a good balance of sharpness and smooth reconstruction.\n\nLanczos (Windowed Sinc):\nSinc windowed by a wider sinc. The gold-standard low-pass reconstruction filter.'), ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'WordWrap', 'on');
                
            iy = iy - 120;
            uilabel(obj.PanResampleInfo_, ...
                'Position', [15, iy, 230, 110], ...
                'Text', sprintf('Nyquist-Shannon Theorem:\nTo prevent aliasing (Moire) during downsampling, we stretch the interpolation kernel width proportionally to 1/scale. This limits the bandwidth to prevent high-frequencies folding back as noise.'), ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
                
            % Show Theory button
            iy = iy - 55;
            obj.BtnResample_Theory_ = uibutton(obj.PanResampleInfo_, ...
                'Position', [15, iy, 230, 28], ...
                'Text', '[ ? ] Learn: Resampling Theory', ...
                'FontWeight', 'bold', ...
                'FontSize', 10, ...
                'BackgroundColor', [0.13, 0.25, 0.45], ...
                'FontColor', [0.50, 0.85, 1.00], ...
                'ButtonPushedFcn', @(~,~) obj.onResampleShowTheory_());
        end
        
        function updateTabResample_(obj)
            % 1. Load the active image if needed
            script_dir = fileparts(mfilename('fullpath'));
            project_root = fullfile(script_dir, '..', '..', '..');
            
            if isempty(obj.StateResample_Image_)
                if strcmp(obj.StateResample_ImageName_, 'Synthetic Chirp')
                    gridSize = 256;
                    [x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
                    r = sqrt(x.^2 + y.^2);
                    obj.StateResample_Image_ = 0.5 + 0.5 * sin(120 * r.^2);
                else
                    img_path = fullfile(project_root, 'Data', 'images', 'ocean_img.jpg');
                    if ~exist(img_path, 'file')
                        img_path = fullfile(script_dir, '..', '..', '..', 'Data', 'images', 'ocean_img.jpg');
                    end
                    if exist(img_path, 'file')
                        img = imread(img_path);
                        obj.StateResample_Image_ = double(img) / 255.0;
                    else
                        % Fallback to chirp
                        gridSize = 256;
                        [x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
                        r = sqrt(x.^2 + y.^2);
                        obj.StateResample_Image_ = 0.5 + 0.5 * sin(120 * r.^2);
                    end
                end
            end
            
            % 2. Calculate target size
            [h_orig, w_orig, c_orig] = size(obj.StateResample_Image_);
            h_new = round(h_orig * obj.StateResample_Scale_);
            w_new = round(w_orig * obj.StateResample_Scale_);
            
            % Clamp target size to at least 4x4
            h_new = max(4, h_new);
            w_new = max(4, w_new);
            
            % Update original shape label
            obj.LblResample_OriginalShape_.Text = sprintf('Original Shape: %d x %d', w_orig, h_orig);
            
            % 3. Instantiate resamplers and apply
            resA = OceanMath.Image.Resampling.ImageResampler('TargetSize', [h_new, w_new], 'FilterType', obj.StateResample_FilterA_);
            resB = OceanMath.Image.Resampling.ImageResampler('TargetSize', [h_new, w_new], 'FilterType', obj.StateResample_FilterB_);
            
            I_resA = resA.apply(obj.StateResample_Image_);
            I_resB = resB.apply(obj.StateResample_Image_);
            
            % 4. Display results
            if obj.ChkResample_FreqAnalysis_.Value
                % Compute Magnitude Spectrum
                getSpectrum = @(I) log(1.0 + abs(fftshift(fft2(double(mean(I, 3))))));
                
                F_Ref = getSpectrum(obj.StateResample_Image_);
                F_A = getSpectrum(I_resA);
                F_B = getSpectrum(I_resB);
                
                % Normalize for visualization
                normSpect = @(F) (F - min(F(:))) / max(1e-5, max(F(:)) - min(F(:)));
                
                % Grids and vectors for spatial frequencies
                u_ref = linspace(-0.5, 0.5, w_orig);
                v_ref = linspace(-0.5, 0.5, h_orig);
                u_new = linspace(-0.5, 0.5, w_new);
                v_new = linspace(-0.5, 0.5, h_new);
                
                % Original reference spectrum
                imagesc(obj.AxResampleRef_, u_ref, v_ref, normSpect(F_Ref));
                colormap(obj.AxResampleRef_, 'gray');
                axis(obj.AxResampleRef_, 'on');
                grid(obj.AxResampleRef_, 'on');
                set(obj.AxResampleRef_, 'YDir', 'normal', 'XColor', obj.COLOR_TEXT, 'YColor', obj.COLOR_TEXT);
                xlabel(obj.AxResampleRef_, 'u (cycles/px)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.AxResampleRef_, 'v (cycles/px)', 'Color', obj.COLOR_TEXT);
                title(obj.AxResampleRef_, sprintf('Original Spectrum (FFT) (%dx%d)', w_orig, h_orig), 'Color', obj.COLOR_TEXT);
                
                % Filter A spectrum
                if obj.StateResample_BlinkState_
                    imagesc(obj.AxResampleA_, u_new, v_new, normSpect(F_B));
                    title(obj.AxResampleA_, sprintf('Blink Spectrum: %s (%dx%d)', obj.StateResample_FilterB_, w_new, h_new), 'Color', obj.COLOR_ACCENT);
                else
                    imagesc(obj.AxResampleA_, u_new, v_new, normSpect(F_A));
                    title(obj.AxResampleA_, sprintf('Spectrum A: %s (%dx%d)', obj.StateResample_FilterA_, w_new, h_new), 'Color', obj.COLOR_TEXT);
                end
                colormap(obj.AxResampleA_, 'gray');
                axis(obj.AxResampleA_, 'on');
                grid(obj.AxResampleA_, 'on');
                set(obj.AxResampleA_, 'YDir', 'normal', 'XColor', obj.COLOR_TEXT, 'YColor', obj.COLOR_TEXT);
                xlabel(obj.AxResampleA_, 'u (cycles/px)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.AxResampleA_, 'v (cycles/px)', 'Color', obj.COLOR_TEXT);
                
                % Filter B spectrum
                imagesc(obj.AxResampleB_, u_new, v_new, normSpect(F_B));
                colormap(obj.AxResampleB_, 'gray');
                axis(obj.AxResampleB_, 'on');
                grid(obj.AxResampleB_, 'on');
                set(obj.AxResampleB_, 'YDir', 'normal', 'XColor', obj.COLOR_TEXT, 'YColor', obj.COLOR_TEXT);
                xlabel(obj.AxResampleB_, 'u (cycles/px)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.AxResampleB_, 'v (cycles/px)', 'Color', obj.COLOR_TEXT);
                title(obj.AxResampleB_, sprintf('Spectrum B: %s (%dx%d)', obj.StateResample_FilterB_, w_new, h_new), 'Color', obj.COLOR_TEXT);
            else
                % Display standard spatial domain images
                imagesc(obj.AxResampleRef_, 1:w_orig, 1:h_orig, obj.StateResample_Image_);
                axis(obj.AxResampleRef_, 'on', 'image');
                grid(obj.AxResampleRef_, 'off');
                set(obj.AxResampleRef_, 'YDir', 'reverse', 'XColor', obj.COLOR_TEXT, 'YColor', obj.COLOR_TEXT);
                xlabel(obj.AxResampleRef_, 'x (pixels)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.AxResampleRef_, 'y (pixels)', 'Color', obj.COLOR_TEXT);
                title(obj.AxResampleRef_, sprintf('Original Image (Reference) (%dx%d)', w_orig, h_orig), 'Color', obj.COLOR_TEXT);
                
                if obj.StateResample_BlinkState_
                    imagesc(obj.AxResampleA_, 1:w_new, 1:h_new, I_resB);
                    title(obj.AxResampleA_, sprintf('Blink: %s (%dx%d)', obj.StateResample_FilterB_, w_new, h_new), 'Color', obj.COLOR_ACCENT);
                else
                    imagesc(obj.AxResampleA_, 1:w_new, 1:h_new, I_resA);
                    title(obj.AxResampleA_, sprintf('Filter A: %s (%dx%d)', obj.StateResample_FilterA_, w_new, h_new), 'Color', obj.COLOR_TEXT);
                end
                axis(obj.AxResampleA_, 'on', 'image');
                grid(obj.AxResampleA_, 'off');
                set(obj.AxResampleA_, 'YDir', 'reverse', 'XColor', obj.COLOR_TEXT, 'YColor', obj.COLOR_TEXT);
                xlabel(obj.AxResampleA_, 'x (pixels)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.AxResampleA_, 'y (pixels)', 'Color', obj.COLOR_TEXT);
                
                imagesc(obj.AxResampleB_, 1:w_new, 1:h_new, I_resB);
                axis(obj.AxResampleB_, 'on', 'image');
                grid(obj.AxResampleB_, 'off');
                set(obj.AxResampleB_, 'YDir', 'reverse', 'XColor', obj.COLOR_TEXT, 'YColor', obj.COLOR_TEXT);
                xlabel(obj.AxResampleB_, 'x (pixels)', 'Color', obj.COLOR_TEXT);
                ylabel(obj.AxResampleB_, 'y (pixels)', 'Color', obj.COLOR_TEXT);
                title(obj.AxResampleB_, sprintf('Filter B: %s (%dx%d)', obj.StateResample_FilterB_, w_new, h_new), 'Color', obj.COLOR_TEXT);
                
                % Ensure grayscale color mapping if spatial image is 2D
                if ndims(obj.StateResample_Image_) == 2
                    colormap(obj.AxResampleRef_, 'gray');
                    colormap(obj.AxResampleA_, 'gray');
                    colormap(obj.AxResampleB_, 'gray');
                end
            end
        end
        
        function onResampleImageChanged_(obj)
            obj.StateResample_ImageName_ = obj.DropdownResample_Image_.Value;
            obj.StateResample_Image_ = []; % reload
            obj.updateTabResample_();
        end
        
        function onResampleLoadCustomPressed_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files (*.jpg, *.png, *.bmp, *.tiff)'}, ...
                                     'Load Resampling Test Image');
            if isequal(file, 0) || isequal(path, 0)
                return;
            end
            try
                img = imread(fullfile(path, file));
                obj.StateResample_Image_ = double(img) / 255.0;
                obj.StateResample_ImageName_ = file;
                
                % Add to dropdown items if not already present
                if ~any(strcmp(obj.DropdownResample_Image_.Items, file))
                    obj.DropdownResample_Image_.Items = [obj.DropdownResample_Image_.Items, {file}];
                end
                obj.DropdownResample_Image_.Value = file;
                
                obj.updateTabResample_();
            catch ME
                uialert(obj.Fig_, sprintf('Failed to load image:\n%s', ME.message), 'Error');
            end
        end
        
        function onResampleFilterChanged_(obj)
            obj.StateResample_FilterA_ = obj.DropdownResample_FilterA_.Value;
            obj.StateResample_FilterB_ = obj.DropdownResample_FilterB_.Value;
            obj.updateTabResample_();
        end
        
        function onResampleScaleSliderChanged_(obj)
            obj.StateResample_Scale_ = obj.SliderResample_Scale_.Value;
            obj.LblResample_Scale_.Text = sprintf('%.2fx', obj.StateResample_Scale_);
            obj.updateTabResample_();
        end
        
        function onResampleBlinkPressed_(obj)
            if ~isempty(obj.ResampleTimer_) && isvalid(obj.ResampleTimer_)
                stop(obj.ResampleTimer_);
                delete(obj.ResampleTimer_);
                obj.ResampleTimer_ = [];
                obj.StateResample_BlinkState_ = false;
                obj.BtnResample_Blink_.Text = 'Start Blink Compare';
                obj.BtnResample_Blink_.BackgroundColor = obj.COLOR_BTN;
            else
                if obj.StateResample_Animate_
                    obj.onResampleAnimatePressed_();
                end
                
                obj.BtnResample_Blink_.Text = 'Stop Blink Compare';
                obj.BtnResample_Blink_.BackgroundColor = obj.COLOR_SECONDARY;
                
                obj.ResampleTimer_ = timer('ExecutionMode', 'fixedRate', ...
                    'Period', 0.5, ...
                    'TimerFcn', @(~,~) obj.onResampleBlinkTick_());
                start(obj.ResampleTimer_);
            end
            obj.updateTabResample_();
        end
        
        function onResampleBlinkTick_(obj)
            obj.StateResample_BlinkState_ = ~obj.StateResample_BlinkState_;
            obj.updateTabResample_();
        end
        
        function onResampleAnimatePressed_(obj)
            if obj.StateResample_Animate_
                obj.StateResample_Animate_ = false;
                obj.BtnResample_Animate_.Text = 'Start Zoom Animation';
                obj.BtnResample_Animate_.BackgroundColor = obj.COLOR_BTN;
                if ~isempty(obj.ResampleTimer_) && isvalid(obj.ResampleTimer_)
                    stop(obj.ResampleTimer_);
                    delete(obj.ResampleTimer_);
                    obj.ResampleTimer_ = [];
                end
            else
                if ~isempty(obj.ResampleTimer_) && isvalid(obj.ResampleTimer_)
                    stop(obj.ResampleTimer_);
                    delete(obj.ResampleTimer_);
                    obj.ResampleTimer_ = [];
                    obj.StateResample_BlinkState_ = false;
                    obj.BtnResample_Blink_.Text = 'Start Blink Compare';
                    obj.BtnResample_Blink_.BackgroundColor = obj.COLOR_BTN;
                end
                
                obj.StateResample_Animate_ = true;
                obj.BtnResample_Animate_.Text = 'Stop Zoom Animation';
                obj.BtnResample_Animate_.BackgroundColor = obj.COLOR_SECONDARY;
                
                obj.ResampleTimer_ = timer('ExecutionMode', 'fixedRate', ...
                    'Period', 0.1, ...
                    'TimerFcn', @(~,~) obj.onResampleAnimateTick_());
                start(obj.ResampleTimer_);
            end
        end
        
        function onResampleAnimateTick_(obj)
            current_scale = obj.SliderResample_Scale_.Value;
            step_sz = 0.08 * obj.ResampleAnimDirection_;
            next_scale = current_scale + step_sz;
            
            if next_scale >= 3.5
                next_scale = 3.5;
                obj.ResampleAnimDirection_ = -1;
            elseif next_scale <= 0.15
                next_scale = 0.15;
                obj.ResampleAnimDirection_ = 1;
            end
            
            obj.SliderResample_Scale_.Value = next_scale;
            obj.StateResample_Scale_ = next_scale;
            obj.LblResample_Scale_.Text = sprintf('%.2fx', next_scale);
            obj.updateTabResample_();
        end
        
        function onResampleResetPressed_(obj)
            if ~isempty(obj.ResampleTimer_) && isvalid(obj.ResampleTimer_)
                stop(obj.ResampleTimer_);
                delete(obj.ResampleTimer_);
                obj.ResampleTimer_ = [];
            end
            
            obj.StateResample_ImageName_ = 'Synthetic Chirp';
            obj.StateResample_Image_ = [];
            obj.StateResample_FilterA_ = 'Nearest';
            obj.StateResample_FilterB_ = 'Lanczos';
            obj.StateResample_Scale_ = 1.0;
            obj.StateResample_BlinkState_ = false;
            obj.StateResample_Animate_ = false;
            obj.ResampleAnimDirection_ = 1;
            
            obj.DropdownResample_Image_.Value = 'Synthetic Chirp';
            obj.DropdownResample_FilterA_.Value = 'Nearest';
            obj.DropdownResample_FilterB_.Value = 'Lanczos';
            obj.SliderResample_Scale_.Value = 1.0;
            obj.LblResample_Scale_.Text = '1.00x';
            obj.ChkResample_FreqAnalysis_.Value = false;
            obj.BtnResample_Blink_.Text = 'Start Blink Compare';
            obj.BtnResample_Blink_.BackgroundColor = obj.COLOR_BTN;
            obj.BtnResample_Animate_.Text = 'Start Zoom Animation';
            obj.BtnResample_Animate_.BackgroundColor = obj.COLOR_BTN;
            
            obj.updateTabResample_();
        end
        
        function onResampleShowTheory_(obj)
            msg = sprintf(['IMAGE RESAMPLING - Theory\n' ...
                '=========================================\n\n' ...
                'What is Resampling?\n' ...
                'Changing the number of pixels in an image (scaling up = magnification,\n' ...
                'scaling down = minification) requires reconstructing the continuous signal\n' ...
                'from its samples, then re-sampling at new locations.\n\n' ...
                'Nearest Neighbor (Order 0):\n' ...
                '  Each output pixel copies the nearest input pixel.\n' ...
                '  Very fast, but produces blocky artifacts and sharp staircasing.\n\n' ...
                'Bilinear (Order 1):\n' ...
                '  Blends the 4 nearest pixels with area-weighted coefficients.\n' ...
                '  Smooth but blurry; kills high-frequency details.\n\n' ...
                'Bicubic - Keys Spline (Order 3):\n' ...
                '  Convolves with a cubic spline kernel (parameter a = -0.5).\n' ...
                '  Balances sharpness and smoothness; standard in most software.\n\n' ...
                'Lanczos (Windowed Sinc):\n' ...
                '  Approximates the ideal sinc low-pass filter using a sinc-shaped window.\n' ...
                '  Highest quality; can introduce slight ringing near sharp edges.\n\n' ...
                'Nyquist-Shannon Sampling Theorem:\n' ...
                '  To avoid aliasing when downsampling by factor s, the kernel must\n' ...
                '  cut off frequencies above 1/(2s) cycles/pixel.\n' ...
                '  We achieve this by stretching the kernel support by factor 1/s.\n\n' ...
                'Reference: Szeliski Sec 3.4 (Image Resampling and Reconstruction)']);
            uialert(obj.Fig_, msg, 'Theory: Image Resampling', 'Icon', 'info');
        end
        
        % ================================================================== %
        % Tab 7: Image Blending (Szeliski Ex 3.18)
        % ================================================================== %
        function buildTabBlend_(obj)
            % Sidebar
            obj.PanBlendControl_ = uipanel(obj.TabBlend_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
            
            % Canvas
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.PanBlendCanvas_ = uipanel(obj.TabBlend_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, cW, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
            
            % Info panel
            obj.PanBlendInfo_ = uipanel(obj.TabBlend_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
            
            % ---- Left controls ----
            yPos = 655;
            uilabel(obj.PanBlendControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Pyramid Blending (Ex 3.18)', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 14, ...
                'FontWeight', 'bold');
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Image A
            yPos = yPos - 35;
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos+2, 80, 20], 'Text', 'Image A:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownBlend_ImageA_ = uidropdown(obj.PanBlendControl_, ...
                'Position', [100, yPos, 115, 22], ...
                'Items', {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg'}, ...
                'Value', 'ocean_img.jpg', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onBlendImageAChanged_());
            obj.BtnBlend_LoadA_ = uibutton(obj.PanBlendControl_, ...
                'Position', [222, yPos, 66, 22], ...
                'Text', 'Load A...', ...
                'FontSize', 9, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onBlendLoadAPressed_());
            
            % Image B
            yPos = yPos - 30;
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos+2, 80, 20], 'Text', 'Image B:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownBlend_ImageB_ = uidropdown(obj.PanBlendControl_, ...
                'Position', [100, yPos, 115, 22], ...
                'Items', {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg'}, ...
                'Value', 'elon_musk.jpg', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onBlendImageBChanged_());
            obj.BtnBlend_LoadB_ = uibutton(obj.PanBlendControl_, ...
                'Position', [222, yPos, 66, 22], ...
                'Text', 'Load B...', ...
                'FontSize', 9, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onBlendLoadBPressed_());
            
            % Mask type
            yPos = yPos - 35;
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos+2, 80, 20], 'Text', 'Mask Type:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownBlend_MaskType_ = uidropdown(obj.PanBlendControl_, ...
                'Position', [100, yPos, 130, 22], ...
                'Items', {'Vertical Split', 'Horizontal Split', 'Center Circle', 'Feathered Split', 'Load Custom...'}, ...
                'Value', 'Feathered Split', ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onBlendMaskTypeChanged_());
            obj.BtnBlend_LoadMask_ = uibutton(obj.PanBlendControl_, ...
                'Position', [238, yPos, 55, 22], ...
                'Text', 'Load...', ...
                'FontSize', 9, ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onBlendLoadMaskPressed_());
            
            % Pyramid Levels
            yPos = yPos - 45;
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Pyramid Levels:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblBlend_Levels_ = uilabel(obj.PanBlendControl_, 'Position', [200, yPos+18, 70, 20], 'Text', num2str(obj.StateBlend_Levels_), 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderBlend_Levels_ = uislider(obj.PanBlendControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [2, 8], ...
                'Value', obj.StateBlend_Levels_, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onBlendLevelsChanged_());
            
            % Kernel type
            yPos = yPos - 45;
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Smoothing Kernel:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownBlend_Kernel_ = uidropdown(obj.PanBlendControl_, ...
                'Items', {'Gaussian', 'Triangle', 'Box', 'Windowed Sinc'}, ...
                'Value', obj.StateBlend_Kernel_, ...
                'Position', [15, yPos-10, 280, 28], ...
                'BackgroundColor', obj.COLOR_BTN, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onBlendKernelChanged_());
            
            % Kernel size
            yPos = yPos - 55;
            uilabel(obj.PanBlendControl_, 'Position', [15, yPos+18, 120, 20], 'Text', 'Kernel Size (Odd):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblBlend_KernelSize_ = uilabel(obj.PanBlendControl_, 'Position', [200, yPos+18, 70, 20], 'Text', num2str(obj.StateBlend_KernelSize_), 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderBlend_KernelSize_ = uislider(obj.PanBlendControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [3, 9], ...
                'Value', obj.StateBlend_KernelSize_, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onBlendKernelSizeChanged_());
            
            % Bypass filter checkbox
            yPos = yPos - 35;
            obj.ChkBlend_Bypass_ = uicheckbox(obj.PanBlendControl_, ...
                'Position', [20, yPos, 250, 22], ...
                'Text', 'Bypass Low-Pass Filter (Show Aliasing)', ...
                'Value', false, ...
                'FontColor', [1.0, 0.75, 0.20], ...
                'FontSize', 10, ...
                'ValueChangedFcn', @(~,~) obj.onBlendBypassChanged_());
            
            % Reset button
            yPos = yPos - 42;
            obj.BtnBlend_Reset_ = uibutton(obj.PanBlendControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', 'Reset Parameters', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onBlendResetPressed_());
            
            % Theory button
            yPos = yPos - 38;
            obj.BtnBlend_Theory_ = uibutton(obj.PanBlendControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', '[ ? ] Learn: Pyramid Blending', ...
                'FontWeight', 'bold', ...
                'FontSize', 10, ...
                'BackgroundColor', [0.13, 0.25, 0.45], ...
                'FontColor', [0.50, 0.85, 1.00], ...
                'ButtonPushedFcn', @(~,~) obj.onBlendShowTheory_());
            
            % ---- Canvas: 2 rows x 3 cols of axes ----
            axW = floor((cW - 40) / 3);
            axH = floor((680 - 40) / 2);
            gap = 10;
            
            % Top row: Image A, Image B, Mask
            obj.AxBlendA_ = uiaxes(obj.PanBlendCanvas_, ...
                'Position', [10, axH + 30, axW, axH], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxBlendA_);
            disableDefaultInteractivity(obj.AxBlendA_);
            
            obj.AxBlendB_ = uiaxes(obj.PanBlendCanvas_, ...
                'Position', [10 + axW + gap, axH + 30, axW, axH], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxBlendB_);
            disableDefaultInteractivity(obj.AxBlendB_);
            
            obj.AxBlendMask_ = uiaxes(obj.PanBlendCanvas_, ...
                'Position', [10 + 2*(axW + gap), axH + 30, axW, axH], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxBlendMask_);
            disableDefaultInteractivity(obj.AxBlendMask_);
            
            % Bottom row: Linear Blend, Pyramid Blend
            obj.AxBlendLinear_ = uiaxes(obj.PanBlendCanvas_, ...
                'Position', [10, 15, axW, axH], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxBlendLinear_);
            disableDefaultInteractivity(obj.AxBlendLinear_);
            
            obj.AxBlendPyramid_ = uiaxes(obj.PanBlendCanvas_, ...
                'Position', [10 + axW + gap, 15, axW, axH], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxBlendPyramid_);
            disableDefaultInteractivity(obj.AxBlendPyramid_);
            
            % ---- Right info panel ----
            iy = 650;
            uilabel(obj.PanBlendInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Blending Statistics', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanBlendInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 200;
            uilabel(obj.PanBlendInfo_, ...
                'Position', [15, iy, 230, 190], ...
                'Text', sprintf(['Algorithm (Burt & Adelson 1983):\n\n' ...
                    '1. Build Laplacian pyramid LP_A, LP_B.\n\n' ...
                    '2. Build Gaussian pyramid GP_W for mask.\n\n' ...
                    '3. At each level l:\n   B_l = LP_A_l*GP_W_l + LP_B_l*(1-GP_W_l)\n\n' ...
                    '4. Reconstruct blended image from B_l.\n\n' ...
                    'Key insight: high-freq details blend over\n' ...
                    'a narrow zone; low-freq content blends\n' ...
                    'over a wide zone. Matches human vision.']), ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
            
            iy = iy - 130;
            obj.LblBlend_Info_Stats_ = uilabel(obj.PanBlendInfo_, ...
                'Position', [15, iy, 230, 120], ...
                'Text', 'Run blend to see statistics...', ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 9, ...
                'WordWrap', 'on');
        end
        
        function updateTabBlend_(obj)
            % Load images
            script_dir = fileparts(mfilename('fullpath'));
            project_root = fullfile(script_dir, '..', '..', '..');
            img_dir = fullfile(project_root, 'Data', 'images');
            
            if isempty(obj.StateBlend_ImageA_)
                try
                    obj.StateBlend_ImageA_ = double(imread(fullfile(img_dir, obj.StateBlend_ImageNameA_))) / 255.0;
                catch
                    obj.StateBlend_ImageA_ = 0.2 * ones(256, 256, 3);
                end
            end
            
            if isempty(obj.StateBlend_ImageB_)
                try
                    obj.StateBlend_ImageB_ = double(imread(fullfile(img_dir, obj.StateBlend_ImageNameB_))) / 255.0;
                catch
                    obj.StateBlend_ImageB_ = 0.5 * ones(256, 256, 3);
                end
            end
            
            % Resize B to match A size
            [hA, wA, cA] = size(obj.StateBlend_ImageA_);
            [hB, wB, ~] = size(obj.StateBlend_ImageB_);
            if hA ~= hB || wA ~= wB
                resB = OceanMath.Image.Resampling.ImageResampler('TargetSize', [hA, wA], 'FilterType', 'Bilinear');
                obj.StateBlend_ImageB_ = resB.apply(obj.StateBlend_ImageB_);
            end
            
            % Build or use existing mask
            if isempty(obj.StateBlend_Mask_)
                obj.StateBlend_Mask_ = obj.buildBlendMask_(obj.DropdownBlend_MaskType_.Value, hA, wA);
            end
            
            % Resolve kernel
            try
                if strcmp(obj.StateBlend_Kernel_, 'Windowed Sinc')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('sinc', 'Size', obj.StateBlend_KernelSize_);
                elseif strcmp(obj.StateBlend_Kernel_, 'Triangle')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('triangle', obj.StateBlend_KernelSize_);
                elseif strcmp(obj.StateBlend_Kernel_, 'Box')
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('box', obj.StateBlend_KernelSize_);
                else
                    kernel = OceanMath.Signal.Kernels.KernelFactory.create('gaussian', 'Size', obj.StateBlend_KernelSize_);
                end
            catch
                kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 5);
            end
            
            % Clamp levels to valid range
            max_levels = floor(log2(min(hA, wA)));
            nL = min(obj.StateBlend_Levels_, max_levels);
            
            % Build pyramid blend
            blender = OceanMath.Image.MultiResolution.PyramidBlender( ...
                'Kernel', kernel, ...
                'NumLevels', nL, ...
                'BypassFilter', obj.StateBlend_Bypass_);
            
            maskA = obj.StateBlend_Mask_;
            maskB = 1 - maskA;
            
            I_pyramid = blender.blend({obj.StateBlend_ImageA_, obj.StateBlend_ImageB_}, {maskA, maskB});
            I_pyramid = min(max(I_pyramid, 0), 1);
            
            % Linear (direct) alpha blend
            if size(maskA, 3) == 1 && cA > 1
                maskA3 = repmat(maskA, [1, 1, cA]);
                maskB3 = repmat(maskB, [1, 1, cA]);
            else
                maskA3 = maskA;
                maskB3 = maskB;
            end
            I_linear = obj.StateBlend_ImageA_ .* maskA3 + obj.StateBlend_ImageB_ .* maskB3;
            I_linear = min(max(I_linear, 0), 1);
            
            % Render top row: Image A, Image B, Mask
            imshow(min(max(obj.StateBlend_ImageA_, 0), 1), 'Parent', obj.AxBlendA_);
            title(obj.AxBlendA_, sprintf('Image A: %s', obj.StateBlend_ImageNameA_), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
            
            imshow(min(max(obj.StateBlend_ImageB_, 0), 1), 'Parent', obj.AxBlendB_);
            title(obj.AxBlendB_, sprintf('Image B: %s', obj.StateBlend_ImageNameB_), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
            
            imshow(maskA(:,:,1), 'Parent', obj.AxBlendMask_);
            title(obj.AxBlendMask_, sprintf('Mask: %s', obj.DropdownBlend_MaskType_.Value), 'Color', obj.COLOR_TEXT, 'FontSize', 9);
            
            % Render bottom row: Linear blend, Pyramid blend
            imshow(I_linear, 'Parent', obj.AxBlendLinear_);
            title(obj.AxBlendLinear_, 'Direct Alpha Blend (seam visible)', 'Color', obj.COLOR_SUBTEXT, 'FontSize', 9);
            
            imshow(I_pyramid, 'Parent', obj.AxBlendPyramid_);
            bypass_str = '';
            if obj.StateBlend_Bypass_; bypass_str = ' [BYPASS]'; end
            title(obj.AxBlendPyramid_, sprintf('Pyramid Blend (%d levels)%s', nL, bypass_str), 'Color', obj.COLOR_ACCENT, 'FontSize', 9);
            
            % Update stats
            diff_mse = mean((I_pyramid(:) - I_linear(:)).^2);
            diff_psnr = 10 * log10(1 / max(diff_mse, 1e-12));
            bypass_label = 'No (anti-aliased)';
            if obj.StateBlend_Bypass_; bypass_label = 'YES (shows aliasing)'; end
            obj.LblBlend_Info_Stats_.Text = sprintf(['Blend Statistics:\n' ...
                '----------------------------\n' ...
                'Image size: %d x %d\n' ...
                'Pyramid levels: %d\n' ...
                'Kernel: %s (%d tap)\n' ...
                'Bypass filter: %s\n' ...
                'PSNR (Pyr vs Lin): %.1f dB'], ...
                wA, hA, nL, obj.StateBlend_Kernel_, obj.StateBlend_KernelSize_, ...
                bypass_label, diff_psnr);
        end
        
        function M = buildBlendMask_(~, mask_type, H, W)
            % BUILDBLENDMASK_  Build a soft blend mask based on selected type.
            [x, ~] = meshgrid(linspace(0, 1, W), linspace(0, 1, H));
            [~, y] = meshgrid(linspace(0, 1, W), linspace(0, 1, H));
            
            switch mask_type
                case 'Vertical Split'
                    M = double(x < 0.5);
                case 'Horizontal Split'
                    M = double(y < 0.5);
                case 'Center Circle'
                    cx = 0.5; cy = 0.5; r = 0.35;
                    M = double(((x - cx).^2 + (y - cy).^2) < r^2);
                case 'Feathered Split'
                    % Smooth sigmoid feathering around x=0.5
                    M = 1 ./ (1 + exp(-30 * (0.5 - x)));
                otherwise
                    M = double(x < 0.5);
            end
        end
        
        % ---- Tab 7 callbacks ----
        function onBlendImageAChanged_(obj)
            obj.StateBlend_ImageNameA_ = obj.DropdownBlend_ImageA_.Value;
            obj.StateBlend_ImageA_ = [];
            obj.updateTabBlend_();
        end
        
        function onBlendImageBChanged_(obj)
            obj.StateBlend_ImageNameB_ = obj.DropdownBlend_ImageB_.Value;
            obj.StateBlend_ImageB_ = [];
            obj.updateTabBlend_();
        end
        
        function onBlendMaskTypeChanged_(obj)
            val = obj.DropdownBlend_MaskType_.Value;
            if strcmp(val, 'Load Custom...')
                obj.onBlendLoadMaskPressed_();
            else
                obj.StateBlend_Mask_ = [];
                obj.updateTabBlend_();
            end
        end
        
        function onBlendLoadAPressed_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files'}, 'Load Image A');
            if isequal(file, 0); return; end
            try
                obj.StateBlend_ImageA_ = double(imread(fullfile(path, file))) / 255.0;
                obj.StateBlend_ImageNameA_ = file;
                if ~any(strcmp(obj.DropdownBlend_ImageA_.Items, file))
                    obj.DropdownBlend_ImageA_.Items = [obj.DropdownBlend_ImageA_.Items, {file}];
                end
                obj.DropdownBlend_ImageA_.Value = file;
                obj.updateTabBlend_();
            catch ME
                uialert(obj.Fig_, ME.message, 'Load Error');
            end
        end
        
        function onBlendLoadBPressed_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files'}, 'Load Image B');
            if isequal(file, 0); return; end
            try
                obj.StateBlend_ImageB_ = double(imread(fullfile(path, file))) / 255.0;
                obj.StateBlend_ImageNameB_ = file;
                if ~any(strcmp(obj.DropdownBlend_ImageB_.Items, file))
                    obj.DropdownBlend_ImageB_.Items = [obj.DropdownBlend_ImageB_.Items, {file}];
                end
                obj.DropdownBlend_ImageB_.Value = file;
                obj.updateTabBlend_();
            catch ME
                uialert(obj.Fig_, ME.message, 'Load Error');
            end
        end
        
        function onBlendLoadMaskPressed_(obj)
            [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files'}, 'Load Mask');
            if isequal(file, 0); return; end
            try
                M = double(imread(fullfile(path, file))) / 255.0;
                if size(M, 3) > 1; M = M(:,:,1); end
                obj.StateBlend_Mask_ = M;
                obj.updateTabBlend_();
            catch ME
                uialert(obj.Fig_, ME.message, 'Load Error');
            end
        end
        
        function onBlendLevelsChanged_(obj)
            obj.StateBlend_Levels_ = round(obj.SliderBlend_Levels_.Value);
            obj.SliderBlend_Levels_.Value = obj.StateBlend_Levels_;
            obj.LblBlend_Levels_.Text = num2str(obj.StateBlend_Levels_);
            obj.updateTabBlend_();
        end
        
        function onBlendKernelChanged_(obj)
            obj.StateBlend_Kernel_ = obj.DropdownBlend_Kernel_.Value;
            obj.updateTabBlend_();
        end
        
        function onBlendKernelSizeChanged_(obj)
            val = round(obj.SliderBlend_KernelSize_.Value);
            if mod(val, 2) == 0; val = val + 1; end
            val = max(3, min(9, val));
            obj.StateBlend_KernelSize_ = val;
            obj.SliderBlend_KernelSize_.Value = val;
            obj.LblBlend_KernelSize_.Text = num2str(val);
            obj.updateTabBlend_();
        end
        
        function onBlendBypassChanged_(obj)
            obj.StateBlend_Bypass_ = obj.ChkBlend_Bypass_.Value;
            obj.updateTabBlend_();
        end
        
        function onBlendResetPressed_(obj)
            obj.StateBlend_ImageA_ = [];
            obj.StateBlend_ImageB_ = [];
            obj.StateBlend_Mask_ = [];
            obj.StateBlend_ImageNameA_ = 'ocean_img.jpg';
            obj.StateBlend_ImageNameB_ = 'elon_musk.jpg';
            obj.StateBlend_Levels_ = 6;
            obj.StateBlend_Kernel_ = 'Gaussian';
            obj.StateBlend_KernelSize_ = 5;
            obj.StateBlend_Bypass_ = false;
            
            obj.DropdownBlend_ImageA_.Value = 'ocean_img.jpg';
            obj.DropdownBlend_ImageB_.Value = 'elon_musk.jpg';
            obj.DropdownBlend_MaskType_.Value = 'Feathered Split';
            obj.SliderBlend_Levels_.Value = 6;
            obj.LblBlend_Levels_.Text = '6';
            obj.DropdownBlend_Kernel_.Value = 'Gaussian';
            obj.SliderBlend_KernelSize_.Value = 5;
            obj.LblBlend_KernelSize_.Text = '5';
            obj.ChkBlend_Bypass_.Value = false;
            obj.updateTabBlend_();
        end
        
        function onBlendShowTheory_(obj)
            msg = sprintf(['PYRAMID BLENDING - Theory (Ex 3.18)\n' ...
                '=========================================\n\n' ...
                'Motivation:\n' ...
                'Direct alpha blending creates an abrupt edge or a blurry band\n' ...
                'depending on mask sharpness. Pyramid blending solves this by applying\n' ...
                'a DIFFERENT transition width at each spatial scale.\n\n' ...
                'Algorithm (Burt & Adelson, 1983):\n' ...
                '1. Build Laplacian pyramids LP_A and LP_B for each image.\n' ...
                '2. Build Gaussian pyramid GP_W for the blending mask W.\n' ...
                '3. At each level l: B_l = LP_A_l*GP_W_l + LP_B_l*(1-GP_W_l)\n' ...
                '4. Reconstruct the final blend from B_l (Laplacian collapse).\n\n' ...
                'Why it works:\n' ...
                '  At fine levels (high freq, level 1): mask transition is SHARP.\n' ...
                '  At coarse levels (low freq, level N): mask is blurred heavily.\n' ...
                '  The gradual mask change matches the spatial frequency band.\n' ...
                '  Result: seamless transition with no visible seam.\n\n' ...
                'Generalization to N images:\n' ...
                '  B_l = sum_i(LP_i_l * GP_Wi_l), optionally renormalized.\n' ...
                '  Value 0 reserves "no image" regions (label map extension).\n' ...
                '  Applications: HDR fusion, panorama stitching, medical imaging.\n\n' ...
                'Reference: Burt & Adelson 1983; Szeliski Sec 3.5.2']);
            uialert(obj.Fig_, msg, 'Theory: Pyramid Blending', 'Icon', 'info');
        end
        
        % ================================================================== %
        % Tab 8: Local Warping (Szeliski Ex 3.23)
        % ================================================================== %
        function buildTabLocalWarp_(obj)
            % Sidebar
            obj.PanLocalWarpControl_ = uipanel(obj.TabLocalWarp_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', ...
                'Scrollable', 'on');
                
            % Canvas
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.PanLocalWarpCanvas_ = uipanel(obj.TabLocalWarp_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, cW, 690], ...
                'BackgroundColor', obj.COLOR_BG, ...
                'BorderType', 'none');
                
            % Info
            obj.PanLocalWarpInfo_ = uipanel(obj.TabLocalWarp_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none');
                
            % Controls - Left panel
            yPos = 655;
            uilabel(obj.PanLocalWarpControl_, ...
                'Position', [15, yPos, 280, 25], ...
                'Text', 'Local Warping Controls', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanLocalWarpControl_, 'Position', [15, yPos-5, 280, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Image Selection
            yPos = yPos - 35;
            uilabel(obj.PanLocalWarpControl_, 'Position', [15, yPos+2, 110, 20], 'Text', 'Test Image:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownLocalWarp_Image_ = uidropdown(obj.PanLocalWarpControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Synthetic Grid', 'ocean_img.jpg', 'elon_musk.jpg'}, ...
                'Value', obj.StateLocalWarp_ImageName_, ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onLocalWarpImageChanged_());
                
            % Warp Mode
            yPos = yPos - 35;
            uilabel(obj.PanLocalWarpControl_, 'Position', [15, yPos+2, 110, 20], 'Text', 'Warp Mode:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownLocalWarp_Mode_ = uidropdown(obj.PanLocalWarpControl_, ...
                'Position', [130, yPos, 160, 22], ...
                'Items', {'Piecewise Affine', 'Thin Plate Spline', 'Gaussian RBF', 'Multiquadric RBF', 'Inverse Distance RBF', 'Beier-Neely (Lines)', 'B-Spline Grid', 'Paintbrush (Flow)'}, ...
                'Value', obj.StateLocalWarp_Mode_, ...
                'BackgroundColor', obj.COLOR_BG, ...
                'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onLocalWarpModeChanged_());
                
            % Grid Resolution Slider
            yPos = yPos - 45;
            obj.SliderLocalWarp_Grid_ = uislider(obj.PanLocalWarpControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [3, 8], ...
                'Value', 4, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onLocalWarpGridSliderChanged_());
            obj.LblLocalWarp_Grid_ = uilabel(obj.PanLocalWarpControl_, 'Position', [15, yPos+18, 270, 20], 'Text', 'Grid Resolution: 4x4', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
                
            % Show Grid Checkbox
            yPos = yPos - 30;
            obj.ChkLocalWarp_ShowGrid_ = uicheckbox(obj.PanLocalWarpControl_, ...
                'Position', [20, yPos, 250, 20], ...
                'Text', 'Show Deformable Mesh Overlay', ...
                'Value', true, ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'ValueChangedFcn', @(~,~) obj.onLocalWarpShowGridChanged_());
                
            % Paintbrush controls (Part 4)
            obj.SliderLocalWarp_BrushSize_ = uislider(obj.PanLocalWarpControl_, ...
                'Position', [25, yPos+40, 240, 3], ...
                'Limits', [10, 80], ...
                'Value', 30, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'Visible', 'off', ...
                'ValueChangedFcn', @(~,~) obj.onLocalWarpBrushSizeChanged_());
            obj.LblLocalWarp_BrushSize_ = uilabel(obj.PanLocalWarpControl_, 'Position', [15, yPos+48, 270, 20], ...
                'Text', 'Brush Size: 30 px', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11, 'Visible', 'off');
                
            obj.SliderLocalWarp_BrushStrength_ = uislider(obj.PanLocalWarpControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [0.1, 1.0], ...
                'Value', 0.5, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'Visible', 'off', ...
                'ValueChangedFcn', @(~,~) obj.onLocalWarpBrushStrengthChanged_());
            obj.LblLocalWarp_BrushStrength_ = uilabel(obj.PanLocalWarpControl_, 'Position', [15, yPos+18, 270, 20], ...
                'Text', 'Brush Strength: 0.50', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11, 'Visible', 'off');
                
            % Reset & Learn Buttons
            yPos = yPos - 50;
            obj.BtnLocalWarp_Reset_ = uibutton(obj.PanLocalWarpControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', 'Reset Control Grid', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1 1 1], ...
                'ButtonPushedFcn', @(~,~) obj.onLocalWarpResetPressed_());
                
            obj.BtnLocalWarp_ResetFlow_ = uibutton(obj.PanLocalWarpControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', 'Reset Flow Field', ...
                'FontWeight', 'bold', ...
                'FontSize', 11, ...
                'BackgroundColor', obj.COLOR_SECONDARY, ...
                'FontColor', [1 1 1], ...
                'Visible', 'off', ...
                'ButtonPushedFcn', @(~,~) obj.onLocalWarpResetFlowPressed_());
                
            yPos = yPos - 38;
            obj.BtnLocalWarp_Theory_ = uibutton(obj.PanLocalWarpControl_, ...
                'Position', [20, yPos, 250, 28], ...
                'Text', '[ ? ] Learn: Local Warping Theory', ...
                'FontWeight', 'bold', ...
                'FontSize', 10, ...
                'BackgroundColor', [0.13, 0.25, 0.45], ...
                'FontColor', [0.50, 0.85, 1.00], ...
                'ButtonPushedFcn', @(~,~) obj.onLocalWarpShowTheory_());
                
            % ---------------- Canvas Panel ----------------
            obj.AxLocalWarp_ = uiaxes(obj.PanLocalWarpCanvas_, ...
                'Position', [15, 15, cW - 30, 660], ...
                'Color', [0.10, 0.12, 0.16]);
            obj.VizCore_.styleAxes(obj.AxLocalWarp_);
            disableDefaultInteractivity(obj.AxLocalWarp_);
            
            % ---------------- Right Info Panel ----------------
            iy = 650;
            uilabel(obj.PanLocalWarpInfo_, ...
                'Position', [15, iy, 230, 25], ...
                'Text', 'Local Warping Math', ...
                'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            uilabel(obj.PanLocalWarpInfo_, 'Position', [15, iy-5, 230, 1], 'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            iy = iy - 190;
            obj.LblLocalWarp_Info_Desc_ = uilabel(obj.PanLocalWarpInfo_, ...
                'Position', [15, iy, 230, 180], ...
                'Text', '', ...
                'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 10, ...
                'WordWrap', 'on');
                
            iy = iy - 200;
            obj.LblLocalWarp_Info_Stats_ = uilabel(obj.PanLocalWarpInfo_, ...
                'Position', [15, iy, 230, 185], ...
                'Text', '', ...
                'FontColor', obj.COLOR_SUBTEXT, ...
                'FontSize', 10, ...
                'WordWrap', 'on');
        end
        
        function buildLocalWarpGeometry_(obj)
            res = obj.StateLocalWarp_GridRes_;
            if isempty(obj.StateLocalWarp_Image_)
                H = 256; W = 256;
            else
                [H, W, ~] = size(obj.StateLocalWarp_Image_);
            end
            
            % 1. Grid geometry (RBF, Affine, B-Spline)
            x_coords = linspace(1, W, res);
            y_coords = linspace(1, H, res);
            [X, Y] = meshgrid(x_coords, y_coords);
            obj.StateLocalWarp_PtsSrc_ = [X(:), Y(:)];
            obj.StateLocalWarp_PtsTgt_ = obj.StateLocalWarp_PtsSrc_;
            
            % 2. Beier-Neely Line geometry (Part 2)
            obj.StateLocalWarp_LinesSrc_P_ = [0.25*W, 0.45*H; 0.55*W, 0.25*H];
            obj.StateLocalWarp_LinesSrc_Q_ = [0.75*W, 0.45*H; 0.55*W, 0.75*H];
            obj.StateLocalWarp_LinesTgt_P_ = obj.StateLocalWarp_LinesSrc_P_;
            obj.StateLocalWarp_LinesTgt_Q_ = obj.StateLocalWarp_LinesSrc_Q_;
            
            % 3. Paintbrush flow fields (Part 4)
            obj.StateLocalWarp_Dx_ = zeros(H, W);
            obj.StateLocalWarp_Dy_ = zeros(H, W);
        end
        
        function redrawLocalWarpGrid_(obj)
            if ~isempty(obj.PlotLocalWarp_GridLines_)
                valid_lines = obj.PlotLocalWarp_GridLines_(isgraphics(obj.PlotLocalWarp_GridLines_));
                delete(valid_lines);
                obj.PlotLocalWarp_GridLines_ = [];
            end
            if ~isempty(obj.PlotLocalWarp_Handles_)
                valid_handles = obj.PlotLocalWarp_Handles_(isgraphics(obj.PlotLocalWarp_Handles_));
                delete(valid_handles);
                obj.PlotLocalWarp_Handles_ = [];
            end
            
            if ~obj.StateLocalWarp_ShowGrid_
                return;
            end
            
            res = obj.StateLocalWarp_GridRes_;
            pts = obj.StateLocalWarp_PtsTgt_;
            
            hold(obj.AxLocalWarp_, 'on');
            
            if strcmp(obj.StateLocalWarp_Mode_, 'Beier-Neely (Lines)')
                % Draw lines (Part 2)
                L = size(obj.StateLocalWarp_LinesSrc_P_, 1);
                for i = 1:L
                    P_src = obj.StateLocalWarp_LinesSrc_P_(i, :);
                    Q_src = obj.StateLocalWarp_LinesSrc_Q_(i, :);
                    P_tgt = obj.StateLocalWarp_LinesTgt_P_(i, :);
                    Q_tgt = obj.StateLocalWarp_LinesTgt_Q_(i, :);
                    
                    % 1. Draw source reference line (cyan dashed)
                    h_src_line = plot(obj.AxLocalWarp_, [P_src(1), Q_src(1)], [P_src(2), Q_src(2)], ...
                        '--', 'Color', [0.0, 0.8, 1.0], 'LineWidth', 1.0, 'HandleVisibility', 'off');
                    obj.PlotLocalWarp_GridLines_(end+1) = h_src_line;
                    
                    % 2. Draw target deformed line (orange solid)
                    h_tgt_line = plot(obj.AxLocalWarp_, [P_tgt(1), Q_tgt(1)], [P_tgt(2), Q_tgt(2)], ...
                        '-', 'Color', [1.0, 0.4, 0.0], 'LineWidth', 2.0, 'HandleVisibility', 'off');
                    obj.PlotLocalWarp_GridLines_(end+1) = h_tgt_line;
                    
                    % 3. Draw source endpoint markers (cyan crosses)
                    h_sp = plot(obj.AxLocalWarp_, P_src(1), P_src(2), 'x', 'Color', [0.0, 0.8, 1.0], 'MarkerSize', 6, 'LineWidth', 1.5);
                    h_sq = plot(obj.AxLocalWarp_, Q_src(1), Q_src(2), 'x', 'Color', [0.0, 0.8, 1.0], 'MarkerSize', 6, 'LineWidth', 1.5);
                    obj.PlotLocalWarp_Handles_(end+1) = h_sp;
                    obj.PlotLocalWarp_Handles_(end+1) = h_sq;
                    
                    % 4. Draw target endpoint markers (draggable, P = Red Circle, Q = Orange Square)
                    h_tp = plot(obj.AxLocalWarp_, P_tgt(1), P_tgt(2), 'o', 'MarkerSize', 8, ...
                        'MarkerFaceColor', [1.0, 0.2, 0.2], 'MarkerEdgeColor', [1 1 1], ...
                        'ButtonDownFcn', @(src,e) obj.onLocalWarpLineEndDown_(src, e, i, 'P'));
                    h_tq = plot(obj.AxLocalWarp_, Q_tgt(1), Q_tgt(2), 's', 'MarkerSize', 8, ...
                        'MarkerFaceColor', [1.0, 0.7, 0.0], 'MarkerEdgeColor', [1 1 1], ...
                        'ButtonDownFcn', @(src,e) obj.onLocalWarpLineEndDown_(src, e, i, 'Q'));
                    obj.PlotLocalWarp_Handles_(end+1) = h_tp;
                    obj.PlotLocalWarp_Handles_(end+1) = h_tq;
                end
            elseif strcmp(obj.StateLocalWarp_Mode_, 'Paintbrush (Flow)')
                % Paintbrush mode has no grid lines overlay
                return;
            else
                % Grid modes (Piecewise Affine, RBFs, B-Spline)
                if strcmp(obj.StateLocalWarp_Mode_, 'Piecewise Affine')
                    tri = delaunay(pts(:, 1), pts(:, 2));
                    obj.PlotLocalWarp_GridLines_ = triplot(tri, pts(:, 1), pts(:, 2), ...
                        'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.2, 'Parent', obj.AxLocalWarp_);
                else
                    % Draw horizontal and vertical grid lines
                    for r = 1:res
                        idx_row = (r-1)*res + (1:res);
                        obj.PlotLocalWarp_GridLines_(end+1) = plot(obj.AxLocalWarp_, pts(idx_row, 1), pts(idx_row, 2), ...
                            '-', 'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    end
                    for c = 1:res
                        idx_col = c:res:((res-1)*res + c);
                        obj.PlotLocalWarp_GridLines_(end+1) = plot(obj.AxLocalWarp_, pts(idx_col, 1), pts(idx_col, 2), ...
                            '-', 'Color', [1.0, 0.4, 0.0], 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    end
                end
                
                % Draw drag handles
                for idx = 1:size(pts, 1)
                    h = plot(obj.AxLocalWarp_, pts(idx, 1), pts(idx, 2), 'o', ...
                        'MarkerSize', 6, ...
                        'MarkerFaceColor', [1.0, 0.2, 0.2], ...
                        'MarkerEdgeColor', [1 1 1], ...
                        'ButtonDownFcn', @(src,e) obj.onLocalWarpPointDown_(src, e, idx));
                    obj.PlotLocalWarp_Handles_(end+1) = h;
                end
            end
        end
        
        function updateTabLocalWarp_(obj)
            if isempty(obj.StateLocalWarp_Image_)
                if strcmp(obj.StateLocalWarp_ImageName_, 'Synthetic Grid')
                    gridSize = 256;
                    [x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
                    obj.StateLocalWarp_Image_ = 0.5 + 0.5 * sign(sin(20 * pi * x) .* sign(sin(20 * pi * y)));
                else
                    script_dir = fileparts(mfilename('fullpath'));
                    project_root = fullfile(script_dir, '..', '..', '..');
                    image_path = fullfile(project_root, 'Data', 'images', obj.StateLocalWarp_ImageName_);
                    if exist(image_path, 'file')
                        img = imread(image_path);
                        obj.StateLocalWarp_Image_ = double(img) / 255.0;
                    else
                        gridSize = 256;
                        [x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
                        obj.StateLocalWarp_Image_ = 0.5 + 0.5 * sign(sin(20 * pi * x) .* sign(sin(20 * pi * y)));
                    end
                end
                obj.buildLocalWarpGeometry_();
            end
            
            I_src = obj.StateLocalWarp_Image_;
            [H, W, C] = size(I_src, 1:3);
            pts_src = obj.StateLocalWarp_PtsSrc_;
            pts_tgt = obj.StateLocalWarp_PtsTgt_;
            
            % Perform warping based on selected mode
            if strcmp(obj.StateLocalWarp_Mode_, 'Piecewise Affine')
                tri = delaunay(pts_src(:, 1), pts_src(:, 2));
                I_warp = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(I_src, pts_src, pts_tgt, tri);
            elseif strcmp(obj.StateLocalWarp_Mode_, 'Beier-Neely (Lines)')
                I_warp = OceanMath.Image.Resampling.BeierNeelyWarp.warp(I_src, ...
                    obj.StateLocalWarp_LinesTgt_P_, obj.StateLocalWarp_LinesTgt_Q_, ...
                    obj.StateLocalWarp_LinesSrc_P_, obj.StateLocalWarp_LinesSrc_Q_);
            elseif strcmp(obj.StateLocalWarp_Mode_, 'B-Spline Grid')
                res = obj.StateLocalWarp_GridRes_;
                I_warp = OceanMath.Image.Resampling.BSplineWarp.warp(I_src, pts_src, pts_tgt, [res, res]);
            elseif strcmp(obj.StateLocalWarp_Mode_, 'Paintbrush (Flow)')
                % Backward warp using dense Painted flow field StateLocalWarp_Dx_ / Dy_
                [X_grid, Y_grid] = meshgrid(1:W, 1:H);
                u_src = X_grid - obj.StateLocalWarp_Dx_;
                v_src = Y_grid - obj.StateLocalWarp_Dy_;
                
                u0 = floor(u_src);
                v0 = floor(v_src);
                u1 = u0 + 1;
                v1 = v0 + 1;
                
                u0_c = max(1, min(W, u0));
                u1_c = max(1, min(W, u1));
                v0_c = max(1, min(H, v0));
                v1_c = max(1, min(H, v1));
                
                du = u_src - u0;
                dv = v_src - v0;
                
                w00 = (1 - du) .* (1 - dv);
                w01 = du .* (1 - dv);
                w10 = (1 - du) .* dv;
                w11 = du .* dv;
                
                idx00 = v0_c + (u0_c - 1) * H;
                idx01 = v0_c + (u1_c - 1) * H;
                idx10 = v1_c + (u0_c - 1) * H;
                idx11 = v1_c + (u1_c - 1) * H;
                
                I_warp = zeros(H, W, C);
                for c = 1:C
                    Ic = I_src(:, :, c);
                    I_warp(:, :, c) = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                end
            else
                % RBF Interpolations
                if strcmp(obj.StateLocalWarp_Mode_, 'Thin Plate Spline')
                    kernel_type = 'thinplate';
                elseif strcmp(obj.StateLocalWarp_Mode_, 'Gaussian RBF')
                    kernel_type = 'gaussian';
                elseif strcmp(obj.StateLocalWarp_Mode_, 'Multiquadric RBF')
                    kernel_type = 'multiquadric';
                else
                    kernel_type = 'inversedistance';
                end
                tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, kernel_type);
                I_warp = tps.warpImage(I_src);
            end
            
            cla(obj.AxLocalWarp_);
            obj.PlotLocalWarp_ImgHandle_ = imagesc(obj.AxLocalWarp_, I_warp);
            axis(obj.AxLocalWarp_, 'image');
            obj.VizCore_.styleAxes(obj.AxLocalWarp_);
            
            % Setup Paintbrush interaction directly on the image CData (Part 4)
            if strcmp(obj.StateLocalWarp_Mode_, 'Paintbrush (Flow)')
                obj.PlotLocalWarp_ImgHandle_.ButtonDownFcn = @(src,e) obj.onLocalWarpPaintStart_(src, e);
            end
            
            obj.redrawLocalWarpGrid_();
            obj.updateControlVisibility_();
            obj.updateLocalWarpInfo_();
        end
        
        function onLocalWarpPointDown_(obj, src, event, pt_idx)
            obj.StateLocalWarp_ActivePointIdx_ = pt_idx;
            obj.Fig_.WindowButtonMotionFcn = @(~,e) obj.onLocalWarpPointDrag_(e);
            obj.Fig_.WindowButtonUpFcn = @(~,e) obj.onLocalWarpPointUp_(e);
        end
        
        function onLocalWarpPointDrag_(obj, event)
            if obj.StateLocalWarp_ActivePointIdx_ == -1, return; end
            cp = obj.AxLocalWarp_.CurrentPoint;
            x = cp(1,1);
            y = cp(1,2);
            [H, W, ~] = size(obj.StateLocalWarp_Image_);
            x = max(1, min(W, x));
            y = max(1, min(H, y));
            obj.StateLocalWarp_PtsTgt_(obj.StateLocalWarp_ActivePointIdx_, :) = [x, y];
            obj.redrawLocalWarpGrid_();
        end
        
        function onLocalWarpPointUp_(obj, event)
            obj.StateLocalWarp_ActivePointIdx_ = -1;
            obj.Fig_.WindowButtonMotionFcn = '';
            obj.Fig_.WindowButtonUpFcn = '';
            obj.updateTabLocalWarp_();
        end
        
        % Beier-Neely line endpoints drag callbacks (Part 2)
        function onLocalWarpLineEndDown_(obj, src, event, line_idx, end_type)
            obj.StateLocalWarp_ActiveLineEnd_ = struct('idx', line_idx, 'type', end_type);
            obj.Fig_.WindowButtonMotionFcn = @(~,e) obj.onLocalWarpLineEndDrag_(e);
            obj.Fig_.WindowButtonUpFcn = @(~,e) obj.onLocalWarpLineEndUp_(e);
        end
        
        function onLocalWarpLineEndDrag_(obj, event)
            if isempty(obj.StateLocalWarp_ActiveLineEnd_), return; end
            cp = obj.AxLocalWarp_.CurrentPoint;
            x = cp(1,1);
            y = cp(1,2);
            [H, W, ~] = size(obj.StateLocalWarp_Image_);
            x = max(1, min(W, x));
            y = max(1, min(H, y));
            
            idx = obj.StateLocalWarp_ActiveLineEnd_.idx;
            type = obj.StateLocalWarp_ActiveLineEnd_.type;
            
            if strcmp(type, 'P')
                obj.StateLocalWarp_LinesTgt_P_(idx, :) = [x, y];
            else
                obj.StateLocalWarp_LinesTgt_Q_(idx, :) = [x, y];
            end
            obj.redrawLocalWarpGrid_();
        end
        
        function onLocalWarpLineEndUp_(obj, event)
            obj.StateLocalWarp_ActiveLineEnd_ = [];
            obj.Fig_.WindowButtonMotionFcn = '';
            obj.Fig_.WindowButtonUpFcn = '';
            obj.updateTabLocalWarp_();
        end
        
        % Paintbrush paint callbacks (Part 4)
        function onLocalWarpPaintStart_(obj, src, event)
            obj.StateLocalWarp_PrevMousePos_ = obj.AxLocalWarp_.CurrentPoint(1, 1:2);
            obj.Fig_.WindowButtonMotionFcn = @(~,e) obj.onLocalWarpPaintDrag_(e);
            obj.Fig_.WindowButtonUpFcn = @(~,e) obj.onLocalWarpPaintEnd_(e);
        end
        
        function onLocalWarpPaintDrag_(obj, event)
            cp = obj.AxLocalWarp_.CurrentPoint(1, 1:2);
            v = cp - obj.StateLocalWarp_PrevMousePos_;
            obj.StateLocalWarp_PrevMousePos_ = cp;
            
            if norm(v) < 0.1, return; end
            
            [H, W, C] = size(obj.StateLocalWarp_Image_);
            
            % Accumulate paintbrush displacements
            [X_grid, Y_grid] = meshgrid(1:W, 1:H);
            dist2 = (X_grid - cp(1)).^2 + (Y_grid - cp(2)).^2;
            radius = obj.StateLocalWarp_BrushSize_;
            mask = dist2 <= radius^2;
            
            if ~any(mask(:)), return; end
            
            % Apply Gaussian falloff inside brush stroke footprint
            falloff = exp(-dist2(mask) / (2 * (radius/2.5)^2)) * obj.StateLocalWarp_BrushStrength_;
            obj.StateLocalWarp_Dx_(mask) = obj.StateLocalWarp_Dx_(mask) + v(1) * falloff;
            obj.StateLocalWarp_Dy_(mask) = obj.StateLocalWarp_Dy_(mask) + v(2) * falloff;
            
            % Compute fast backward bilinear warp
            u_src = X_grid - obj.StateLocalWarp_Dx_;
            v_src = Y_grid - obj.StateLocalWarp_Dy_;
            
            u0 = floor(u_src);
            v0 = floor(v_src);
            u1 = u0 + 1;
            v1 = v0 + 1;
            
            u0_c = max(1, min(W, u0));
            u1_c = max(1, min(W, u1));
            v0_c = max(1, min(H, v0));
            v1_c = max(1, min(H, v1));
            
            du = u_src - u0;
            dv = v_src - v0;
            
            w00 = (1 - du) .* (1 - dv);
            w01 = du .* (1 - dv);
            w10 = (1 - du) .* dv;
            w11 = du .* dv;
            
            idx00 = v0_c + (u0_c - 1) * H;
            idx01 = v0_c + (u1_c - 1) * H;
            idx10 = v1_c + (u0_c - 1) * H;
            idx11 = v1_c + (u1_c - 1) * H;
            
            I_src = obj.StateLocalWarp_Image_;
            I_warp = zeros(H, W, C);
            for c = 1:C
                Ic = I_src(:, :, c);
                I_warp(:, :, c) = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
            end
            
            % Update handle directly (ultra fluid 60fps)
            if ~isempty(obj.PlotLocalWarp_ImgHandle_) && isvalid(obj.PlotLocalWarp_ImgHandle_)
                obj.PlotLocalWarp_ImgHandle_.CData = I_warp;
            end
        end
        
        function onLocalWarpPaintEnd_(obj, event)
            obj.Fig_.WindowButtonMotionFcn = '';
            obj.Fig_.WindowButtonUpFcn = '';
            obj.updateTabLocalWarp_();
        end
        
        function onLocalWarpBrushSizeChanged_(obj)
            obj.StateLocalWarp_BrushSize_ = round(obj.SliderLocalWarp_BrushSize_.Value);
            obj.SliderLocalWarp_BrushSize_.Value = obj.StateLocalWarp_BrushSize_;
            obj.LblLocalWarp_BrushSize_.Text = sprintf('Brush Size: %d px', obj.StateLocalWarp_BrushSize_);
        end
        
        function onLocalWarpBrushStrengthChanged_(obj)
            obj.StateLocalWarp_BrushStrength_ = obj.SliderLocalWarp_BrushStrength_.Value;
            obj.LblLocalWarp_BrushStrength_.Text = sprintf('Brush Strength: %.2f', obj.StateLocalWarp_BrushStrength_);
        end
        
        function onLocalWarpResetFlowPressed_(obj)
            if ~isempty(obj.StateLocalWarp_Image_)
                [H, W, ~] = size(obj.StateLocalWarp_Image_);
                obj.StateLocalWarp_Dx_ = zeros(H, W);
                obj.StateLocalWarp_Dy_ = zeros(H, W);
            end
            obj.updateTabLocalWarp_();
        end
        
        function updateControlVisibility_(obj)
            is_paint = strcmp(obj.StateLocalWarp_Mode_, 'Paintbrush (Flow)');
            if is_paint
                vis_grid = 'off';
                vis_paint = 'on';
            else
                vis_grid = 'on';
                vis_paint = 'off';
            end
            
            if ~isempty(obj.SliderLocalWarp_Grid_)
                obj.SliderLocalWarp_Grid_.Visible = vis_grid;
                obj.LblLocalWarp_Grid_.Visible = vis_grid;
                obj.ChkLocalWarp_ShowGrid_.Visible = vis_grid;
                obj.BtnLocalWarp_Reset_.Visible = vis_grid;
                
                obj.SliderLocalWarp_BrushSize_.Visible = vis_paint;
                obj.LblLocalWarp_BrushSize_.Visible = vis_paint;
                obj.SliderLocalWarp_BrushStrength_.Visible = vis_paint;
                obj.LblLocalWarp_BrushStrength_.Visible = vis_paint;
                obj.BtnLocalWarp_ResetFlow_.Visible = vis_paint;
                
                if strcmp(obj.StateLocalWarp_Mode_, 'Beier-Neely (Lines)')
                    obj.BtnLocalWarp_Reset_.Text = 'Reset Control Lines';
                    obj.LblLocalWarp_Grid_.Visible = 'off';
                    obj.SliderLocalWarp_Grid_.Visible = 'off';
                else
                    obj.BtnLocalWarp_Reset_.Text = 'Reset Control Grid';
                end
            end
        end
        
        function onLocalWarpImageChanged_(obj)
            obj.StateLocalWarp_ImageName_ = obj.DropdownLocalWarp_Image_.Value;
            obj.StateLocalWarp_Image_ = [];
            obj.updateTabLocalWarp_();
        end
        
        function onLocalWarpModeChanged_(obj)
            obj.StateLocalWarp_Mode_ = obj.DropdownLocalWarp_Mode_.Value;
            obj.updateControlVisibility_();
            obj.updateTabLocalWarp_();
        end
        
        function onLocalWarpGridSliderChanged_(obj)
            obj.StateLocalWarp_GridRes_ = round(obj.SliderLocalWarp_Grid_.Value);
            obj.SliderLocalWarp_Grid_.Value = obj.StateLocalWarp_GridRes_;
            obj.LblLocalWarp_Grid_.Text = sprintf('Grid Resolution: %dx%d', obj.StateLocalWarp_GridRes_, obj.StateLocalWarp_GridRes_);
            obj.buildLocalWarpGeometry_();
            obj.updateTabLocalWarp_();
        end
        
        function onLocalWarpShowGridChanged_(obj)
            obj.StateLocalWarp_ShowGrid_ = obj.ChkLocalWarp_ShowGrid_.Value;
            obj.redrawLocalWarpGrid_();
        end
        
        function onLocalWarpResetPressed_(obj)
            if strcmp(obj.StateLocalWarp_Mode_, 'Beier-Neely (Lines)')
                if ~isempty(obj.StateLocalWarp_Image_)
                    [H, W, ~] = size(obj.StateLocalWarp_Image_);
                    obj.StateLocalWarp_LinesTgt_P_ = [0.25*W, 0.45*H; 0.55*W, 0.25*H];
                    obj.StateLocalWarp_LinesTgt_Q_ = [0.75*W, 0.45*H; 0.55*W, 0.75*H];
                end
            else
                obj.StateLocalWarp_PtsTgt_ = obj.StateLocalWarp_PtsSrc_;
            end
            obj.updateTabLocalWarp_();
        end
        
        function onLocalWarpShowTheory_(obj)
            msg = sprintf(['LOCAL IMAGE WARPING - Theory\n' ...
                '=========================================\n\n' ...
                'Mesh-Based Triangulation (Piecewise Affine):\n' ...
                '  - Divides the image into a set of triangles (mesh).\n' ...
                '  - Computes a separate affine transformation matrix inside each triangle.\n' ...
                '  - Uses barycentric coordinates to map target pixels back to the source image.\n' ...
                '  - C0 continuous (displacements are continuous across boundaries, but derivatives are not).\n\n' ...
                'Thin Plate Splines & RBFs (Variational Interpolants):\n' ...
                '  - Fits a smooth coordinate mapping function using radial basis functions.\n' ...
                '  - Solves a linear system L * W = Y where L includes distance kernels K and affine components P.\n' ...
                '  - Thin Plate Kernel: U(r) = r^2 * log(r) - minimizes physical bending energy.\n' ...
                '  - Gaussian RBF: U(r) = exp(-r^2 / s^2) - local influence.\n\n' ...
                'Beier-Neely Feature Morphing (Lines):\n' ...
                '  - Maps pixels using correspondences between directed line segments.\n' ...
                '  - Computes projection coordinate u and perpendicular distance v for each line.\n' ...
                '  - Aggregates weighted line displacements based on distance-decay parameter p.\n\n' ...
                'Cubic B-Spline FFD (Grid):\n' ...
                '  - Smooth local coordinates interpolation defined over a control grid.\n' ...
                '  - C2 continuous interpolation field using tensor-product cubic B-spline kernels.\n\n' ...
                'Paintbrush Flow (Paint):\n' ...
                '  - Interactive mouse stroke creates localized velocity forces.\n' ...
                '  - Displacements accumulate in a dense pixel-level flow field (Dx, Dy).\n\n' ...
                'Reference: Szeliski "Computer Vision" Sec 3.6']);
            uialert(obj.Fig_, msg, 'Theory: Local Warping', 'Icon', 'info');
        end
        
        function updateLocalWarpInfo_(obj)
            if strcmp(obj.StateLocalWarp_Mode_, 'Piecewise Affine')
                descText = sprintf([...
                    'Piecewise Affine Mesh Warp:\n', ...
                    'This method divides the image grid into triangular patches.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Perform Delaunay triangulation over control points.\n', ...
                    '2. For each triangle, compute the 3x3 affine mapping from source to target.\n', ...
                    '3. Map target pixels back to source using barycentric weights.']);
            elseif strcmp(obj.StateLocalWarp_Mode_, 'Beier-Neely (Lines)')
                descText = sprintf([...
                    'Beier-Neely Feature Morphing:\n', ...
                    'Geometric mapping using corresponding directed line segments.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Define directed lines in source (cyan) and target (orange).\n', ...
                    '2. Find projection parameters (u, v) for all pixels w.r.t target lines.\n', ...
                    '3. Map back to source using local coordinate system and decay weight.']);
            elseif strcmp(obj.StateLocalWarp_Mode_, 'B-Spline Grid')
                descText = sprintf([...
                    'Cubic B-Spline FFD:\n', ...
                    'Smooth local grid warping using C2-continuous cubic B-splines.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Set up a grid of control point displacements.\n', ...
                    '2. Interpolate dense pixel offsets using 2D tensor B-splines.\n', ...
                    '3. Backward map target pixels using cubic spline displacements.']);
            elseif strcmp(obj.StateLocalWarp_Mode_, 'Paintbrush (Flow)')
                descText = sprintf([...
                    'Paintbrush Flow Field:\n', ...
                    'Interactively paint dense vector displacement fields.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Click and drag directly on the canvas image to paint.\n', ...
                    '2. Mouse drag applies Gaussian forces in the stroke direction.\n', ...
                    '3. Backward warp using accumulated flow field (Dx, Dy).']);
            else
                descText = sprintf([...
                    'RBF Variational Interpolation:\n', ...
                    'Uses Radial Basis Functions to interpolate sparse control point displacements.\n\n', ...
                    'Mathematical Steps:\n', ...
                    '1. Compute distance matrix between all control points.\n', ...
                    '2. Fit TPS/RBF coefficients (weights W and affine matrix A).\n', ...
                    '3. Map target coordinates back to source by evaluating the radial basis kernel.']);
            end
            obj.LblLocalWarp_Info_Desc_.Text = descText;
            
            if strcmp(obj.StateLocalWarp_Mode_, 'Piecewise Affine')
                numTri = (obj.StateLocalWarp_GridRes_-1)^2 * 2;
            else
                numTri = 0;
            end
            
            statsText = sprintf([...
                'Warping Statistics:\n', ...
                '----------------------------\n', ...
                'Active Image: %s\n', ...
                'Control Points: %d\n', ...
                'Warp Method: %s\n', ...
                'Triangles: %d'], ...
                obj.StateLocalWarp_ImageName_, ...
                size(obj.StateLocalWarp_PtsSrc_, 1), ...
                obj.StateLocalWarp_Mode_, ...
                numTri);
            obj.LblLocalWarp_Info_Stats_.Text = statsText;
        end


        % ================================================================== %
        % Shared helper: load a named image from Data/images/
        % ================================================================== %
        function I_d = loadTabImage_(obj, name)
            script_dir   = fileparts(mfilename('fullpath'));
            project_root = fullfile(script_dir, '..', '..', '..');
            img_path     = fullfile(project_root, 'Data', 'images', name);
            try
                I_d = double(imread(img_path)) / 255.0;
            catch
                I_d = repmat(reshape([0.2, 0.25, 0.32], 1, 1, 3), 256, 256);
                warning('ComputerVisionSandbox:ImageNotFound', ...
                    'Could not load image: %s. Using placeholder.', name);
            end
        end
        
        % ================================================================== %
        % Tab 9: Feature Morphing (Ex 3.25) - Builder
        % ================================================================== %
        function buildTabMorph_(obj)
            % Sidebar
            obj.PanMorphControl_ = uipanel(obj.TabMorph_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', 'Scrollable', 'on');
            
            % Canvas
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.PanMorphCanvas_ = uipanel(obj.TabMorph_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, cW, 690], ...
                'BackgroundColor', obj.COLOR_BG, 'BorderType', 'none');
            
            % Info panel
            obj.PanMorphInfo_ = uipanel(obj.TabMorph_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, 'BorderType', 'none');
            
            % ---- Sidebar controls ----
            yPos = 655;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 25], ...
                'Text', 'Feature Morphing (Ex 3.25)', ...
                'FontColor', obj.COLOR_ACCENT, 'FontSize', 14, 'FontWeight', 'bold');
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos-5, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Image A row
            yPos = yPos - 38;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos+2, 60, 20], ...
                'Text', 'Image A:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownMorph_ImageA_ = uidropdown(obj.PanMorphControl_, ...
                'Position', [80, yPos, 120, 22], ...
                'Items', {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg'}, ...
                'Value', obj.StateMorph_ImageNameA_, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onMorphImageChanged_('A'));
            obj.BtnMorph_LoadA_ = uibutton(obj.PanMorphControl_, ...
                'Position', [208, yPos, 82, 22], 'Text', 'Load A...', 'FontSize', 9, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphLoadImage_('A'));
            
            % Image B row
            yPos = yPos - 30;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos+2, 60, 20], ...
                'Text', 'Image B:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownMorph_ImageB_ = uidropdown(obj.PanMorphControl_, ...
                'Position', [80, yPos, 120, 22], ...
                'Items', {'ocean_img.jpg', 'ocean_img_gray.jpg', 'elon_musk.jpg'}, ...
                'Value', obj.StateMorph_ImageNameB_, ...
                'BackgroundColor', obj.COLOR_BG, 'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onMorphImageChanged_('B'));
            obj.BtnMorph_LoadB_ = uibutton(obj.PanMorphControl_, ...
                'Position', [208, yPos, 82, 22], 'Text', 'Load B...', 'FontSize', 9, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphLoadImage_('B'));
            
            % Separator
            yPos = yPos - 20;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Warp mode
            yPos = yPos - 30;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos+2, 90, 20], ...
                'Text', 'Warp Mode:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownMorph_Mode_ = uidropdown(obj.PanMorphControl_, ...
                'Position', [15, yPos-20, 280, 28], ...
                'Items', {'Piecewise Affine', 'Thin Plate Spline', 'B-Spline Grid'}, ...
                'Value', obj.StateMorph_Mode_, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onMorphModeChanged_());
            
            % Grid resolution slider
            yPos = yPos - 60;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos+18, 130, 20], ...
                'Text', 'Grid Resolution:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblMorph_GridRes_ = uilabel(obj.PanMorphControl_, ...
                'Position', [200, yPos+18, 70, 20], ...
                'Text', sprintf('%d x %d', obj.StateMorph_GridRes_, obj.StateMorph_GridRes_), ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderMorph_GridRes_ = uislider(obj.PanMorphControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [2, 8], 'Value', obj.StateMorph_GridRes_, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onMorphGridResChanged_());
            
            % Show grid checkbox
            yPos = yPos - 28;
            obj.ChkMorph_ShowGrid_ = uicheckbox(obj.PanMorphControl_, ...
                'Position', [15, yPos, 160, 22], 'Text', 'Show Mesh/Grid', ...
                'FontColor', obj.COLOR_TEXT, 'FontSize', 11, ...
                'Value', obj.StateMorph_ShowGrid_, ...
                'ValueChangedFcn', @(~,~) obj.onMorphShowGridChanged_());
            
            % Separator
            yPos = yPos - 18;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Morph t slider
            yPos = yPos - 32;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos+18, 120, 20], ...
                'Text', 'Morph Parameter t:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblMorph_t_ = uilabel(obj.PanMorphControl_, ...
                'Position', [200, yPos+18, 70, 20], ...
                'Text', sprintf('%.2f', obj.StateMorph_t_), ...
                'FontColor', obj.COLOR_ACCENT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderMorph_t_ = uislider(obj.PanMorphControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [0, 1], 'Value', obj.StateMorph_t_, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onMorphTChanged_());
            
            % Play / Save GIF buttons
            yPos = yPos - 35;
            obj.BtnMorph_Play_ = uibutton(obj.PanMorphControl_, ...
                'Position', [15, yPos, 130, 26], 'Text', '▶  Play Animation', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_ACCENT, 'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onMorphPlayPressed_());
            obj.BtnMorph_SaveGif_ = uibutton(obj.PanMorphControl_, ...
                'Position', [155, yPos, 130, 26], 'Text', 'Save GIF', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphSaveGifPressed_());
            
            % Separator
            yPos = yPos - 20;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Correspondence section
            yPos = yPos - 25;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 20], ...
                'Text', 'Correspondence Management', ...
                'FontColor', obj.COLOR_SECONDARY, 'FontSize', 12, 'FontWeight', 'bold');
            
            yPos = yPos - 22;
            obj.LblMorph_PairCount_ = uilabel(obj.PanMorphControl_, ...
                'Position', [15, yPos, 280, 18], ...
                'Text', 'Control pairs: 0', 'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10);
            
            yPos = yPos - 32;
            obj.BtnMorph_AddPair_ = uibutton(obj.PanMorphControl_, ...
                'Position', [15, yPos, 82, 26], 'Text', '+ Add Pair', 'FontSize', 10, ...
                'BackgroundColor', [0.18, 0.55, 0.32], 'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onMorphAddPairPressed_());
            obj.BtnMorph_DeletePair_ = uibutton(obj.PanMorphControl_, ...
                'Position', [105, yPos, 88, 26], 'Text', '- Del Last', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphDeletePairPressed_());
            obj.BtnMorph_ClearPairs_ = uibutton(obj.PanMorphControl_, ...
                'Position', [201, yPos, 84, 26], 'Text', 'Clear All', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_SECONDARY, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphClearPairsPressed_());
            
            yPos = yPos - 32;
            obj.BtnMorph_ResetPts_ = uibutton(obj.PanMorphControl_, ...
                'Position', [15, yPos, 185, 26], 'Text', 'Reset to Uniform Grid', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphResetPtsPressed_());
            
            % Separator
            yPos = yPos - 20;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Rotation test section
            yPos = yPos - 25;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos, 280, 20], ...
                'Text', 'Part 2: Rotation via Morphing', ...
                'FontColor', obj.COLOR_ACCENT, 'FontSize', 12, 'FontWeight', 'bold');
            
            yPos = yPos - 40;
            uilabel(obj.PanMorphControl_, 'Position', [15, yPos+18, 130, 20], ...
                'Text', 'Rotation Angle (deg):', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.LblMorph_RotAngle_ = uilabel(obj.PanMorphControl_, ...
                'Position', [200, yPos+18, 70, 20], ...
                'Text', sprintf('%d deg', obj.StateMorph_RotAngle_), ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 10, 'HorizontalAlignment', 'right');
            obj.SliderMorph_RotAngle_ = uislider(obj.PanMorphControl_, ...
                'Position', [25, yPos+10, 240, 3], ...
                'Limits', [5, 180], 'Value', obj.StateMorph_RotAngle_, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'ValueChangedFcn', @(~,~) obj.onMorphRotAngleChanged_());
            
            yPos = yPos - 35;
            obj.BtnMorph_TestRotation_ = uibutton(obj.PanMorphControl_, ...
                'Position', [15, yPos, 205, 26], 'Text', 'Apply Rotation as Morph', 'FontSize', 10, ...
                'BackgroundColor', [0.40, 0.25, 0.65], 'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onMorphTestRotationPressed_());
            
            yPos = yPos - 35;
            obj.BtnMorph_Theory_ = uibutton(obj.PanMorphControl_, ...
                'Position', [15, yPos, 120, 26], 'Text', '?  Show Theory', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_SUBTEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onMorphShowTheory_());
            
            % ---- 3-panel canvas ----
            cW_px = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            halfW   = floor((cW_px - 12) / 2);
            topH    = 300;
            botH    = 345;
            botY    = 10;
            topY    = botY + botH + 15;
            
            % Image A axis (top-left)
            obj.AxMorph_A_ = axes(obj.PanMorphCanvas_, ...
                'Position', [8/cW_px, (topY+18)/690, (halfW-6)/cW_px, (topH-30)/690], ...
                'Color', obj.COLOR_BG, 'XColor', obj.COLOR_MUTED, 'YColor', obj.COLOR_MUTED, ...
                'Units', 'normalized', 'XTick', [], 'YTick', []);
            title(obj.AxMorph_A_, 'Image A  (source — drag handles)', ...
                'Color', obj.COLOR_TEXT, 'FontSize', 10);
            axis(obj.AxMorph_A_, 'image');
            
            % Image B axis (top-right)
            obj.AxMorph_B_ = axes(obj.PanMorphCanvas_, ...
                'Position', [(halfW+12)/cW_px, (topY+18)/690, (halfW-6)/cW_px, (topH-30)/690], ...
                'Color', obj.COLOR_BG, 'XColor', obj.COLOR_MUTED, 'YColor', obj.COLOR_MUTED, ...
                'Units', 'normalized', 'XTick', [], 'YTick', []);
            title(obj.AxMorph_B_, 'Image B  (target — drag handles)', ...
                'Color', obj.COLOR_TEXT, 'FontSize', 10);
            axis(obj.AxMorph_B_, 'image');
            
            % Result axis (bottom, full width)
            obj.AxMorph_Result_ = axes(obj.PanMorphCanvas_, ...
                'Position', [8/cW_px, (botY+18)/690, (cW_px-16)/cW_px, (botH-28)/690], ...
                'Color', obj.COLOR_BG, 'XColor', obj.COLOR_MUTED, 'YColor', obj.COLOR_MUTED, ...
                'Units', 'normalized', 'XTick', [], 'YTick', []);
            title(obj.AxMorph_Result_, 'Morphed Result  (t = 0.50)', ...
                'Color', obj.COLOR_ACCENT, 'FontSize', 11);
            axis(obj.AxMorph_Result_, 'image');
            
            % ---- Info panel ----
            iW = obj.INFO_W - 20;
            uilabel(obj.PanMorphInfo_, 'Position', [10, 655, iW, 22], ...
                'Text', 'Morphing Info', 'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 13, 'FontWeight', 'bold');
            uilabel(obj.PanMorphInfo_, 'Position', [10, 648, iW, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            obj.LblMorph_Info_Desc_ = uilabel(obj.PanMorphInfo_, ...
                'Position', [10, 360, iW, 280], 'Text', '', 'WordWrap', 'on', ...
                'FontColor', obj.COLOR_TEXT, 'FontSize', 9.5, 'VerticalAlignment', 'top');
            obj.LblMorph_Info_Stats_ = uilabel(obj.PanMorphInfo_, ...
                'Position', [10, 10, iW, 340], 'Text', '', 'WordWrap', 'on', ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 9, 'VerticalAlignment', 'top');
        end
        
        % ================================================================== %
        % Tab 9: Geometry initialization
        % ================================================================== %
        function buildMorphGeometry_(obj)
            obj.StateMorph_ImageA_ = obj.loadTabImage_(obj.StateMorph_ImageNameA_);
            obj.StateMorph_ImageB_ = obj.loadTabImage_(obj.StateMorph_ImageNameB_);
            [HA, WA, ~] = size(obj.StateMorph_ImageA_);
            if ~isequal(size(obj.StateMorph_ImageB_, [1, 2]), [HA, WA])
                obj.StateMorph_ImageB_ = imresize(obj.StateMorph_ImageB_, [HA, WA]);
            end
            [pts, ~] = OceanMath.Image.Resampling.ImageMorpher.uniformGridPoints( ...
                [HA, WA], obj.StateMorph_GridRes_);
            obj.StateMorph_PtsA_ = pts;
            obj.StateMorph_PtsB_ = pts;
            
            obj.PlotMorph_ImgA_ = imshow(obj.StateMorph_ImageA_, 'Parent', obj.AxMorph_A_);
            set(obj.PlotMorph_ImgA_, 'ButtonDownFcn', @(~,e) obj.onMorphCanvasClick_(e, 'A'));
            axis(obj.AxMorph_A_, 'image'); axis(obj.AxMorph_A_, 'off');
            
            obj.PlotMorph_ImgB_ = imshow(obj.StateMorph_ImageB_, 'Parent', obj.AxMorph_B_);
            set(obj.PlotMorph_ImgB_, 'ButtonDownFcn', @(~,e) obj.onMorphCanvasClick_(e, 'B'));
            axis(obj.AxMorph_B_, 'image'); axis(obj.AxMorph_B_, 'off');
            
            obj.PlotMorph_Result_ = imshow(obj.StateMorph_ImageA_, 'Parent', obj.AxMorph_Result_);
            axis(obj.AxMorph_Result_, 'image'); axis(obj.AxMorph_Result_, 'off');
            
            obj.LblMorph_PairCount_.Text = sprintf('Control pairs: %d', size(pts, 1));
            obj.redrawMorphGrid_('A');
            obj.redrawMorphGrid_('B');
            obj.updateMorphInfo_();
        end
        
        % ================================================================== %
        % Tab 9: Core morph update
        % ================================================================== %
        function updateTabMorph_(obj)
            if isempty(obj.StateMorph_PtsA_) || ...
               isempty(obj.StateMorph_ImageA_) || isempty(obj.StateMorph_ImageB_)
                return;
            end
            if size(obj.StateMorph_PtsA_, 1) < 3, return; end
            t = obj.StateMorph_t_;
            switch obj.StateMorph_Mode_
                case 'Piecewise Affine', wMode = 'PiecewiseAffine';
                case 'Thin Plate Spline', wMode = 'ThinPlateSpline';
                case 'B-Spline Grid', wMode = 'BSplineGrid';
                otherwise, wMode = 'PiecewiseAffine';
            end
            res = obj.StateMorph_GridRes_;
            try
                I_morph = OceanMath.Image.Resampling.ImageMorpher.morphFrame( ...
                    obj.StateMorph_ImageA_, obj.StateMorph_ImageB_, ...
                    obj.StateMorph_PtsA_,   obj.StateMorph_PtsB_, t, ...
                    'WarpMode', wMode, 'GridRes', [res, res]);
            catch ME
                warning('ImageMorpher:UpdateFailed', '%s', ME.message);
                return;
            end
            if isgraphics(obj.PlotMorph_Result_)
                set(obj.PlotMorph_Result_, 'CData', im2uint8(I_morph));
            else
                obj.PlotMorph_Result_ = imshow(I_morph, 'Parent', obj.AxMorph_Result_);
                axis(obj.AxMorph_Result_, 'image'); axis(obj.AxMorph_Result_, 'off');
            end
            title(obj.AxMorph_Result_, sprintf('Morphed Result  (t = %.2f)', t), ...
                'Color', obj.COLOR_ACCENT, 'FontSize', 11);
            obj.updateMorphInfo_();
            drawnow('limitrate');
        end
        
        % ================================================================== %
        % Tab 9: Draw control point handles on A or B panel
        % ================================================================== %
        function redrawMorphGrid_(obj, side)
            if strcmp(side, 'A')
                ax  = obj.AxMorph_A_;
                pts = obj.StateMorph_PtsA_;
                col = [0.20, 0.80, 1.00];  % cyan
            else
                ax  = obj.AxMorph_B_;
                pts = obj.StateMorph_PtsB_;
                col = [1.00, 0.55, 0.20];  % orange
            end
            hold(ax, 'on');
            % Clear old overlays
            if strcmp(side, 'A')
                if ~isempty(obj.PlotMorph_GridA_) && isgraphics(obj.PlotMorph_GridA_)
                    delete(obj.PlotMorph_GridA_); end
                if isgraphics(obj.PlotMorph_HandlesA_), delete(obj.PlotMorph_HandlesA_); end
                for k = 1:numel(obj.PlotMorph_LabelsA_)
                    if isgraphics(obj.PlotMorph_LabelsA_{k}), delete(obj.PlotMorph_LabelsA_{k}); end
                end
                obj.PlotMorph_GridA_ = [];
                obj.PlotMorph_LabelsA_ = {};
            else
                if ~isempty(obj.PlotMorph_GridB_) && isgraphics(obj.PlotMorph_GridB_)
                    delete(obj.PlotMorph_GridB_); end
                if isgraphics(obj.PlotMorph_HandlesB_), delete(obj.PlotMorph_HandlesB_); end
                for k = 1:numel(obj.PlotMorph_LabelsB_)
                    if isgraphics(obj.PlotMorph_LabelsB_{k}), delete(obj.PlotMorph_LabelsB_{k}); end
                end
                obj.PlotMorph_GridB_ = [];
                obj.PlotMorph_LabelsB_ = {};
            end
            if isempty(pts), hold(ax, 'off'); return; end
            N = size(pts, 1);
            % Draw mesh
            if obj.StateMorph_ShowGrid_ && N >= 3
                try
                    tri = delaunay(pts(:,1), pts(:,2));
                    mh  = triplot(tri, pts(:,1), pts(:,2), 'Parent', ax, ...
                        'Color', [col, 0.30], 'LineWidth', 0.7);
                    if strcmp(side, 'A'), obj.PlotMorph_GridA_ = mh;
                    else,                obj.PlotMorph_GridB_ = mh; end
                catch; end
            end
            % Draw handles (scatter)
            sh = scatter(ax, pts(:,1), pts(:,2), 36, col, 'filled', ...
                'MarkerEdgeColor', [1,1,1], 'LineWidth', 0.8, ...
                'ButtonDownFcn', @(src,e) obj.onMorphHandleDown_(src, e, side));
            % Numbered labels
            lbls = cell(N, 1);
            for k = 1:N
                lbls{k} = text(ax, pts(k,1)+4, pts(k,2)-4, num2str(k), ...
                    'Color', col, 'FontSize', 7, 'FontWeight', 'bold', ...
                    'HitTest', 'off', 'PickableParts', 'none');
            end
            if strcmp(side, 'A')
                obj.PlotMorph_HandlesA_ = sh;
                obj.PlotMorph_LabelsA_  = lbls;
            else
                obj.PlotMorph_HandlesB_ = sh;
                obj.PlotMorph_LabelsB_  = lbls;
            end
            hold(ax, 'off');
        end
        
        % ================================================================== %
        % Tab 9: Callbacks
        % ================================================================== %
        function onMorphImageChanged_(obj, side)
            if strcmp(side, 'A')
                obj.StateMorph_ImageNameA_ = obj.DropdownMorph_ImageA_.Value;
                obj.StateMorph_ImageA_ = obj.loadTabImage_(obj.StateMorph_ImageNameA_);
                [HA, WA, ~] = size(obj.StateMorph_ImageA_);
                if ~isequal(size(obj.StateMorph_ImageB_, [1, 2]), [HA, WA])
                    obj.StateMorph_ImageB_ = imresize(obj.StateMorph_ImageB_, [HA, WA]);
                end
                if isgraphics(obj.PlotMorph_ImgA_)
                    set(obj.PlotMorph_ImgA_, 'CData', im2uint8(obj.StateMorph_ImageA_));
                end
            else
                obj.StateMorph_ImageNameB_ = obj.DropdownMorph_ImageB_.Value;
                obj.StateMorph_ImageB_ = obj.loadTabImage_(obj.StateMorph_ImageNameB_);
                [HA, WA, ~] = size(obj.StateMorph_ImageA_);
                if ~isequal(size(obj.StateMorph_ImageB_, [1, 2]), [HA, WA])
                    obj.StateMorph_ImageB_ = imresize(obj.StateMorph_ImageB_, [HA, WA]);
                end
                if isgraphics(obj.PlotMorph_ImgB_)
                    set(obj.PlotMorph_ImgB_, 'CData', im2uint8(obj.StateMorph_ImageB_));
                end
            end
            obj.updateTabMorph_();
        end
        
        function onMorphLoadImage_(obj, side)
            [file, pth] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Images'});
            if isequal(file, 0), return; end
            I = imread(fullfile(pth, file));
            I_d = double(I); if max(I_d(:)) > 1.5, I_d = I_d / 255; end
            [HA, WA, ~] = size(obj.StateMorph_ImageA_);
            I_d = imresize(I_d, [HA, WA]);
            if strcmp(side, 'A')
                obj.StateMorph_ImageA_ = I_d;
                if isgraphics(obj.PlotMorph_ImgA_)
                    set(obj.PlotMorph_ImgA_, 'CData', im2uint8(I_d)); end
            else
                obj.StateMorph_ImageB_ = I_d;
                if isgraphics(obj.PlotMorph_ImgB_)
                    set(obj.PlotMorph_ImgB_, 'CData', im2uint8(I_d)); end
            end
            obj.updateTabMorph_();
        end
        
        function onMorphModeChanged_(obj)
            obj.StateMorph_Mode_ = obj.DropdownMorph_Mode_.Value;
            obj.updateTabMorph_();
        end
        
        function onMorphGridResChanged_(obj)
            obj.StateMorph_GridRes_ = round(obj.SliderMorph_GridRes_.Value);
            obj.LblMorph_GridRes_.Text = sprintf('%d x %d', ...
                obj.StateMorph_GridRes_, obj.StateMorph_GridRes_);
            obj.onMorphResetPtsPressed_();
        end
        
        function onMorphShowGridChanged_(obj)
            obj.StateMorph_ShowGrid_ = obj.ChkMorph_ShowGrid_.Value;
            obj.redrawMorphGrid_('A'); obj.redrawMorphGrid_('B');
        end
        
        function onMorphTChanged_(obj)
            obj.StateMorph_t_ = obj.SliderMorph_t_.Value;
            obj.LblMorph_t_.Text = sprintf('%.2f', obj.StateMorph_t_);
            obj.updateTabMorph_();
        end
        
        function onMorphAddPairPressed_(obj)
            obj.StateMorph_AddPairMode_ = true;
            obj.StateMorph_PendingPtA_  = [];
            obj.BtnMorph_AddPair_.BackgroundColor = [0.95, 0.60, 0.10];
            obj.BtnMorph_AddPair_.Text = '-> Click Image A...';
        end
        
        function onMorphCanvasClick_(obj, event, side)
            if ~obj.StateMorph_AddPairMode_, return; end
            pt = event.IntersectionPoint(1:2);
            if strcmp(side, 'A') && isempty(obj.StateMorph_PendingPtA_)
                obj.StateMorph_PendingPtA_ = pt;
                obj.BtnMorph_AddPair_.Text = '-> Click Image B...';
                hold(obj.AxMorph_A_, 'on');
                plot(obj.AxMorph_A_, pt(1), pt(2), 'o', 'MarkerSize', 8, ...
                    'MarkerFaceColor', [1,0.85,0], 'MarkerEdgeColor', [1,1,1], ...
                    'LineWidth', 1.2, 'HitTest', 'off');
                hold(obj.AxMorph_A_, 'off');
            elseif strcmp(side, 'B') && ~isempty(obj.StateMorph_PendingPtA_)
                obj.StateMorph_PtsA_ = [obj.StateMorph_PtsA_; obj.StateMorph_PendingPtA_];
                obj.StateMorph_PtsB_ = [obj.StateMorph_PtsB_; pt];
                obj.StateMorph_PendingPtA_ = [];
                obj.StateMorph_AddPairMode_ = false;
                obj.BtnMorph_AddPair_.BackgroundColor = [0.18, 0.55, 0.32];
                obj.BtnMorph_AddPair_.Text = '+ Add Pair';
                obj.LblMorph_PairCount_.Text = sprintf('Control pairs: %d', ...
                    size(obj.StateMorph_PtsA_, 1));
                obj.redrawMorphGrid_('A'); obj.redrawMorphGrid_('B');
                obj.updateTabMorph_();
            end
        end
        
        function onMorphHandleDown_(obj, ~, event, side)
            if obj.StateMorph_AddPairMode_, return; end
            pt = event.IntersectionPoint(1:2);
            if strcmp(side, 'A'), pts = obj.StateMorph_PtsA_;
            else,                  pts = obj.StateMorph_PtsB_; end
            if isempty(pts), return; end
            dists = sqrt((pts(:,1)-pt(1)).^2 + (pts(:,2)-pt(2)).^2);
            [~, idx] = min(dists);
            obj.StateMorph_ActivePtIdx_ = idx;
            obj.StateMorph_ActiveSide_  = side;
            obj.Fig_.WindowButtonMotionFcn = @(~,~) obj.onMorphHandleDrag_();
            obj.Fig_.WindowButtonUpFcn     = @(~,~) obj.onMorphHandleUp_();
        end
        
        function onMorphHandleDrag_(obj)
            if obj.StateMorph_ActivePtIdx_ < 0, return; end
            side = obj.StateMorph_ActiveSide_;
            if strcmp(side, 'A'), ax = obj.AxMorph_A_;
            else,                  ax = obj.AxMorph_B_; end
            cp = get(ax, 'CurrentPoint');
            pt = cp(1, 1:2);
            if strcmp(side, 'A')
                obj.StateMorph_PtsA_(obj.StateMorph_ActivePtIdx_, :) = pt;
            else
                obj.StateMorph_PtsB_(obj.StateMorph_ActivePtIdx_, :) = pt;
            end
            obj.redrawMorphGrid_(side);
            obj.updateTabMorph_();
        end
        
        function onMorphHandleUp_(obj)
            obj.StateMorph_ActivePtIdx_  = -1;
            obj.StateMorph_ActiveSide_   = '';
            obj.Fig_.WindowButtonMotionFcn = '';
            obj.Fig_.WindowButtonUpFcn     = '';
        end
        
        function onMorphDeletePairPressed_(obj)
            if size(obj.StateMorph_PtsA_, 1) < 1, return; end
            obj.StateMorph_PtsA_ = obj.StateMorph_PtsA_(1:end-1, :);
            obj.StateMorph_PtsB_ = obj.StateMorph_PtsB_(1:end-1, :);
            obj.LblMorph_PairCount_.Text = sprintf('Control pairs: %d', ...
                size(obj.StateMorph_PtsA_, 1));
            obj.redrawMorphGrid_('A'); obj.redrawMorphGrid_('B');
            obj.updateTabMorph_();
        end
        
        function onMorphClearPairsPressed_(obj)
            obj.StateMorph_PtsA_ = zeros(0, 2);
            obj.StateMorph_PtsB_ = zeros(0, 2);
            obj.LblMorph_PairCount_.Text = 'Control pairs: 0';
            obj.redrawMorphGrid_('A'); obj.redrawMorphGrid_('B');
            obj.updateTabMorph_();
        end
        
        function onMorphResetPtsPressed_(obj)
            [HA, WA, ~] = size(obj.StateMorph_ImageA_);
            [pts, ~] = OceanMath.Image.Resampling.ImageMorpher.uniformGridPoints( ...
                [HA, WA], obj.StateMorph_GridRes_);
            obj.StateMorph_PtsA_ = pts;
            obj.StateMorph_PtsB_ = pts;
            obj.LblMorph_PairCount_.Text = sprintf('Control pairs: %d', size(pts, 1));
            obj.redrawMorphGrid_('A'); obj.redrawMorphGrid_('B');
            obj.updateTabMorph_();
        end
        
        function onMorphPlayPressed_(obj)
            if ~isempty(obj.MorphTimer_) && isvalid(obj.MorphTimer_) && ...
               strcmp(obj.MorphTimer_.Running, 'on')
                stop(obj.MorphTimer_); delete(obj.MorphTimer_);
                obj.BtnMorph_Play_.Text = '▶  Play Animation';
                obj.BtnMorph_Play_.BackgroundColor = obj.COLOR_ACCENT;
                return;
            end
            obj.MorphTimerT_ = obj.StateMorph_t_;
            obj.BtnMorph_Play_.Text = '■  Stop';
            obj.BtnMorph_Play_.BackgroundColor = obj.COLOR_SECONDARY;
            obj.MorphTimer_ = timer('ExecutionMode', 'fixedRate', 'Period', 0.08, ...
                'TimerFcn', @(~,~) obj.onMorphTimerTick_());
            start(obj.MorphTimer_);
        end
        
        function onMorphTimerTick_(obj)
            if ~isvalid(obj) || isempty(obj.Fig_) || ~isvalid(obj.Fig_)
                stop(obj.MorphTimer_); return;
            end
            obj.MorphTimerT_ = mod(obj.MorphTimerT_ + 0.025, 1.0);
            obj.StateMorph_t_ = obj.MorphTimerT_;
            obj.SliderMorph_t_.Value = obj.MorphTimerT_;
            obj.LblMorph_t_.Text = sprintf('%.2f', obj.MorphTimerT_);
            obj.updateTabMorph_();
        end
        
        function onMorphSaveGifPressed_(obj)
            [file, pth] = uiputfile('morph_animation.gif', 'Save Morph GIF');
            if isequal(file, 0), return; end
            N = 24; t_vals = linspace(0, 1, N);
            gifPath = fullfile(pth, file);
            switch obj.StateMorph_Mode_
                case 'Piecewise Affine',  wMode = 'PiecewiseAffine';
                case 'Thin Plate Spline', wMode = 'ThinPlateSpline';
                case 'B-Spline Grid',     wMode = 'BSplineGrid';
                otherwise,                wMode = 'PiecewiseAffine';
            end
            res = obj.StateMorph_GridRes_;
            for i = 1:N
                fr = OceanMath.Image.Resampling.ImageMorpher.morphFrame( ...
                    obj.StateMorph_ImageA_, obj.StateMorph_ImageB_, ...
                    obj.StateMorph_PtsA_,   obj.StateMorph_PtsB_, t_vals(i), ...
                    'WarpMode', wMode, 'GridRes', [res, res]);
                [rgb_idx, cmap] = rgb2ind(im2uint8(fr), 256);
                if i == 1
                    imwrite(rgb_idx, cmap, gifPath, 'gif', 'Loopcount', inf, 'DelayTime', 0.06);
                else
                    imwrite(rgb_idx, cmap, gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', 0.06);
                end
            end
            msgbox(sprintf('Saved: %s', gifPath), 'GIF Saved');
        end
        
        function onMorphRotAngleChanged_(obj)
            obj.StateMorph_RotAngle_ = round(obj.SliderMorph_RotAngle_.Value);
            obj.LblMorph_RotAngle_.Text = sprintf('%d deg', obj.StateMorph_RotAngle_);
        end
        
        function onMorphTestRotationPressed_(obj)
            [HA, WA, ~] = size(obj.StateMorph_ImageA_);
            res = obj.StateMorph_GridRes_;
            [pA, pB] = OceanMath.Image.Resampling.ImageMorpher.rotationControlPoints( ...
                [HA, WA], obj.StateMorph_RotAngle_, res);
            obj.StateMorph_PtsA_ = pA;
            obj.StateMorph_PtsB_ = pB;
            obj.LblMorph_PairCount_.Text = sprintf('Control pairs: %d (rotation test)', ...
                size(pA, 1));
            obj.redrawMorphGrid_('A'); obj.redrawMorphGrid_('B');
            obj.updateTabMorph_();
        end
        
        function onMorphShowTheory_(obj)
            msg = sprintf([...
                'Feature-Based Morphing (Sec 3.6.3)\n', ...
                '====================================\n\n', ...
                '4-Step Pipeline for each frame at t:\n\n', ...
                'Step 1. Build Delaunay triangulation\n', ...
                '   from source control points.\n\n', ...
                'Step 2. Interpolate control positions:\n', ...
                '   pts_mid = (1-t)*pts_A + t*pts_B\n\n', ...
                'Step 3. Inverse-warp BOTH images toward\n', ...
                '   the intermediate skeleton:\n', ...
                '   I_A_warp = warp(I_A, pts_A -> pts_mid)\n', ...
                '   I_B_warp = warp(I_B, pts_B -> pts_mid)\n\n', ...
                'Step 4. Cross-dissolve:\n', ...
                '   I_morph = (1-t)*I_A_warp + t*I_B_warp\n\n', ...
                'Part 2: Rotation Artifact\n', ...
                '   Linear interpolation of Cartesian\n', ...
                '   positions traces chords, not arcs.\n', ...
                '   At t=0.5, radius shrinks by cos(theta/2).\n', ...
                '   This causes an "implosion" artifact\n', ...
                '   for large rotation angles (> 30 deg).']);
            msgbox(msg, 'Morphing Theory', 'help');
        end
        
        function updateMorphInfo_(obj)
            switch obj.StateMorph_Mode_
                case 'Piecewise Affine'
                    desc = sprintf([...
                        'Piecewise Affine Morphing:\n\n', ...
                        'Delaunay triangulation over control\n', ...
                        'points. Each pixel maps back to its\n', ...
                        'source triangle via barycentric weights.\n\n', ...
                        'C0-continuous (continuous across\n', ...
                        'triangle boundaries, no derivative\n', ...
                        'continuity guaranteed).\n\n', ...
                        'Step 2: pts_mid = lerp(pts_A, pts_B, t)\n', ...
                        'Step 3: inverse-warp both images\n', ...
                        'Step 4: (1-t)*I_A_warp + t*I_B_warp']);
                case 'Thin Plate Spline'
                    desc = sprintf([...
                        'Thin Plate Spline Morphing:\n\n', ...
                        'Globally smooth RBF displacement field\n', ...
                        'fitted from sparse control pairs.\n\n', ...
                        'C-infinity continuous: minimizes total\n', ...
                        'bending energy. Slower for large N\n', ...
                        '(solves N x N linear system).']);
                case 'B-Spline Grid'
                    desc = sprintf([...
                        'B-Spline FFD Morphing:\n\n', ...
                        'Tensor-product cubic B-spline kernels\n', ...
                        'on a regular control grid. Each pixel\n', ...
                        'influenced by its 4x4 neighborhood.\n\n', ...
                        'C2-continuous: curvature-smooth.\n', ...
                        'Highly efficient for dense grids.']);
                otherwise
                    desc = '';
            end
            if isgraphics(obj.LblMorph_Info_Desc_)
                obj.LblMorph_Info_Desc_.Text = desc; end
            N = size(obj.StateMorph_PtsA_, 1);
            nTri = 0;
            if N >= 3
                try
                    tri = delaunay(obj.StateMorph_PtsA_(:,1), obj.StateMorph_PtsA_(:,2));
                    nTri = size(tri, 1);
                catch; end
            end
            stats = sprintf([...
                'Statistics:\n', ...
                '----------------------------\n', ...
                'Image A: %s\n', ...
                'Image B: %s\n', ...
                'Image size: %d x %d\n', ...
                'Control pairs: %d\n', ...
                'Triangles: %d\n', ...
                'Warp mode: %s\n', ...
                'Grid res: %d x %d\n', ...
                't = %.3f'], ...
                obj.StateMorph_ImageNameA_, obj.StateMorph_ImageNameB_, ...
                size(obj.StateMorph_ImageA_, 1), size(obj.StateMorph_ImageA_, 2), ...
                N, nTri, obj.StateMorph_Mode_, ...
                obj.StateMorph_GridRes_, obj.StateMorph_GridRes_, ...
                obj.StateMorph_t_);
            if isgraphics(obj.LblMorph_Info_Stats_)
                obj.LblMorph_Info_Stats_.Text = stats; end
        end



        % ================================================================== %
        % Tab 10: 2D Collage Editor (Ex 3.26) - Builder
        % ================================================================== %
        function buildTabCollage_(obj)
            % Sidebar
            obj.PanCollageControl_ = uipanel(obj.TabCollage_, ...
                'Position', [10, 10, obj.SIDEBAR_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, ...
                'BorderType', 'none', 'Scrollable', 'on');
            
            % Canvas
            cW = obj.FIG_WIDTH - obj.SIDEBAR_W - obj.INFO_W - 40;
            obj.PanCollageCanvas_ = uipanel(obj.TabCollage_, ...
                'Position', [obj.SIDEBAR_W + 20, 10, cW, 690], ...
                'BackgroundColor', obj.COLOR_BG, 'BorderType', 'none');
            
            % Info panel
            obj.PanCollageInfo_ = uipanel(obj.TabCollage_, ...
                'Position', [obj.FIG_WIDTH - obj.INFO_W - 10, 10, obj.INFO_W, 690], ...
                'BackgroundColor', obj.COLOR_PANEL, 'BorderType', 'none');
            
            % ---- Sidebar controls ----
            yPos = 655;
            uilabel(obj.PanCollageControl_, 'Position', [15, yPos, 280, 25], ...
                'Text', '2D Collage Editor (Ex 3.26)', ...
                'FontColor', obj.COLOR_ACCENT, 'FontSize', 14, 'FontWeight', 'bold');
            uilabel(obj.PanCollageControl_, 'Position', [15, yPos-5, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Import Image Button
            yPos = yPos - 38;
            obj.BtnCollage_Import_ = uibutton(obj.PanCollageControl_, ...
                'Position', [15, yPos, 250, 30], ...
                'Text', '➕  Import Image & Crop...', 'FontSize', 10, ...
                'BackgroundColor', [0.18, 0.55, 0.32], 'FontColor', [1,1,1], ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~,~) obj.onCollageImportPressed_());
            
            % Separator
            yPos = yPos - 18;
            uilabel(obj.PanCollageControl_, 'Position', [15, yPos, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Deformation Mode
            yPos = yPos - 30;
            uilabel(obj.PanCollageControl_, 'Position', [15, yPos+2, 120, 20], ...
                'Text', 'Deformation Mode:', 'FontColor', obj.COLOR_TEXT, 'FontSize', 11);
            obj.DropdownCollage_Mode_ = uidropdown(obj.PanCollageControl_, ...
                'Position', [15, yPos-20, 250, 28], ...
                'Items', {'Translation', 'Rigid', 'Similarity', 'Affine', 'Perspective'}, ...
                'Value', obj.StateCollage_Mode_, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ValueChangedFcn', @(~,~) obj.onCollageModeChanged_());
            
            % Layer controls
            yPos = yPos - 55;
            uilabel(obj.PanCollageControl_, 'Position', [15, yPos, 250, 20], ...
                'Text', 'Layer Management:', 'FontColor', obj.COLOR_TEXT, ...
                'FontSize', 11, 'FontWeight', 'bold');
            
            yPos = yPos - 32;
            obj.BtnCollage_BringFront_ = uibutton(obj.PanCollageControl_, ...
                'Position', [15, yPos, 120, 26], 'Text', '▲ Bring to Front', 'FontSize', 9.5, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onCollageLayerAction_('front'));
            obj.BtnCollage_SendBack_ = uibutton(obj.PanCollageControl_, ...
                'Position', [145, yPos, 120, 26], 'Text', '▼ Send to Back', 'FontSize', 9.5, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_TEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onCollageLayerAction_('back'));
            
            yPos = yPos - 32;
            obj.BtnCollage_Delete_ = uibutton(obj.PanCollageControl_, ...
                'Position', [15, yPos, 120, 26], 'Text', '❌ Delete Layer', 'FontSize', 9.5, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_SECONDARY, ...
                'ButtonPushedFcn', @(~,~) obj.onCollageLayerAction_('delete'));
            obj.BtnCollage_ClearAll_ = uibutton(obj.PanCollageControl_, ...
                'Position', [145, yPos, 120, 26], 'Text', 'Clear All', 'FontSize', 9.5, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_SECONDARY, ...
                'ButtonPushedFcn', @(~,~) obj.onCollageLayerAction_('clear'));
            
            % Separator
            yPos = yPos - 20;
            uilabel(obj.PanCollageControl_, 'Position', [15, yPos, 280, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            % Show grid checkbox
            yPos = yPos - 30;
            obj.ChkCollage_ShowGrid_ = uicheckbox(obj.PanCollageControl_, ...
                'Position', [15, yPos, 180, 22], 'Text', 'Show Active Outlines', ...
                'FontColor', obj.COLOR_TEXT, 'FontSize', 11, ...
                'Value', obj.StateCollage_ShowGrid_, ...
                'ValueChangedFcn', @(~,~) obj.updateTabCollage_());
            
            % Theory button
            yPos = yPos - 35;
            obj.BtnCollage_Theory_ = uibutton(obj.PanCollageControl_, ...
                'Position', [15, yPos, 120, 26], 'Text', '?  Show Theory', 'FontSize', 10, ...
                'BackgroundColor', obj.COLOR_BTN, 'FontColor', obj.COLOR_SUBTEXT, ...
                'ButtonPushedFcn', @(~,~) obj.onCollageShowTheory_());
            
            % Instructions
            yPos = yPos - 120;
            uilabel(obj.PanCollageControl_, ...
                'Position', [15, yPos, 250, 100], ...
                'Text', sprintf('Instructions:\n- Click "Import Image & Crop" to add a layer.\n- Click inside any layer on the canvas to select it.\n- Drag orange handles to deform the layer.\n- Change layers order with Bring/Send buttons.'), ...
                'FontSize', 9, 'FontColor', obj.COLOR_SUBTEXT, 'WordWrap', 'on');
            
            % ---- Canvas panel ----
            obj.AxCollage_ = axes(obj.PanCollageCanvas_, ...
                'Position', [10/cW, 10/690, (cW-20)/cW, 670/690], ...
                'Color', obj.COLOR_BG, 'XColor', obj.COLOR_MUTED, 'YColor', obj.COLOR_MUTED, ...
                'Units', 'normalized', 'XTick', [], 'YTick', []);
            axis(obj.AxCollage_, 'image');
            
            % ---- Info panel ----
            iW = obj.INFO_W - 20;
            uilabel(obj.PanCollageInfo_, 'Position', [10, 655, iW, 22], ...
                'Text', 'Collage Details', 'FontColor', obj.COLOR_ACCENT, ...
                'FontSize', 13, 'FontWeight', 'bold');
            uilabel(obj.PanCollageInfo_, 'Position', [10, 648, iW, 1], ...
                'BackgroundColor', obj.COLOR_MUTED, 'Text', '');
            
            obj.LblCollage_Info_Desc_ = uilabel(obj.PanCollageInfo_, ...
                'Position', [10, 360, iW, 280], 'Text', '', 'WordWrap', 'on', ...
                'FontColor', obj.COLOR_TEXT, 'FontSize', 9.5, 'VerticalAlignment', 'top');
            
            obj.LblCollage_Info_Stats_ = uilabel(obj.PanCollageInfo_, ...
                'Position', [10, 10, iW, 340], 'Text', '', 'WordWrap', 'on', ...
                'FontColor', obj.COLOR_SUBTEXT, 'FontSize', 9, 'VerticalAlignment', 'top');
        end
        
        % ================================================================== %
        % Tab 10: Geometry initialization
        % ================================================================== %
        function buildCollageGeometry_(obj)
            obj.StateCollage_Layers_ = {};
            obj.StateCollage_SelectedIdx_ = -1;
            obj.StateCollage_ActivePtIdx_ = -1;
            
            % Composite initial empty canvas
            I_comp = obj.compositeCanvas_();
            obj.PlotCollage_ImgHandle_ = imshow(I_comp, 'Parent', obj.AxCollage_);
            set(obj.PlotCollage_ImgHandle_, 'ButtonDownFcn', @(src, e) obj.onCollageMouseDown_(src, e));
            axis(obj.AxCollage_, 'image'); axis(obj.AxCollage_, 'off');
            
            obj.updateCollageInfo_();
        end
        
        % ================================================================== %
        % Tab 10: High-performance Compositing Pipeline
        % ================================================================== %
        function I_comp = compositeCanvas_(obj, excludeIdx)
            if nargin < 2, excludeIdx = -1; end
            
            % Fill canvas with dark background
            I_comp = repmat(reshape([0.10, 0.12, 0.16], 1, 1, 3), 600, 800);
            
            numLayers = numel(obj.StateCollage_Layers_);
            for idx = 1:numLayers
                if idx == excludeIdx, continue; end
                layer = obj.StateCollage_Layers_{idx};
                I_comp = obj.warpLayerOntoCanvas_(I_comp, layer);
            end
        end
        
        function I_canvas = warpLayerOntoCanvas_(obj, I_canvas, layer)
            I_src = layer.img;
            [H_src, W_src, C] = size(I_src);
            H_mat = layer.H;
            
            % 1. Target bounding box
            pts_hom = [layer.base_corners, ones(4, 1)]';
            warped = H_mat * pts_hom;
            w = warped(3, :);
            w(abs(w) < 1e-12) = 1e-12;
            cx = warped(1, :) ./ w;
            cy = warped(2, :) ./ w;
            
            min_x = max(1, floor(min(cx)));
            max_x = min(800, ceil(max(cx)));
            min_y = max(1, floor(min(cy)));
            max_y = min(600, ceil(max(cy)));
            
            if min_x > max_x || min_y > max_y, return; end
            
            % 2. Target coordinates grid
            [X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);
            N_pts = numel(X_grid);
            pts_dst = [X_grid(:), Y_grid(:), ones(N_pts, 1)]';
            
            % 3. Backward mapping
            H_inv = inv(H_mat);
            pts_src = H_inv * pts_dst;
            ws = pts_src(3, :);
            ws(abs(ws) < 1e-12) = 1e-12;
            src_x = pts_src(1, :) ./ ws;
            src_y = pts_src(2, :) ./ ws;
            
            % 4. Sample and overlay
            inside = (src_x >= 0 & src_x <= W_src-1 & src_y >= 0 & src_y <= H_src-1);
            if ~any(inside), return; end
            
            src_x_1 = src_x(inside) + 1;
            src_y_1 = src_y(inside) + 1;
            x_in = X_grid(inside);
            y_in = Y_grid(inside);
            
            x0 = floor(src_x_1);
            y0 = floor(src_y_1);
            x1 = x0 + 1;
            y1 = y0 + 1;
            
            x0_c = max(1, min(W_src, x0));
            x1_c = max(1, min(W_src, x1));
            y0_c = max(1, min(H_src, y0));
            y1_c = max(1, min(H_src, y1));
            
            dx = src_x_1 - x0;
            dy = src_y_1 - y0;
            
            w00 = (1 - dx) .* (1 - dy);
            w01 = dx .* (1 - dy);
            w10 = (1 - dx) .* dy;
            w11 = dx .* dy;
            
            idx00 = y0_c + (x0_c - 1) * H_src;
            idx01 = y0_c + (x1_c - 1) * H_src;
            idx10 = y1_c + (x0_c - 1) * H_src;
            idx11 = y1_c + (x1_c - 1) * H_src;
            
            canvas_idx = y_in + (x_in - 1) * 600;
            
            for c = 1:C
                Ic = I_src(:, :, c);
                val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                I_canvas(canvas_idx + (c - 1) * 600 * 800) = val;
            end
        end
        
        % ================================================================== %
        % Tab 10: Canvas rendering update
        % ================================================================== %
        function updateTabCollage_(obj)
            % Redraw collage canvas
            I_comp = obj.compositeCanvas_();
            if isgraphics(obj.PlotCollage_ImgHandle_)
                set(obj.PlotCollage_ImgHandle_, 'CData', im2uint8(I_comp));
            else
                obj.PlotCollage_ImgHandle_ = imshow(I_comp, 'Parent', obj.AxCollage_);
                set(obj.PlotCollage_ImgHandle_, 'ButtonDownFcn', @(src, e) obj.onCollageMouseDown_(src, e));
                axis(obj.AxCollage_, 'image'); axis(obj.AxCollage_, 'off');
            end
            
            % Draw/clear outlines & handles
            obj.redrawCollageHandles_();
            obj.updateCollageInfo_();
            drawnow('limitrate');
        end
        
        function corners = getCollageShapeCorners_(obj, layer)
            pts_hom = [layer.base_corners, ones(4, 1)]';
            warped = layer.H * pts_hom;
            w = warped(3, :);
            w(abs(w) < 1e-15) = 1e-15;
            corners = [warped(1, :) ./ w; warped(2, :) ./ w]';
        end
        
        function redrawCollageHandles_(obj)
            hold(obj.AxCollage_, 'on');
            
            % Clear old handles
            if ~isempty(obj.PlotCollage_Wireframe_) && isgraphics(obj.PlotCollage_Wireframe_)
                delete(obj.PlotCollage_Wireframe_); end
            obj.PlotCollage_Wireframe_ = [];
            
            if ~isempty(obj.PlotCollage_Handles_)
                for k = 1:numel(obj.PlotCollage_Handles_)
                    if isgraphics(obj.PlotCollage_Handles_{k}), delete(obj.PlotCollage_Handles_{k}); end
                end
            end
            obj.PlotCollage_Handles_ = [];
            
            for k = 1:numel(obj.PlotCollage_Texts_)
                if isgraphics(obj.PlotCollage_Texts_{k}), delete(obj.PlotCollage_Texts_{k}); end
            end
            obj.PlotCollage_Texts_ = {};
            
            % Draw new handles if a layer is selected
            if obj.StateCollage_SelectedIdx_ ~= -1 && obj.StateCollage_ShowGrid_
                layer = obj.StateCollage_Layers_{obj.StateCollage_SelectedIdx_};
                corners = obj.getCollageShapeCorners_(layer);
                
                % Draw wireframe border (looping back to 1)
                wf = plot(obj.AxCollage_, corners([1,2,3,4,1], 1), corners([1,2,3,4,1], 2), ...
                    'Color', obj.COLOR_ACCENT, 'LineWidth', 1.2, 'HitTest', 'off');
                obj.PlotCollage_Wireframe_ = wf;
                
                % Draw 4 corner handles
                col = [1.00, 0.45, 0.15]; % orange handle
                obj.PlotCollage_Handles_ = cell(1, 4);
                for c = 1:4
                    h = plot(obj.AxCollage_, corners(c, 1), corners(c, 2), 'o', ...
                        'MarkerSize', 7, 'MarkerFaceColor', col, 'MarkerEdgeColor', [1,1,1], ...
                        'LineWidth', 1.0, 'HitTest', 'off');
                    obj.PlotCollage_Handles_{c} = h;
                end
                
                % Layer name text label
                cx = mean(corners(:, 1));
                cy = mean(corners(:, 2));
                obj.PlotCollage_Texts_{1} = text(obj.AxCollage_, cx, cy, layer.name, ...
                    'Color', [1,1,1], 'FontSize', 8, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', 'BackgroundColor', [0.11,0.11,0.13, 0.6], ...
                    'HitTest', 'off');
            end
            
            hold(obj.AxCollage_, 'off');
        end
        
        function inside = pointInCollagePolygon_(~, px, py, corners)
            inside = inpolygon(px, py, corners(:, 1), corners(:, 2));
        end
        
        % ================================================================== %
        % Tab 10: Crop Window Interaction
        % ================================================================== %
        function onCollageImportPressed_(obj)
            [file, pth] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp', 'Images'});
            if isequal(file, 0), return; end
            img = imread(fullfile(pth, file));
            
            % Create separate helper cropping window
            cropFig = uifigure('Name', 'Shift-Drag Rubber-band Crop Area', ...
                'Position', [200, 200, 550, 600], 'Color', obj.COLOR_BG);
            
            ax = uiaxes(cropFig, 'Position', [10, 80, 530, 500], ...
                'Color', obj.COLOR_PANEL);
            imshow(img, 'Parent', ax);
            axis(ax, 'image');
            
            % Draw rectangle ROI tool
            roi = drawrectangle(ax, 'Position', [10, 10, size(img, 2)-20, size(img, 1)-20], ...
                'Color', obj.COLOR_ACCENT);
            
            uibutton(cropFig, ...
                'Position', [175, 20, 200, 36], ...
                'Text', '✂  Confirm Crop & Paste', 'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.18, 0.55, 0.32], 'FontColor', [1,1,1], ...
                'ButtonPushedFcn', @(~,~) obj.onCollageCropDoneCallback_(cropFig, roi, img));
        end
        
        function onCollageCropDoneCallback_(obj, fig, roi, img)
            pos = roi.Position; % [x, y, w, h]
            delete(fig); % close window
            
            % Crop boundary clamps
            x0 = max(1, floor(pos(1)));
            x1 = min(size(img, 2), ceil(pos(1)+pos(3)));
            y0 = max(1, floor(pos(2)));
            y1 = min(size(img, 1), ceil(pos(2)+pos(4)));
            
            if (x1 - x0) < 5 || (y1 - y0) < 5
                warning('Crop area is too small.');
                return;
            end
            
            cropped = img(y0:y1, x0:x1, :);
            obj.onCollageCropDone_(cropped);
        end
        
        function onCollageCropDone_(obj, cropped)
            [h, w, ~] = size(cropped);
            
            % Constraint: Scale image to maximum size of 200 px to fit canvas nicely
            maxDim = 200;
            if max(h, w) > maxDim
                scale = maxDim / max(h, w);
                cropped = imresize(cropped, scale);
                [h, w, ~] = size(cropped);
            end
            
            layer.img = double(cropped) / 255.0;
            if size(layer.img, 3) == 1
                layer.img = repmat(layer.img, 1, 1, 3);
            end
            layer.base_corners = [ ...
                0, 0; ...
                w, 0; ...
                w, h; ...
                0, h ...
            ];
            layer.H = eye(3);
            % Position layer initially at the center of the 800x600 canvas
            layer.H(1, 3) = (800 - w) / 2;
            layer.H(2, 3) = (600 - h) / 2;
            layer.name = sprintf('Layer %d', numel(obj.StateCollage_Layers_) + 1);
            
            obj.StateCollage_Layers_{end+1} = layer;
            obj.StateCollage_SelectedIdx_ = numel(obj.StateCollage_Layers_);
            
            obj.updateTabCollage_();
        end
        
        % ================================================================== %
        % Tab 10: Mouse dragging interaction handlers
        % ================================================================== %
        function onCollageMouseDown_(obj, ~, event)
            cp = obj.AxCollage_.CurrentPoint;
            x = cp(1, 1);
            y = cp(1, 2);
            
            % 1. Check if click hits corner handle of the selected layer
            if obj.StateCollage_SelectedIdx_ ~= -1
                layer = obj.StateCollage_Layers_{obj.StateCollage_SelectedIdx_};
                corners = obj.getCollageShapeCorners_(layer);
                for ptIdx = 1:4
                    dist = norm(corners(ptIdx, :) - [x, y]);
                    if dist < 12 % click radius
                        obj.StateCollage_ActivePtIdx_ = ptIdx;
                        % Pre-composite and cache other layers to run at 60fps!
                        obj.StateCollage_BgCache_ = obj.compositeCanvas_(obj.StateCollage_SelectedIdx_);
                        
                        obj.Fig_.WindowButtonMotionFcn = @(~,~) obj.onCollageMouseDrag_();
                        obj.Fig_.WindowButtonUpFcn     = @(~,~) obj.onCollageMouseUp_();
                        return;
                    end
                end
            end
            
            % 2. Check if click is inside any layer (z-order: top down)
            numLayers = numel(obj.StateCollage_Layers_);
            for idx = numLayers:-1:1
                layer = obj.StateCollage_Layers_{idx};
                corners = obj.getCollageShapeCorners_(layer);
                if obj.pointInCollagePolygon_(x, y, corners)
                    obj.StateCollage_SelectedIdx_ = idx;
                    obj.updateTabCollage_();
                    return;
                end
            end
            
            % Clicked empty space -> deselect
            obj.StateCollage_SelectedIdx_ = -1;
            obj.updateTabCollage_();
        end
        
        function onCollageMouseDrag_(obj)
            if obj.StateCollage_SelectedIdx_ < 1 || obj.StateCollage_ActivePtIdx_ < 1
                return;
            end
            cp = obj.AxCollage_.CurrentPoint;
            x = cp(1, 1);
            y = cp(1, 2);
            
            layer = obj.StateCollage_Layers_{obj.StateCollage_SelectedIdx_};
            mode = obj.StateCollage_Mode_;
            corners = obj.getCollageShapeCorners_(layer);
            k = obj.StateCollage_ActivePtIdx_;
            
            Q = corners; % Copy target corners
            
            if strcmp(mode, 'Translation')
                dx = x - corners(k, 1);
                dy = y - corners(k, 2);
                Q = Q + [dx, dy];
                
            elseif strcmp(mode, 'Rigid')
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
                opp_map = [3, 4, 1, 2];
                opp = opp_map(k);
                p_opp = corners(opp, :);
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
                catch; end
                
            elseif strcmp(mode, 'Affine')
                opp_map = [3, 4, 1, 2];
                opp = opp_map(k);
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
                catch; end
                
            elseif strcmp(mode, 'Perspective')
                Q(k, :) = [x, y];
            end
            
            % Update homography
            layer.H = obj.fitCollageTransformation_(mode, layer.base_corners, Q);
            obj.StateCollage_Layers_{obj.StateCollage_SelectedIdx_} = layer;
            
            % Composite active layer onto cached background
            I_comp = obj.warpLayerOntoCanvas_(obj.StateCollage_BgCache_, layer);
            set(obj.PlotCollage_ImgHandle_, 'CData', im2uint8(I_comp));
            
            obj.redrawCollageHandles_();
            drawnow('limitrate');
        end
        
        function H = fitCollageTransformation_(obj, mode, src, dst)
            if strcmp(mode, 'Translation')
                tx = mean(dst(:,1)) - mean(src(:,1));
                ty = mean(dst(:,2)) - mean(src(:,2));
                H = eye(3); H(1, 3) = tx; H(2, 3) = ty;
            elseif strcmp(mode, 'Rigid')
                c_src = mean(src, 1); c_dst = mean(dst, 1);
                src_c = src - c_src;  dst_c = dst - c_dst;
                [U, ~, V] = svd(src_c' * dst_c);
                R = V * U';
                if det(R) < 0
                    V(:, 2) = V(:, 2) * -1; R = V * U';
                end
                H = eye(3); H(1:2, 1:2) = R; H(1:2, 3) = c_dst' - R * c_src';
            elseif strcmp(mode, 'Similarity')
                A = zeros(8, 4); B = zeros(8, 1);
                for i = 1:4
                    x = src(i,1); y = src(i,2);
                    u = dst(i,1); v = dst(i,2);
                    A(2*i-1, :) = [x, -y, 1, 0]; A(2*i, :) = [y, x, 0, 1];
                    B(2*i-1) = u; B(2*i) = v;
                end
                c = A \ B;
                H = eye(3);
                H(1, 1) = c(1); H(1, 2) = -c(2); H(1, 3) = c(3);
                H(2, 1) = c(2); H(2, 2) = c(1);  H(2, 3) = c(4);
            elseif strcmp(mode, 'Affine')
                A = [src, ones(4,1)];
                H = eye(3); H(1, :) = (A \ dst(:,1))'; H(2, :) = (A \ dst(:,2))';
            elseif strcmp(mode, 'Perspective')
                A = zeros(8, 8); B = zeros(8, 1);
                for i = 1:4
                    x = src(i,1); y = src(i,2);
                    u = dst(i,1); v = dst(i,2);
                    A(2*i-1, :) = [x, y, 1, 0, 0, 0, -u*x, -u*y];
                    A(2*i, :)   = [0, 0, 0, x, y, 1, -v*x, -v*y];
                    B(2*i-1) = u; B(2*i) = v;
                end
                h = A \ B;
                H = eye(3);
                H(1, 1) = h(1); H(1, 2) = h(2); H(1, 3) = h(3);
                H(2, 1) = h(4); H(2, 2) = h(5); H(2, 3) = h(6);
                H(3, 1) = h(7); H(3, 2) = h(8); H(3, 3) = 1.0;
            end
        end
        
        function onCollageMouseUp_(obj)
            obj.StateCollage_ActivePtIdx_ = -1;
            obj.Fig_.WindowButtonMotionFcn = '';
            obj.Fig_.WindowButtonUpFcn     = '';
            obj.updateTabCollage_();
        end
        
        % ================================================================== %
        % Tab 10: Layer management actions
        % ================================================================== %
        function onCollageLayerAction_(obj, action)
            idx = obj.StateCollage_SelectedIdx_;
            if strcmp(action, 'clear')
                obj.StateCollage_Layers_ = {};
                obj.StateCollage_SelectedIdx_ = -1;
                obj.updateTabCollage_();
                return;
            end
            if idx == -1, return; end
            
            switch action
                case 'delete'
                    obj.StateCollage_Layers_(idx) = [];
                    obj.StateCollage_SelectedIdx_ = -1;
                case 'front'
                    % Move to end of cell array
                    layer = obj.StateCollage_Layers_{idx};
                    obj.StateCollage_Layers_(idx) = [];
                    obj.StateCollage_Layers_{end+1} = layer;
                    obj.StateCollage_SelectedIdx_ = numel(obj.StateCollage_Layers_);
                case 'back'
                    % Move to index 1
                    layer = obj.StateCollage_Layers_{idx};
                    obj.StateCollage_Layers_(idx) = [];
                    obj.StateCollage_Layers_ = [{layer}, obj.StateCollage_Layers_];
                    obj.StateCollage_SelectedIdx_ = 1;
            end
            obj.updateTabCollage_();
        end
        
        function onCollageModeChanged_(obj)
            obj.StateCollage_Mode_ = obj.DropdownCollage_Mode_.Value;
            obj.updateTabCollage_();
        end
        
        function onCollageShowTheory_(obj)
            msg = sprintf([...
                '2D Collage Editor (Ex 3.26)\n', ...
                '===========================\n\n', ...
                'Core operations:\n', ...
                '1. Shift-drag Selection (Rubber-band crop):\n', ...
                '   Extracts local region from imported image.\n\n', ...
                '2. Layer-wise Homography Fitting:\n', ...
                '   Given base corners and dragged handle coordinates,\n', ...
                '   estimates a 3x3 pose matrix H mapping the sub-image.\n\n', ...
                '3. Vectorized Composite Pipeline:\n', ...
                '   Composites active layer on top of background cache\n', ...
                '   via inverse projective mapping to sustain 60+ FPS.\n\n', ...
                '4. Z-Order Control:\n', ...
                '   Determines rendering precedence from bottom to top.']);
            msgbox(msg, 'Collage Editor Theory', 'help');
        end
        
        function updateCollageInfo_(obj)
            desc = sprintf([...
                '2D Image Collage Editor:\n\n', ...
                'Assemble a scene by importing, cropping,\n', ...
                'and transforming multiple image layers.\n\n', ...
                'Select any layer to view and edit its\n', ...
                'poses usingTranslation, Rigid, Similarity,\n', ...
                'Affine, or Perspective models.\n\n', ...
                'Bilinear backward mapping coordinates are\n', ...
                'computed in destination space and sampled\n', ...
                'on-the-fly inside bounding box limits.']);
            if isgraphics(obj.LblCollage_Info_Desc_)
                obj.LblCollage_Info_Desc_.Text = desc; end
            
            selLayer = 'None';
            w = 0; h = 0;
            if obj.StateCollage_SelectedIdx_ ~= -1
                l = obj.StateCollage_Layers_{obj.StateCollage_SelectedIdx_};
                selLayer = l.name;
                [h, w, ~] = size(l.img);
            end
            
            stats = sprintf([...
                'Collage Statistics:\n', ...
                '----------------------------\n', ...
                'Total Layers: %d\n', ...
                'Selected Layer: %s\n', ...
                'Layer crop size: %d x %d\n', ...
                'Deformation Mode: %s\n', ...
                'Canvas size: 800 x 600 px'], ...
                numel(obj.StateCollage_Layers_), selLayer, w, h, obj.StateCollage_Mode_);
            if isgraphics(obj.LblCollage_Info_Stats_)
                obj.LblCollage_Info_Stats_.Text = stats; end
        end


    end
end

