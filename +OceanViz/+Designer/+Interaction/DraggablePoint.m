classdef DraggablePoint < handle
    % DRAGGABLEPOINT  A draggable marker on an axes.
    %
    % Creates a plot marker that the user can click and drag. Fires a
    % callback on every mouse move and on release.
    %
    % Usage:
    %   dp = OceanViz.Designer.Interaction.DraggablePoint(ax, [3, 5]);
    %   dp.OnMove = @(pos) disp(pos);
    %   dp.OnRelease = @(pos) disp('done');
    %
    % See also: OceanViz.Designer.TrackDesigner

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Position   (1,:) double
        MarkerHandle         % line/plot handle
        Axes_                % parent axes
        Figure_              % parent figure
    end

    properties
        OnGrab      % function handle: @(startPos)
        OnMove      % function handle: @(newPos)
        OnRelease   % function handle: @(newPos)
        Color       (1,3) double = [1, 0.4, 0]
        MarkerSize  (1,1) double = 10
        Tag         (1,:) char = ''
    end

    % ------------------------------------------------------------------ %
    properties (Access = private)
        Dragging_   (1,1) logical = false
        OldMotionFcn_
        OldReleaseFcn_
        
        % 3D Z-Editing States
        DragMode_         (1,:) char = 'normal'
        DragStartScreenY_ (1,1) double = 0
        DragStartZ_       (1,1) double = 0
        DragStartPos_     (1,3) double = [0 0 0]
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = DraggablePoint(ax, position, options)
            % DRAGGABLEPOINT  Create a draggable marker.
            %
            %   dp = DraggablePoint(ax, [x, y])
            %   dp = DraggablePoint(ax, [x, y], 'Color', [0 1 0])
            arguments
                ax
                position (1,:) double
                options.Color     (1,3) double = [1, 0.4, 0]
                options.MarkerSize (1,1) double = 10
                options.Tag       (1,:) char = ''
                options.OnGrab    = []
                options.OnMove    = []
                options.OnRelease = []
            end
            obj.Axes_     = ax;
            obj.Figure_   = ancestor(ax, 'figure');
            obj.Position  = position;
            obj.Color     = options.Color;
            obj.MarkerSize = options.MarkerSize;
            obj.Tag       = options.Tag;
            obj.OnGrab    = options.OnGrab;
            obj.OnMove    = options.OnMove;
            obj.OnRelease = options.OnRelease;

            % Draw the marker
            hold(ax, 'on');
            z = 0; if numel(position) >= 3, z = position(3); end
            obj.MarkerHandle = plot3(ax, position(1), position(2), z, 'o', ...
                'MarkerFaceColor', obj.Color, ...
                'MarkerEdgeColor', obj.Color * 0.7, ...
                'MarkerSize', obj.MarkerSize, ...
                'LineWidth', 1.5, ...
                'ButtonDownFcn', @(~,~) obj.onMouseDown_());
        end

        function setPosition(obj, newPos)
            % SETPOSITION  Move the marker programmatically.
            obj.Position = newPos;
            z = 0; if numel(newPos) >= 3, z = newPos(3); end
            set(obj.MarkerHandle, 'XData', newPos(1), 'YData', newPos(2), 'ZData', z);
        end

        function setColor(obj, c)
            % SETCOLOR  Change marker color.
            obj.Color = c;
            set(obj.MarkerHandle, 'MarkerFaceColor', c, ...
                'MarkerEdgeColor', c * 0.7);
        end

        function setVisible(obj, tf)
            % SETVISIBLE  Show or hide the marker.
            if tf
                set(obj.MarkerHandle, 'Visible', 'on');
            else
                set(obj.MarkerHandle, 'Visible', 'off');
            end
        end

        function delete(obj)
            if ~isempty(obj.MarkerHandle) && isgraphics(obj.MarkerHandle) && isvalid(obj.MarkerHandle)
                delete(obj.MarkerHandle);
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function onMouseDown_(obj)
            obj.Dragging_ = true;
            
            % Capture baseline point specifically initialized into explicit 3D format length-3
            startP = obj.Position;
            if numel(startP) < 3, startP = [startP(1), startP(2), 0]; end
            obj.DragStartPos_ = startP;
            
            if ~isempty(obj.OnGrab)
                obj.OnGrab(obj.Position);
            end
            
            % Capture drag mode (normal vs extend/alt for Z-axis)
            obj.DragMode_ = get(obj.Figure_, 'SelectionType');
            
            % Capture screen-pixel baseline for Y-axis (Z-height manipulation)
            pPix = get(obj.Figure_, 'CurrentPoint');
            obj.DragStartScreenY_ = pPix(2);
            
            obj.DragStartZ_ = 0;
            if numel(obj.Position) >= 3, obj.DragStartZ_ = obj.Position(3); end
            
            obj.OldMotionFcn_  = get(obj.Figure_, 'WindowButtonMotionFcn');
            obj.OldReleaseFcn_ = get(obj.Figure_, 'WindowButtonUpFcn');
            set(obj.Figure_, ...
                'WindowButtonMotionFcn', @(~,~) obj.onMouseMove_(), ...
                'WindowButtonUpFcn',     @(~,~) obj.onMouseUp_());
        end

        function onMouseMove_(obj)
            if ~obj.Dragging_, return; end
            
            if strcmp(obj.DragMode_, 'extend') || strcmp(obj.DragMode_, 'alt')
                % --- Z Axis Vertical Dragging ---
                % Modify Z only based on literal vertical screen mouse pixels
                pPix = get(obj.Figure_, 'CurrentPoint');
                deltaYPixels = pPix(2) - obj.DragStartScreenY_;
                
                % Drag Sensitivity: 1 screen pixel = 0.25 Z-units
                zScale = 0.25;
                newZ = obj.DragStartZ_ + deltaYPixels * zScale;
                newPos = [obj.Position(1), obj.Position(2), newZ];
            else
                % --- Standard 3D Camera-Plane Dragging (Replaces restricted X/Y) ---
                % Whatever angle you are looking at (Top, Side, Front, Isometric),
                % the point slides across the invisible plane parallel to your screen!
                %Find the plane parallel to eye ray perspective. The object movement should move on the plane parallel to the shine from our eye 
                %Ray equation P = P_front + tv 
                % Plane: (P - P_start) * N = 0. Valid P must give 0.
                cp = get(obj.Axes_, 'CurrentPoint');
                pFront = cp(1, :); %Where the eye shine hit the screen window
                pBack  = cp(2, :); %Where the shine hit the clipping plane
                vDir = pBack - pFront; %Directional vector. Colinear with eye -> glass window -> where light hit the clipping plane in the void of simulated world
                
                % Compute Camera Normal Vector. Global view.
                camPos = get(obj.Axes_, 'CameraPosition'); %The Visual camera position, 
                %Your physical body is sitting in a chair holding a mouse.
                %Your virtual body is standing inside a 3D simulated room holding a simulated gun.
                camTgt = get(obj.Axes_, 'CameraTarget'); %The point in space that the camera is looking at
                N = camPos - camTgt; %Get normal vec. From target point directly to simulated camera
                if norm(N) > 1e-6
                    N = N / norm(N);
                else
                    N = [0 0 1];
                end
                
                % Ray-Plane Intersection: dot(N, P - DragStartPos_) = 0
                denom = dot(N, vDir);
                if abs(denom) > 1e-6
                    t = dot(N, obj.DragStartPos_ - pFront) / denom;
                    newPos = pFront + t * vDir;
                else
                    newPos = pFront;
                end
            end
            
            obj.setPosition(newPos);
            if ~isempty(obj.OnMove)
                obj.OnMove(newPos);
            end
        end

        function onMouseUp_(obj)
            obj.Dragging_ = false;
            set(obj.Figure_, ...
                'WindowButtonMotionFcn', obj.OldMotionFcn_, ...
                'WindowButtonUpFcn',     obj.OldReleaseFcn_);
            if ~isempty(obj.OnRelease)
                obj.OnRelease(obj.Position);
            end
        end
    end
end
