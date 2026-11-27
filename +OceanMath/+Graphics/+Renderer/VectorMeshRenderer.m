classdef VectorMeshRenderer < OceanMath.Graphics.Renderer.IRenderer
    % VECTORMESHRENDERER  Mesh-based renderer utilizing MATLAB patch graphics.
    
    properties
        ShadingMode (1,1) string = "Gouraud" % "Flat" or "Gouraud"
        RenderDimension (1,1) string = "3D"  % "3D" (world space) or "2D" (camera plane space)
    end
    
    methods
        function obj = VectorMeshRenderer(options)
            arguments
                options.ShadingMode (1,1) string {mustBeMember(options.ShadingMode, ["Flat", "Gouraud"])} = "Gouraud"
                options.RenderDimension (1,1) string {mustBeMember(options.RenderDimension, ["3D", "2D"])} = "3D"
            end
            obj.ShadingMode = options.ShadingMode;
            obj.RenderDimension = options.RenderDimension;
        end
        
        function handles = render(obj, scene, cameraProjection, options)
            % RENDER Draws the scene objects with custom shading strategies.
            
            arguments
                obj
                scene OceanMath.Graphics.Scene.Scene
                cameraProjection OceanMath.Geometry.CameraProjection
                options.Axes handle = []
                options.LightingStrategy OceanMath.Graphics.Shading.ILightingStrategy
            end
            
            ax = options.Axes;
            if isempty(ax)
                ax = gca;
            end
            
            strategy = options.LightingStrategy;
            
            % Compute camera center in world coordinates
            T_c2w = inv(cameraProjection.Extrinsics);
            camCenter_world = T_c2w(1:3, 4);
            
            handles = [];
            hold(ax, 'on');
            
            for i = 1:numel(scene.Objects)
                sceneObj = scene.Objects{i};
                
                % Get mesh vertices and faces in world coordinates
                [V_world, F] = sceneObj.toMesh();
                numVertices = size(V_world, 2);
                numFaces = size(F, 2);
                
                if strcmp(obj.ShadingMode, "Gouraud")
                    % Compute lighting per vertex
                    CData = zeros(numVertices, 3);
                    for vIdx = 1:numVertices
                        v_pos = V_world(:, vIdx);
                        
                        % Transform world point to local space to evaluate normal
                        pt_local = inv(sceneObj.Pose) * [v_pos; 1];
                        N_local = sceneObj.Geometry.getNormal(pt_local(1:3));
                        
                        % Map normal vector back to world space using H^(-T)
                        R_sub = sceneObj.Pose(1:3, 1:3);
                        N_world = (inv(R_sub)') * N_local(:);
                        N_world = N_world / norm(N_world);
                        
                        % View direction vector
                        viewDir = camCenter_world - v_pos;
                        v_norm = norm(viewDir);
                        if v_norm < 1e-12
                            viewDir = [0; 0; 1];
                        else
                            viewDir = viewDir / v_norm;
                        end
                        
                        % Compute shading color
                        colorRGB = strategy.computeLighting(v_pos, N_world, viewDir, sceneObj.Material, scene.Lights, scene);
                        CData(vIdx, :) = colorRGB(:)';
                    end
                else
                    % Compute lighting per face (Flat Shading)
                    faceColors = zeros(numFaces, 3);
                    for fIdx = 1:numFaces
                        faceVerticesIdx = F(:, fIdx);
                        faceVerts = V_world(:, faceVerticesIdx);
                        centroid = mean(faceVerts, 2);
                        
                        % Compute face normal from face edge vectors in world space
                        v1 = faceVerts(:, 2) - faceVerts(:, 1);
                        v2 = faceVerts(:, 3) - faceVerts(:, 1);
                        N_world = cross(v1, v2);
                        n_norm = norm(N_world);
                        if n_norm < 1e-12
                            N_world = [0; 0; 1];
                        else
                            N_world = N_world / n_norm;
                        end
                        
                        % View direction vector
                        viewDir = camCenter_world - centroid;
                        v_norm = norm(viewDir);
                        if v_norm < 1e-12
                            viewDir = [0; 0; 1];
                        else
                            viewDir = viewDir / v_norm;
                        end
                        
                        % Compute shading color
                        colorRGB = strategy.computeLighting(centroid, N_world, viewDir, sceneObj.Material, scene.Lights, scene);
                        faceColors(fIdx, :) = colorRGB(:)';
                    end
                end
                
                % Prepare plot coordinates
                if strcmp(obj.RenderDimension, "3D")
                    xData = V_world(1, :);
                    yData = V_world(2, :);
                    zData = V_world(3, :);
                else
                    % Project 3D points to 2D camera viewport plane
                    V_proj = cameraProjection.project(V_world);
                    xData = V_proj(1, :);
                    yData = V_proj(2, :);
                    zData = zeros(size(xData));
                end
                
                % Plot polygons using MATLAB patch objects
                if strcmp(obj.ShadingMode, "Gouraud")
                    h = patch(ax, 'Vertices', [xData', yData', zData'], 'Faces', F', ...
                              'FaceVertexCData', CData, 'FaceColor', 'interp', 'EdgeColor', 'none');
                else
                    h = patch(ax, 'Vertices', [xData', yData', zData'], 'Faces', F', ...
                              'FaceVertexCData', faceColors, 'FaceColor', 'flat', 'EdgeColor', 'none');
                end
                handles(end+1) = h; %#ok<AGROW>
            end
        end
    end
end
