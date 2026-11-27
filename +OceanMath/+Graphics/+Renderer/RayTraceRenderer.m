classdef RayTraceRenderer < OceanMath.Graphics.Renderer.IRenderer
    % RAYTRACERENDERER  Pixel-by-pixel ray tracing renderer with reflections.
    
    properties
        Resolution (1,2) double = [160, 120]  % [width, height]
        BackgroundColor (3,1) double = [0.11; 0.11; 0.13] % Default background RGB
        MaxDepth (1,1) double = 1            % Reflection recursion depth limit
    end
    
    methods
        function obj = RayTraceRenderer(options)
            arguments
                options.Resolution (1,2) double = [160, 120]
                options.BackgroundColor (3,1) double = [0.11; 0.11; 0.13]
                options.MaxDepth (1,1) double = 1
            end
            obj.Resolution = options.Resolution;
            obj.BackgroundColor = options.BackgroundColor;
            obj.MaxDepth = options.MaxDepth;
        end
        
        function img = render(obj, scene, cameraProjection, options)
            % RENDER Render the scene pixel-by-pixel to an RGB image.
            
            arguments
                obj
                scene OceanMath.Graphics.Scene.Scene
                cameraProjection OceanMath.Geometry.CameraProjection
                options.LightingStrategy OceanMath.Graphics.Shading.ILightingStrategy
            end
            
            strategy = options.LightingStrategy;
            
            W = obj.Resolution(1);
            H = obj.Resolution(2);
            
            img = zeros(H, W, 3);
            
            % Compute camera parameters
            K = cameraProjection.Intrinsics;
            T_w2c = cameraProjection.Extrinsics;
            T_c2w = inv(T_w2c);
            O_world = T_c2w(1:3, 4);
            R_c2w = T_c2w(1:3, 1:3);
            
            K_inv = inv(K);
            
            % Sub-pixel scaling factors to translate viewport resolution to intrinsic resolution
            widthScale = (K(1,3) * 2.0) / W;
            heightScale = (K(2,3) * 2.0) / H;
            
            % Loop through each pixel in the viewport
            for v = 1:H
                for u = 1:W
                    % Scale viewport indices to matched camera coordinates
                    u_scale = (u - 0.5) * widthScale;
                    v_scale = (v - 0.5) * heightScale;
                    
                    % Unproject coordinate to camera space
                    p_hom = [u_scale; v_scale; 1];
                    p_cam = K_inv * p_hom;
                    d_cam = p_cam / norm(p_cam);
                    
                    % Transform to world direction
                    D_world = R_c2w * d_cam;
                    D_world = D_world / norm(D_world);
                    
                    % Trace ray to evaluate color
                    pixelColor = obj.traceRay(scene, O_world, D_world, strategy, 0);
                    img(v, u, :) = pixelColor;
                end
            end
        end
    end
    
    methods (Access = private)
        function color = traceRay(obj, scene, rayO, rayD, strategy, depth)
            % TRACERAY Recursively trace a ray into the scene.
            color = obj.BackgroundColor;
            
            [closestObj, tMin, N] = scene.intersect(rayO, rayD);
            if isnan(tMin)
                return;
            end
            
            % Intersection point
            P_hit = rayO(:) + tMin * rayD(:);
            
            % View vector (pointing from hit point to camera)
            viewDir = -rayD(:);
            
            % Compute local illumination (ambient, diffuse, specular)
            localColor = strategy.computeLighting(P_hit, N, viewDir, closestObj.Material, scene.Lights, scene);
            color = localColor(:);
            
            % Specular mirror reflection recursion
            Kr = closestObj.Material.Reflectivity;
            if depth < obj.MaxDepth && any(Kr > 0.01)
                % Reflection ray direction: R = D - 2 * (D . N) * N
                R_dir = rayD(:) - 2.0 * dot(rayD(:), N) * N;
                R_dir = R_dir / norm(R_dir);
                
                % Offset origin slightly along normal to prevent self-intersection acne
                refO = P_hit + 1e-4 * N;
                
                % Trace reflection ray
                reflectedColor = obj.traceRay(scene, refO, R_dir, strategy, depth + 1);
                
                % Blend local color and recursive reflection color
                color = (1.0 - Kr(:)) .* localColor(:) + Kr(:) .* reflectedColor(:);
            end
            
            % Clamping
            color = max(0.0, min(1.0, color));
        end
    end
end
