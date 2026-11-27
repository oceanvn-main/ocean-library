classdef PhongStrategy < OceanMath.Graphics.Shading.ILightingStrategy
    % PHONGSTRATEGY  Phong reflection model (ambient + diffuse + specular).
    
    methods
        function color = computeLighting(~, point, normal, viewDir, material, lights, scene)
            arguments
                ~
                point (3,1) double
                normal (3,1) double
                viewDir (3,1) double
                material OceanMath.Graphics.Scene.Material
                lights cell
                scene = []
            end
            
            color = [0; 0; 0];
            
            % Normalize vectors
            N = normal(:) / norm(normal);
            V = viewDir(:) / norm(viewDir);
            
            for i = 1:numel(lights)
                light = lights{i};
                [L_intensity, L_dir, L_dist] = light.getLightAt(point);
                
                if isa(light, 'OceanMath.Graphics.Lighting.AmbientLight')
                    % Ambient term: Ka * Ia
                    color = color + material.Ambient(:) .* L_intensity(:);
                else
                    % Shadow ray check (occlusion check)
                    isShadowed = false;
                    if ~isempty(scene) && ~isnan(L_dist) && L_dist > 1e-4
                        % Offset origin along normal vector to prevent self-intersection acne
                        shadowO = point(:) + 1e-4 * N;
                        [~, t_shadow, ~] = scene.intersect(shadowO, L_dir(:));
                        if ~isnan(t_shadow) && t_shadow < L_dist
                            isShadowed = true;
                        end
                    end
                    
                    if ~isShadowed
                        % Ensure light is on the front side of surface
                        cosTheta = dot(N, L_dir(:));
                        if cosTheta > 0
                            % Diffuse term: Kd * (N . L) * I
                            diffuse = material.Diffuse(:) .* cosTheta .* L_intensity(:);
                            
                            % Compute reflection vector R = 2 * (N . L) * N - L
                            R = 2.0 * cosTheta * N - L_dir(:);
                            R = R / norm(R);
                            
                            % Specular term: Ks * max(0, R . V)^alpha * I
                            cosAlpha = max(0.0, dot(R, V));
                            specular = material.Specular(:) .* (cosAlpha^material.Shininess) .* L_intensity(:);
                            
                            color = color + diffuse + specular;
                        end
                    end
                end
            end
            
            % Output boundary clamping
            color = max(0.0, min(1.0, color));
        end
    end
end
