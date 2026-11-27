classdef LambertianStrategy < OceanMath.Graphics.Shading.ILightingStrategy
    % LAMBERTIANSTRATEGY  Diffuse-only (Lambertian) reflection model with shadows.
    
    methods
        function color = computeLighting(~, point, normal, ~, material, lights, scene)
            arguments
                ~
                point (3,1) double
                normal (3,1) double
                ~
                material OceanMath.Graphics.Scene.Material
                lights cell
                scene = []
            end
            
            color = [0; 0; 0];
            
            % Ensure unit normal
            N = normal(:) / norm(normal);
            
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
                        % Diffuse term: Kd * max(0, N . L) * I
                        cosTheta = max(0.0, dot(N, L_dir(:)));
                        color = color + material.Diffuse(:) .* cosTheta .* L_intensity(:);
                    end
                end
            end
            
            % Output boundary clamping
            color = max(0.0, min(1.0, color));
        end
    end
end
