classdef test_vision_lighting < OceanTest.BaseTestCase
    % TEST_VISION_LIGHTING  Tests geometry, scene and intersection math for 3D Room simulation.
    
    methods (Test)
        function testPlaneIntersection(testCase)
            import OceanMath.Vision.Geometry.PlaneGeometry
            
            % Infinite plane parallel to XY plane at Z = 5
            plane = PlaneGeometry('Point', [0; 0; 5], 'Normal', [0; 0; 1]);
            
            % Ray pointing straight up along Z axis from origin
            rayO = [0; 0; 0];
            rayD = [0; 0; 1];
            [t, N] = plane.intersect(rayO, rayD);
            testCase.verifyEqual(t, 5, 'AbsTol', 1e-12);
            testCase.assertMatrixEquals(N, [0; 0; -1], 1e-12, 'Normal should face incoming ray');
            
            % Parallel ray (should be NaN)
            [t_par, ~] = plane.intersect(rayO, [1; 0; 0]);
            testCase.verifyTrue(isnan(t_par));
            
            % Ray pointing away (t should be NaN)
            [t_back, ~] = plane.intersect(rayO, [0; 0; -1]);
            testCase.verifyTrue(isnan(t_back));
            
            % Finite plane segment (rectangle [-2,2] x [-3,3] at Z=5)
            finitePlane = PlaneGeometry('Point', [0; 0; 5], 'Normal', [0; 0; 1], ...
                                        'IsFinite', true, 'Width', 2.0, 'Height', 3.0);
                                    
            % Ray hitting within bounds
            [t_hit, ~] = finitePlane.intersect([0.5; -1.0; 0], [0; 0; 1]);
            testCase.verifyEqual(t_hit, 5, 'AbsTol', 1e-12);
            
            % Ray hitting outside bounds (should be NaN)
            [t_miss, ~] = finitePlane.intersect([3.5; 0; 0], [0; 0; 1]);
            testCase.verifyTrue(isnan(t_miss));
        end
        
        function testSphereIntersection(testCase)
            import OceanMath.Vision.Geometry.SphereGeometry
            
            % Sphere centered at [0; 0; 10] with radius 2
            sphere = SphereGeometry('Center', [0; 0; 10], 'Radius', 2.0);
            
            % Ray pointing straight along Z
            rayO = [0; 0; 0];
            rayD = [0; 0; 1];
            [t, N] = sphere.intersect(rayO, rayD);
            % Distance to center is 10, radius is 2, closest hit is 10 - 2 = 8
            testCase.verifyEqual(t, 8.0, 'AbsTol', 1e-12);
            % Normal at hit point [0; 0; 8] should be [0; 0; -1]
            testCase.assertMatrixEquals(N, [0; 0; -1], 1e-12);
            
            % Ray missing sphere
            [t_miss, ~] = sphere.intersect(rayO, [1; 0; 0]);
            testCase.verifyTrue(isnan(t_miss));
        end
        
        function testTransformedSceneObject(testCase)
            import OceanMath.Vision.Geometry.SphereGeometry
            import OceanMath.Graphics.Scene.SceneObject
            import OceanMath.Graphics.Scene.Material
            import OceanMath.Geometry.Transform3D
            
            % Base sphere at local origin, radius 1
            sphereGeom = SphereGeometry('Center', [0; 0; 0], 'Radius', 1.0);
            material = Material();
            
            % Transform: Scale 2, Translate Z by 10
            T = Transform3D.translation(0, 0, 10) * Transform3D.scale(2, 2, 2);
            obj = SceneObject(sphereGeom, material, 'Pose', T);
            
            % Ray along Z axis
            [t, N] = obj.intersect([0; 0; 0], [0; 0; 1]);
            % Sphere scaled to radius 2, center at 10. Expected hit at Z = 10 - 2 = 8.
            testCase.verifyEqual(t, 8.0, 'AbsTol', 1e-12);
            testCase.assertMatrixEquals(N, [0; 0; -1], 1e-12);
            
            % Mesh test
            [V, F] = obj.toMesh();
            % Vertices should all be centered around [0; 0; 10] with distance 2
            v_centered = V - repmat([0; 0; 10], 1, size(V, 2));
            v_radii = sqrt(sum(v_centered.^2, 1));
            testCase.verifyEqual(v_radii, repmat(2.0, 1, size(V, 2)), 'AbsTol', 1e-12);
        end
        
        function testSceneIntersection(testCase)
            import OceanMath.Vision.Geometry.SphereGeometry
            import OceanMath.Vision.Geometry.PlaneGeometry
            import OceanMath.Graphics.Scene.SceneObject
            import OceanMath.Graphics.Scene.Material
            import OceanMath.Graphics.Scene.Scene
            import OceanMath.Geometry.Transform3D
            
            scene = Scene();
            mat = Material();
            
            % Sphere 1: Center [0; 0; 10], radius 1
            s1 = SceneObject(SphereGeometry('Center', [0; 0; 0], 'Radius', 1.0), mat, ...
                             'Pose', Transform3D.translation(0, 0, 10));
            scene.addObject(s1);
            
            % Sphere 2: Center [0; 0; 5], radius 1 (closer!)
            s2 = SceneObject(SphereGeometry('Center', [0; 0; 0], 'Radius', 1.0), mat, ...
                             'Pose', Transform3D.translation(0, 0, 5));
            scene.addObject(s2);
            
            % Ray should hit s2 (closer sphere)
            [closestObj, t, N] = scene.intersect([0; 0; 0], [0; 0; 1]);
            testCase.verifyEqual(closestObj, s2);
            testCase.verifyEqual(t, 4.0, 'AbsTol', 1e-12);
            testCase.assertMatrixEquals(N, [0; 0; -1], 1e-12);
        end
        
        function testLightingAndShading(testCase)
            import OceanMath.Graphics.Lighting.AmbientLight
            import OceanMath.Graphics.Lighting.PointLight
            import OceanMath.Graphics.Lighting.DirectionalLight
            import OceanMath.Graphics.Lighting.SpotLight
            import OceanMath.Graphics.Scene.Material
            import OceanMath.Graphics.Shading.LambertianStrategy
            import OceanMath.Graphics.Shading.PhongStrategy
            import OceanMath.Graphics.Shading.BlinnPhongStrategy
            
            % Setup Point on surface [0; 0; 0], normal [0; 0; 1], viewing along Z
            point = [0; 0; 0];
            normal = [0; 0; 1];
            viewDir = [0; 0; 1];
            
            % Red material: Ka = 0.1, Kd = 0.8, Ks = 0.5, Shininess = 10
            mat = Material('Ambient', [0.1; 0; 0], 'Diffuse', [0.8; 0; 0], 'Specular', [0.5; 0.5; 0.5], 'Shininess', 10);
            
            % 1. Test Ambient Light
            ambient = AmbientLight('Intensity', [0.5; 0.5; 0.5]);
            [intensity, dirToLight, dist] = ambient.getLightAt(point);
            testCase.assertMatrixEquals(intensity, [0.5; 0.5; 0.5], 1e-12);
            testCase.verifyTrue(isinf(dist));
            
            % 2. Test Point Light with attenuation
            pointLight = PointLight('Position', [0; 0; 2], 'Intensity', [1; 1; 1], ...
                                    'ConstantAttenuation', 1, 'LinearAttenuation', 0.5, 'QuadraticAttenuation', 0.25);
            [intensity, dirToLight, dist] = pointLight.getLightAt(point);
            testCase.verifyEqual(dist, 2.0, 'AbsTol', 1e-12);
            testCase.assertMatrixEquals(dirToLight, [0; 0; 1], 1e-12);
            % Attenuation: 1 + 0.5 * 2 + 0.25 * 4 = 1 + 1 + 1 = 3
            testCase.assertMatrixEquals(intensity, [1; 1; 1] / 3.0, 1e-12);
            
            % 3. Test Spot Light cone cutoff
            spotLight_in = SpotLight('Position', [0; 0; 2], 'Direction', [0; 0; -1], 'ConeAngle', 30, 'Intensity', [1; 1; 1]);
            [intensity_in, ~, ~] = spotLight_in.getLightAt(point);
            testCase.verifyGreaterThan(norm(intensity_in), 0);
            
            spotLight_out = SpotLight('Position', [2; 0; 2], 'Direction', [0; 0; -1], 'ConeAngle', 15, 'Intensity', [1; 1; 1]);
            [intensity_out, ~, ~] = spotLight_out.getLightAt(point);
            testCase.assertMatrixEquals(intensity_out, [0; 0; 0], 1e-12);
            
            % 4. Test Lambertian Shading
            lambert = LambertianStrategy();
            lights = {ambient, spotLight_in};
            c_lamb = lambert.computeLighting(point, normal, viewDir, mat, lights);
            % Expected red diffuse + ambient: Red component should be non-zero, G and B should be zero
            testCase.verifyGreaterThan(c_lamb(1), 0.05);
            testCase.verifyEqual(c_lamb(2), 0.0, 'AbsTol', 1e-12);
            testCase.verifyEqual(c_lamb(3), 0.0, 'AbsTol', 1e-12);
            
            % 5. Test Phong vs Blinn-Phong Specular contribution
            phong = PhongStrategy();
            blinn = BlinnPhongStrategy();
            
            % Light from side so specular highlight changes depending on R . V or N . H
            lightSide = PointLight('Position', [1; 0; 1], 'Intensity', [1; 1; 1]);
            c_phong = phong.computeLighting(point, normal, viewDir, mat, {lightSide});
            c_blinn = blinn.computeLighting(point, normal, viewDir, mat, {lightSide});
            testCase.verifyGreaterThan(c_phong(1), 0.0);
            testCase.verifyGreaterThan(c_blinn(1), 0.0);
        end
        
        function testRenderers(testCase)
            import OceanMath.Vision.Geometry.SphereGeometry
            import OceanMath.Graphics.Scene.SceneObject
            import OceanMath.Graphics.Scene.Material
            import OceanMath.Graphics.Scene.Scene
            import OceanMath.Graphics.Lighting.AmbientLight
            import OceanMath.Graphics.Lighting.PointLight
            import OceanMath.Graphics.Shading.BlinnPhongStrategy
            import OceanMath.Graphics.Renderer.VectorMeshRenderer
            import OceanMath.Graphics.Renderer.RayTraceRenderer
            import OceanMath.Geometry.CameraProjection
            import OceanMath.Geometry.Transform3D
            
            % Setup basic scene
            scene = Scene();
            mat = Material('Ambient', [0.1;0.1;0.1], 'Diffuse', [0.8;0;0], 'Specular', [0.5;0.5;0.5]);
            sphere = SceneObject(SphereGeometry('Center', [0; 0; 0], 'Radius', 1.0), mat, ...
                                 'Pose', Transform3D.translation(0, 0, 5));
            scene.addObject(sphere);
            scene.addLight(AmbientLight());
            scene.addLight(PointLight('Position', [0; 2; 2], 'Intensity', [1; 1; 1]));
            
            % Setup camera
            K = [100, 0, 80; 0, 100, 60; 0, 0, 1]; % 160x120 camera
            T_w2c = eye(4);
            camera = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c);
            
            strategy = BlinnPhongStrategy();
            
            % 1. Test Vector Mesh Renderer (3D Mode)
            vector3D = VectorMeshRenderer('RenderDimension', '3D', 'ShadingMode', 'Gouraud');
            fig = figure('Visible', 'off');
            ax = axes(fig);
            handles3D = vector3D.render(scene, camera, 'Axes', ax, 'LightingStrategy', strategy);
            testCase.verifyNotEmpty(handles3D);
            testCase.verifyEqual(numel(handles3D), 1);
            delete(fig);
            
            % 2. Test Vector Mesh Renderer (2D Mode)
            vector2D = VectorMeshRenderer('RenderDimension', '2D', 'ShadingMode', 'Flat');
            fig = figure('Visible', 'off');
            ax = axes(fig);
            handles2D = vector2D.render(scene, camera, 'Axes', ax, 'LightingStrategy', strategy);
            testCase.verifyNotEmpty(handles2D);
            testCase.verifyEqual(numel(handles2D), 1);
            delete(fig);
            
            % 3. Test Ray Trace Renderer
            rayTracer = RayTraceRenderer('Resolution', [40, 30], 'MaxDepth', 1);
            img = rayTracer.render(scene, camera, 'LightingStrategy', strategy);
            testCase.verifyEqual(size(img), [30, 40, 3]);
            testCase.verifyTrue(all(img(:) >= 0) && all(img(:) <= 1));
        end
    end
end
