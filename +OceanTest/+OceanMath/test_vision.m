classdef test_vision < OceanTest.BaseTestCase
    % TEST_VISION  Tests geometric primitives and transformations for computer vision.
    
    methods (Test)
        function testTransform2D(testCase)
            import OceanMath.Geometry.Transform2D
            
            % Test 1: Identity transforms
            pts = [1, 2, 3; 4, 5, 6];
            H_eye = eye(3);
            pts_out = Transform2D.transformPoints(H_eye, pts);
            testCase.assertMatrixEquals(pts_out, pts, 1e-12, 'Homogeneous identity failed');
            
            % Test 2: Translation
            H_trans = Transform2D.translation(5, -10);
            expected_trans = [6, 7, 8; -6, -5, -4];
            pts_out = Transform2D.transformPoints(H_trans, pts);
            testCase.assertMatrixEquals(pts_out, expected_trans, 1e-12, 'Homogeneous translation failed');
            
            % Test 3: Scale
            H_scale = Transform2D.scale(2, 0.5);
            expected_scale = [2, 4, 6; 2, 2.5, 3];
            pts_out = Transform2D.transformPoints(H_scale, pts);
            testCase.assertMatrixEquals(pts_out, expected_scale, 1e-12, 'Homogeneous scale failed');
            
            % Test 4: Rotation
            H_rot = Transform2D.rotation(90);
            expected_rot = [-4, -5, -6; 1, 2, 3];
            pts_out = Transform2D.transformPoints(H_rot, pts);
            testCase.assertMatrixEquals(pts_out, expected_rot, 1e-12, 'Homogeneous rotation 90 deg failed');
            
            % Test 5: Composite Affine (T * R * Sh * S)
            A = Transform2D.affine(1, 2, 0, 2, 3, 0, 0);
            % translation (1,2), scale (2,3), rotation/shear 0
            % pts = [1; 4] -> scale -> [2; 12] -> translation -> [3; 14]
            pt_single = [1; 4];
            pt_out = Transform2D.transformPoints(A, pt_single);
            testCase.assertMatrixEquals(pt_out, [3; 14], 1e-12, 'Composite affine failed');
            
            % Test 6: Rigid / Euclidean (T * R)
            SE = Transform2D.rigid(5, -10, 90);
            % pts = [1; 4] -> rotate 90 -> [-4; 1] -> translate (5, -10) -> [1; -9]
            pt_out = Transform2D.transformPoints(SE, [1; 4]);
            testCase.assertMatrixEquals(pt_out, [1; -9], 1e-12, 'Euclidean/Rigid transform failed');
            
            % Test 7: Similarity (T * R * S)
            Sim = Transform2D.similarity(5, -10, 90, 2);
            % pts = [1; 4] -> scale (2) -> [2; 8] -> rotate 90 -> [-8; 2] -> translate (5, -10) -> [-3; -8]
            pt_out = Transform2D.transformPoints(Sim, [1; 4]);
            testCase.assertMatrixEquals(pt_out, [-3; -8], 1e-12, 'Similarity transform failed');
            
            % Test 8: Co-vector (Line Equation) Transformation (H^-T)
            H = [1.5, 1.0, 0.0; 0.0, 0.8, 0.0; 0.5, 0.0, 1.2];
            l = [4; -1; 0]; % homogeneous line: 4x - y = 0
            pt = [1; 4; 1];  % homogeneous point: 4(1) - 4 = 0
            pt_w = H * pt;
            l_w = Transform2D.transformLine(H, l);
            dot_prod = l_w' * pt_w;
            testCase.verifyEqual(dot_prod, 0, 'AbsTol', 1e-12, 'Co-vector 2D line transformation failed');
            
            % Test 9: Stretch/Squash (T * S)
            SS = Transform2D.stretchSquash(5, -10, 2, 0.5);
            % pts = [1; 4] -> scale non-uniform -> [2; 2] -> translate -> [7; -8]
            pt_out = Transform2D.transformPoints(SS, [1; 4]);
            testCase.assertMatrixEquals(pt_out, [7; -8], 1e-12, 'Stretch/Squash transform failed');
        end
        
        function testTransform3D(testCase)
            import OceanMath.Geometry.Transform3D
            
            % Test 1: Translation
            pts = [1, 2; 3, 4; 5, 6];
            T_trans = Transform3D.translation(1, -2, 3);
            expected_trans = [2, 3; 1, 2; 8, 9];
            pts_out = Transform3D.transformPoints(T_trans, pts);
            testCase.assertMatrixEquals(pts_out, expected_trans, 1e-12, '3D Translation failed');
            
            % Test 2: Scale
            T_scale = Transform3D.scale(2, 3, 4);
            expected_scale = [2, 4; 9, 12; 20, 24];
            pts_out = Transform3D.transformPoints(T_scale, pts);
            testCase.assertMatrixEquals(pts_out, expected_scale, 1e-12, '3D Scale failed');
            
            % Test 3: Co-vector (Plane Equation) Transformation (T^-T)
            T_mat = [1.5, 1.0, 0.0, 0.0; ...
                     0.0, 0.8, 0.0, 0.0; ...
                     0.5, 0.0, 1.2, 0.0; ...
                     0.0, 0.0, 0.0, 1.0];
            pi_val = [3; -2; 1; -2]; % homogeneous plane: 3x - 2y + z - 2 = 0
            pt = [1; 2; 3; 1];       % homogeneous point: 3(1) - 2(2) + 3 - 2 = 0
            pt_w = T_mat * pt;
            pi_w = Transform3D.transformPlane(T_mat, pi_val);
            dot_prod = pi_w' * pt_w;
            testCase.verifyEqual(dot_prod, 0, 'AbsTol', 1e-12, 'Co-vector 3D plane transformation failed');
            
            % Test 4: Shear
            sh_xy = 0.5; sh_xz = 0.2; sh_yx = 0.0; sh_yz = 0.1; sh_zx = 0.0; sh_zy = 0.0;
            T_shear = Transform3D.shear(sh_xy, sh_xz, sh_yx, sh_yz, sh_zx, sh_zy);
            expected_shear = [2.6; 2.3; 3];
            pts_out = Transform3D.transformPoints(T_shear, [1; 2; 3]);
            testCase.assertMatrixEquals(pts_out, expected_shear, 1e-12, '3D Shear failed');
            
            % Test 5: Rigid (Euclidean)
            R_mat = [0, -1, 0; 1, 0, 0; 0, 0, 1]; % 90 deg around Z
            T_rigid = Transform3D.rigid(5, -10, 20, R_mat);
            expected_rigid = [3; -9; 23];
            pts_out = Transform3D.transformPoints(T_rigid, [1; 2; 3]);
            testCase.assertMatrixEquals(pts_out, expected_rigid, 1e-12, '3D Rigid failed');
            
            % Test 6: Similarity
            T_sim = Transform3D.similarity(5, -10, 20, R_mat, 2);
            expected_sim = [1; -8; 26];
            pts_out = Transform3D.transformPoints(T_sim, [1; 2; 3]);
            testCase.assertMatrixEquals(pts_out, expected_sim, 1e-12, '3D Similarity failed');
            
            % Test 7: Affine
            A_mat = [2, 0.5, 0; 0, 1, 0; 0, 0, 0.5];
            T_affine = Transform3D.affine(A_mat, 1, 2, 3);
            expected_affine = [4; 4; 5];
            pts_out = Transform3D.transformPoints(T_affine, [1; 2; 4]);
            testCase.assertMatrixEquals(pts_out, expected_affine, 1e-12, '3D Affine failed');
            
            % Test 8: Projective
            H_mat = eye(4); H_mat(4,3) = 0.5; H_mat(4,4) = 2.0;
            T_proj = Transform3D.projective(H_mat);
            testCase.verifyEqual(T_proj(4,4), 1.0, 'AbsTol', 1e-12, '3D Projective normalization failed');
            
            % Test 9: OOP Quaternion Rotation Strategy & Transform3D Integration (Phase 1)
            import OceanMath.Geometry.Rotations.QuaternionRotation3D
            
            q_rot_y = [sqrt(0.5), 0, sqrt(0.5), 0]; % 90 deg around Y
            rotObj = QuaternionRotation3D(q_rot_y);
            
            % Test polymorphic rigid transform: translation (5, -10, 20) + rotation
            T_rigid = Transform3D.rigid(5, -10, 20, rotObj);
            % [1; 2; 3] -> rotate 90 deg Y -> [3; 2; -1] -> translate -> [8; -8; 19]
            expected_rigid = [8; -8; 19];
            pts_out = Transform3D.transformPoints(T_rigid, [1; 2; 3]);
            testCase.assertMatrixEquals(pts_out, expected_rigid, 1e-12, 'OOP 3D Rigid Transform failed');
            
            % Test polymorphic similarity transform
            T_sim = Transform3D.similarity(5, -10, 20, rotObj, 2);
            % [1; 2; 3] -> scale 2 -> [2; 4; 6] -> rotate 90 Y -> [6; 4; -2] -> translate -> [11; -6; 18]
            expected_sim = [11; -6; 18];
            pts_out = Transform3D.transformPoints(T_sim, [1; 2; 3]);
            testCase.assertMatrixEquals(pts_out, expected_sim, 1e-12, 'OOP 3D Similarity Transform failed');
            
            % Test SLERP interpolation between two OOP Quaternions
            rotObj1 = QuaternionRotation3D([1, 0, 0, 0]);
            rotObj2 = QuaternionRotation3D([0, 1, 0, 0]);
            rotObj_half = rotObj1.interpolate(rotObj2, 0.5);
            q_res = rotObj_half.toQuat();
            testCase.verifyEqual(q_res(1), sqrt(0.5), 'AbsTol', 1e-6);
            testCase.verifyEqual(q_res(2), sqrt(0.5), 'AbsTol', 1e-6);
            
            % Test 10: OOP Euler, Axis-Angle, Exp-Map and Matrix Rotation Strategies (Phase 2)
            import OceanMath.Geometry.Rotations.EulerRotation3D
            import OceanMath.Geometry.Rotations.AxisAngleRotation3D
            import OceanMath.Geometry.Rotations.ExpMapRotation3D
            import OceanMath.Geometry.Rotations.MatrixRotation3D
            
            R_expected = [0, 0, 1; 0, 1, 0; -1, 0, 0]; % 90 deg around Y
            
            % 1. Test EulerRotation3D
            rotEuler = EulerRotation3D(0, 90, 0);
            testCase.assertMatrixEquals(rotEuler.toMatrix(), R_expected, 1e-12, 'EulerRotation3D matrix mismatch');
            [r_out, p_out, y_out] = rotEuler.toEuler();
            testCase.verifyEqual([r_out, p_out, y_out], [0, 90, 0], 'AbsTol', 1e-12, 'EulerRotation3D toEuler mismatch');
            
            % 2. Test AxisAngleRotation3D
            rotAA = AxisAngleRotation3D([0, 1, 0], 90);
            testCase.assertMatrixEquals(rotAA.toMatrix(), R_expected, 1e-12, 'AxisAngleRotation3D matrix mismatch');
            [ax_out, th_out] = rotAA.toAxisAngle();
            testCase.assertMatrixEquals(ax_out, [0, 1, 0], 1e-12, 'AxisAngleRotation3D axis mismatch');
            testCase.verifyEqual(th_out, 90, 'AbsTol', 1e-12, 'AxisAngleRotation3D angle mismatch');
            
            % 3. Test ExpMapRotation3D (Rodrigues' Formula verification)
            rotExp = ExpMapRotation3D([0; pi/2; 0]);
            testCase.assertMatrixEquals(rotExp.toMatrix(), R_expected, 1e-12, 'ExpMapRotation3D matrix mismatch (Rodrigues formula failed)');
            testCase.assertMatrixEquals(rotExp.toExpMap(), [0; pi/2; 0], 1e-12, 'ExpMapRotation3D toExpMap mismatch');
            
            % 4. Test MatrixRotation3D
            rotMat = MatrixRotation3D(R_expected);
            testCase.assertMatrixEquals(rotMat.toMatrix(), R_expected, 1e-12, 'MatrixRotation3D toMatrix mismatch');
            
            % 5. Test cross-strategy interpolations polymorphically
            rotObjA = EulerRotation3D(0, 0, 0);
            rotObjB = AxisAngleRotation3D([0, 1, 0], 90);
            
            % Interpolate Euler (0) to AxisAngle (90 Y)
            rotObj_interp = rotObjA.interpolate(rotObjB, 0.5);
            [r_i, p_i, y_i] = rotObj_interp.toEuler();
            testCase.verifyEqual([r_i, p_i, y_i], [0, 45, 0], 'AbsTol', 1e-12, 'Cross-strategy Euler interpolation failed');
        end
        
        function testRotation3D(testCase)
            import OceanMath.Geometry.Rotation3D
            
            % Test 1: Euler conversions ZYX
            roll = 10; pitch = -20; yaw = 45;
            R = Rotation3D.eulerToMatrix(roll, pitch, yaw);
            [r_out, p_out, y_out] = Rotation3D.matrixToEuler(R);
            testCase.verifyEqual(r_out, roll, 'AbsTol', 1e-6);
            testCase.verifyEqual(p_out, pitch, 'AbsTol', 1e-6);
            testCase.verifyEqual(y_out, yaw, 'AbsTol', 1e-6);
            
            % Test 2: Quaternion conversions
            q = [sqrt(0.5), 0, sqrt(0.5), 0]; % 90 degrees around Y axis
            R_q = Rotation3D.quatToMatrix(q);
            q_out = Rotation3D.matrixToQuat(R_q);
            % Quaternions can have flipped signs (double cover)
            testCase.verifyTrue(norm(q - q_out) < 1e-6 || norm(q + q_out) < 1e-6, ...
                'Quaternion matrix round-trip failed');
                
            % Test 3: Axis-Angle conversions
            axis_vec = [1, 1, 1];
            axis_vec = axis_vec / norm(axis_vec);
            theta = 60;
            R_aa = Rotation3D.axisAngleToMatrix(axis_vec, theta);
            [ax_out, th_out] = Rotation3D.matrixToAxisAngle(R_aa);
            testCase.verifyEqual(th_out, theta, 'AbsTol', 1e-6);
            testCase.assertMatrixEquals(ax_out, axis_vec, 1e-6, 'Axis recovery failed');
            
            % Test 4: Exponential Map (Lie algebra so(3))
            v = [0.1; -0.2; 0.5];
            R_exp = Rotation3D.expMapToMatrix(v);
            v_out = Rotation3D.matrixToExpMap(R_exp);
            testCase.assertMatrixEquals(v_out, v, 1e-6, 'Exponential map round-trip failed');
            
            % Test 5: Slerp
            q1 = [1, 0, 0, 0];
            q2 = [0, 1, 0, 0];
            q_half = Rotation3D.slerp(q1, q2, 0.5);
            testCase.verifyEqual(q_half(1), sqrt(0.5), 'AbsTol', 1e-6);
            testCase.verifyEqual(q_half(2), sqrt(0.5), 'AbsTol', 1e-6);
        end
        
        function testCameraProjection(testCase)
            import OceanMath.Geometry.CameraProjection
            
            K = [400, 0, 320; 0, 400, 240; 0, 0, 1];
            T_w2c = eye(4);
            proj = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c);
            
            % Test 1: Pinhole projection for point on z axis
            pt3D = [0; 0; 10]; % directly in front of camera, distance 10
            [pt2D, inFront] = proj.project(pt3D);
            testCase.verifyTrue(inFront, 'Point should be in front');
            testCase.assertMatrixEquals(pt2D, [320; 240], 1e-6, ...
                'Projection of (0,0,Z) must equal principal point (cx, cy)');
                
            % Test 2: Point off axis
            pt3D = [1; -1; 5]; % X_c = 1, Y_c = -1, Z_c = 5
            % u = X_c/Z_c = 0.2 -> x_img = fx * u + cx = 400 * 0.2 + 320 = 400
            % v = Y_c/Z_c = -0.2 -> y_img = fy * v + cy = 400 * (-0.2) + 240 = 160
            [pt2D, inFront] = proj.project(pt3D);
            testCase.verifyTrue(inFront, 'Point should be in front');
            testCase.assertMatrixEquals(pt2D, [400; 160], 1e-6, 'Off-axis projection failed');
            
            % Test 3: Orthographic projection
            projOrtho = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c, 'ProjectionType', 'Orthographic');
            pt2D_ortho = projOrtho.project(pt3D);
            % x = fx * X_c + cx = 400 * 1 + 320 = 720
            % y = fy * Y_c + cy = 400 * (-1) + 240 = -160
            testCase.assertMatrixEquals(pt2D_ortho, [720; -160], 1e-6, 'Orthographic projection failed');
            
            % Test 4: Weak Perspective projection
            projWeak = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c, 'ProjectionType', 'WeakPerspective');
            pts3D_multi = [1, 2; -1, -2; 5, 15]; % Z_avg = (5 + 15)/2 = 10
            pts2D_weak = projWeak.project(pts3D_multi);
            % Pt 1: x = (400 * 1)/10 + 320 = 360, y = (400 * -1)/10 + 240 = 200
            % Pt 2: x = (400 * 2)/10 + 320 = 400, y = (400 * -2)/10 + 240 = 160
            testCase.assertMatrixEquals(pts2D_weak, [360, 400; 200, 160], 1e-6, 'WeakPerspective projection failed');
            
            % Test 5: Pure Orthographic projection (x = X_c, y = Y_c)
            projPure = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c, 'ProjectionType', 'PureOrthographic');
            pt2D_pure = projPure.project(pt3D);
            testCase.assertMatrixEquals(pt2D_pure, [1; -1], 1e-6, 'PureOrthographic projection failed');
            
            % Test 6: Paraperspective projection
            projPara = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c, 'ProjectionType', 'Paraperspective');
            pts2D_para = projPara.project(pts3D_multi);
            % Centroid is [1.5; -1.5; 10]
            % Pt 1: X_para = 1 - 1.5 * 5/10 = 0.25, Y_para = -1 - (-1.5) * 5/10 = -0.25
            %       x = 400 * 0.25/10 + 320 = 330, y = 400 * -0.25/10 + 240 = 230
            % Pt 2: X_para = 2 - 1.5 * 15/10 = -0.25, Y_para = -2 - (-1.5) * 15/10 = 0.25
            %       x = 400 * -0.25/10 + 320 = 310, y = 400 * 0.25/10 + 240 = 250
            testCase.assertMatrixEquals(pts2D_para, [330, 310; 230, 250], 1e-6, 'Paraperspective projection failed');
        end
        
        function testLensDistortion(testCase)
            import OceanMath.Geometry.LensDistortion
            
            % Zero distortion coefficients
            dist = LensDistortion('Radial', [0, 0, 0], 'Tangential', [0, 0]);
            ptsNorm = [0.5, -0.2; 0.3, 0.4];
            ptsDist = dist.distort(ptsNorm);
            testCase.assertMatrixEquals(ptsDist, ptsNorm, 1e-12, 'Zero distortion failed');
            
            % Radial and tangential distortions
            dist = LensDistortion('Radial', [0.1, -0.05, 0.01], 'Tangential', [0.02, -0.03]);
            ptsDist = dist.distort(ptsNorm);
            testCase.verifySize(ptsDist, [2, 2], 'Distortion size mismatch');
            
            % Distort in image coordinate space
            K = [500, 0, 320; 0, 500, 240; 0, 0, 1];
            ptsImg = [300, 400; 200, 300];
            ptsImgWarped = dist.distortImageCoordinates(ptsImg, K);
            testCase.verifySize(ptsImgWarped, [2, 2], 'Image space distortion size mismatch');
        end
    end
end
