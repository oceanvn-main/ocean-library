classdef ShapesGenerator < OceanData.Generators.IDataGenerator
    % SHAPESGENERATOR Creates non-linear geometric manifolds.
    % Essential for benchmarking Kernel methods and Neural Networks
    % which must learn non-linear decision boundaries.
    
    properties
        Shape   (1,1) string {mustBeMember(Shape, ["Circles", "ConcentricRings", "XOR", "Moons", "Spirals"])} = "Circles"
        Noise   (1,1) double = 0.1
        Classes (1,1) double {mustBePositive, mustBeInteger} = 2
        Winding (1,1) double = 4 * pi
        R_inner (1,1) double = 0.5   % inner ring radius (ConcentricRings)
        R_outer (1,1) double = 1.5   % outer ring radius (ConcentricRings)
    end
    
    methods
        function obj = ShapesGenerator(opts)
            arguments
                opts.N        (1,1) double = 300
                opts.Shape    (1,1) string = "Circles"
                opts.Noise    (1,1) double = 0.1
                opts.Classes  (1,1) double = 2
                opts.Winding  (1,1) double = 4 * pi
                opts.R_inner  (1,1) double = 0.5
                opts.R_outer  (1,1) double = 1.5
            end
            obj.N       = opts.N;
            obj.Shape   = opts.Shape;
            obj.Noise   = opts.Noise;
            obj.Classes = opts.Classes;
            obj.Winding = opts.Winding;
            obj.R_inner = opts.R_inner;
            obj.R_outer = opts.R_outer;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            switch obj.Shape
                case "Circles"
                    % Two Concentric Circles
                    n_inner = floor(obj.N / 2);
                    n_outer = obj.N - n_inner;
                    
                    % Inner Circle (Class 0)
                    theta1 = rand(n_inner, 1) * 2 * pi;
                    r1 = randn(n_inner, 1) * obj.Noise; % Radius ~ 0
                    X1 = [r1 .* cos(theta1), r1 .* sin(theta1)];
                    
                    % Outer Circle (Class 1)
                    theta2 = rand(n_outer, 1) * 2 * pi;
                    r2 = 1.0 + randn(n_outer, 1) * obj.Noise; % Radius ~ 1
                    X2 = [r2 .* cos(theta2), r2 .* sin(theta2)];
                    
                    X = [X1; X2];
                    Y = [zeros(n_inner, 1); ones(n_outer, 1)];
                    
                case "ConcentricRings"
                    % Two concentric ring-shaped classes (ESL Figure 13.14)
                    n_inner = floor(obj.N / 2);
                    n_outer = obj.N - n_inner;
                    
                    % Inner ring (Class 0) — radius ~ R_inner
                    theta1 = rand(n_inner, 1) * 2 * pi;
                    r1 = obj.R_inner + randn(n_inner, 1) * obj.Noise;
                    X1 = [r1 .* cos(theta1), r1 .* sin(theta1)];
                    
                    % Outer ring (Class 1) — radius ~ R_outer
                    theta2 = rand(n_outer, 1) * 2 * pi;
                    r2 = obj.R_outer + randn(n_outer, 1) * obj.Noise;
                    X2 = [r2 .* cos(theta2), r2 .* sin(theta2)];
                    
                    X = [X1; X2];
                    Y = [zeros(n_inner, 1); ones(n_outer, 1)];
                    
                case "XOR"
                    % 4 Quadrants checkerboard pattern
                    X = rand(obj.N, 2) * 2 - 1; % Uniform [-1, 1]
                    Y = xor(X(:,1) > 0, X(:,2) > 0); % Logic XOR
                    % Add Jitter Noise
                    X = X + randn(obj.N, 2) * obj.Noise;
                    
                case "Moons"
                    % Two interleaving semi-circles (Simplified)
                    t = linspace(0, pi, floor(obj.N/2))';
                    X1 = [cos(t), sin(t)];
                    X2 = [1 - cos(t), 0.5 - sin(t)];
                    X = [X1; X2] + randn(obj.N, 2) * obj.Noise;
                    Y = [zeros(floor(obj.N/2), 1); ones(floor(obj.N/2), 1)];
                    
                case "Spirals"
                    % Multi-class interleaving spirals
                    n_per = floor(obj.N / obj.Classes);
                    X = []; Y = [];
                    
                    for k = 1:obj.Classes
                        theta = linspace(0.5 * pi, obj.Winding, n_per)';
                        r = theta;
                        offset = (k - 1) * (2 * pi / obj.Classes);
                        
                        Xk = [r .* cos(theta + offset), r .* sin(theta + offset)];
                        Xk = Xk + randn(n_per, 2) * obj.Noise;
                        
                        X = [X; Xk]; 
                        Y = [Y; k * ones(n_per, 1)]; 
                    end
            end
            obj.MetaData.Shape = obj.Shape;
        end
    end
end
