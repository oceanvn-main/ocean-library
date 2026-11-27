classdef Material
    % MATERIAL  Value class representing optical surface properties.
    
    properties
        Ambient (3,1) double = [0.1; 0.1; 0.1]     % Ka: ambient coefficient [R; G; B]
        Diffuse (3,1) double = [0.8; 0.8; 0.8]     % Kd: diffuse coefficient [R; G; B]
        Specular (3,1) double = [0.5; 0.5; 0.5]    % Ks: specular coefficient [R; G; B]
        Shininess (1,1) double = 32.0              % alpha: specular shininess exponent
        Reflectivity (3,1) double = [0.0; 0.0; 0.0] % Kr: mirror reflection coefficient [R; G; B]
    end
    
    methods
        function obj = Material(options)
            arguments
                options.Ambient (3,1) double = [0.1; 0.1; 0.1]
                options.Diffuse (3,1) double = [0.8; 0.8; 0.8]
                options.Specular (3,1) double = [0.5; 0.5; 0.5]
                options.Shininess (1,1) double = 32.0
                options.Reflectivity (3,1) double = [0.0; 0.0; 0.0]
            end
            obj.Ambient = options.Ambient;
            obj.Diffuse = options.Diffuse;
            obj.Specular = options.Specular;
            obj.Shininess = options.Shininess;
            obj.Reflectivity = options.Reflectivity;
        end
    end
end
