classdef ClassificationGenerator < OceanData.Generators.IDataGenerator
    % CLASSIFICATIONGENERATOR Creates Gaussian Mixture clusters.
    % Useful for testing LDA, QDA, and Linear Classifiers.
    
    properties
        Classes    (1,1) double {mustBePositive, mustBeInteger} = 2
        Dimensions (1,1) double {mustBePositive, mustBeInteger} = 2
        Separation (1,1) double = 2.0
    end
    
    methods
        function obj = ClassificationGenerator(opts)
            arguments
                opts.N          (1,1) double = 200
                opts.Classes    (1,1) double = 2
                opts.Dimensions (1,1) double = 2
                opts.Separation (1,1) double = 2.0
            end
            obj.N          = opts.N;
            obj.Classes    = opts.Classes;
            obj.Dimensions = opts.Dimensions;
            obj.Separation = opts.Separation;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            % Pre-allocate for O(1) memory efficiency
            X = zeros(obj.N, obj.Dimensions);
            Y = zeros(obj.N, 1);

            % Generate random Class Centroids
            centroids = randn(obj.Classes, obj.Dimensions) * obj.Separation;
            obj.MetaData.Centroids = centroids;

            % Vectorized Block Filling
            samplesPerClass = floor(obj.N / obj.Classes);
            startIdx = 1;
            
            for k = 1:obj.Classes
                % Handle remainder for the last class to ensure exact N
                if k == obj.Classes
                    count = obj.N - startIdx + 1; 
                else
                    count = samplesPerClass;
                end
                
                endIdx = startIdx + count - 1;
                
                % Shift standard normal noise to the centroid location
                X(startIdx:endIdx, :) = randn(count, obj.Dimensions) + centroids(k, :);
                Y(startIdx:endIdx) = k;
                
                startIdx = endIdx + 1;
            end
        end
    end
end
