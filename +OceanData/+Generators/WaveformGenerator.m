classdef WaveformGenerator < OceanData.Generators.IDataGenerator
    % WAVEFORMGENERATOR Generates the 3-class waveform dataset from ESL 12.7.1
    % 
    % The predictors are 21-dimensional (j = 1...21):
    % X_j = U h_1(j) + (1-U) h_2(j) + e_j  (Class 1)
    % X_j = U h_1(j) + (1-U) h_3(j) + e_j  (Class 2)
    % X_j = U h_2(j) + (1-U) h_3(j) + e_j  (Class 3)
    %
    % where U is uniform(0,1), e_j is standard normal N(0,1), and
    % h_1(j) = max(6 - |j - 11|, 0)
    % h_2(j) = h_1(j - 4)
    % h_3(j) = h_1(j + 4)
    %
    % Ref: Hastie, Tibshirani, Friedman (2009). The Elements of Statistical Learning.
    
    properties
        IncludeNoise (1,1) logical = true % Set to false to get pure waveforms (e.g. for Fig 12.14)
    end
    
    methods
        function obj = WaveformGenerator(opts)
            arguments
                opts.N (1,1) double = 300 % Default from ESL: 300 training observations
                opts.IncludeNoise (1,1) logical = true
            end
            obj.N = opts.N;
            obj.IncludeNoise = opts.IncludeNoise;
            
            % Save the base waveforms for easy access (e.g., plotting)
            [h1, h2, h3] = obj.computeBaseWaveforms();
            obj.MetaData.BaseWaveforms = [h1; h2; h3];
            obj.MetaData.Dimensions = 21;
            obj.MetaData.Classes = 3;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            N_total = obj.N;
            p = 21;
            
            % Generate equal priors roughly
            % We will just use randi to assign classes uniformly
            Y = randi([1, 3], N_total, 1);
            
            % Preallocate X
            X = zeros(N_total, p);
            
            % Get base waveforms
            [h1, h2, h3] = obj.computeBaseWaveforms();
            
            % Generate U ~ Uniform(0,1) for each sample
            U = rand(N_total, 1);
            
            % Generate epsilon ~ N(0,1) for all samples
            if obj.IncludeNoise
                epsilon = randn(N_total, p);
            else
                epsilon = zeros(N_total, p);
            end
            
            % Generate class by class
            idx1 = (Y == 1);
            idx2 = (Y == 2);
            idx3 = (Y == 3);
            
            % Class 1: U*h1 + (1-U)*h2 + e
            n1 = sum(idx1);
            X(idx1, :) = bsxfun(@times, U(idx1), h1) + bsxfun(@times, (1 - U(idx1)), h2) + epsilon(idx1, :);
            
            % Class 2: U*h1 + (1-U)*h3 + e
            n2 = sum(idx2);
            X(idx2, :) = bsxfun(@times, U(idx2), h1) + bsxfun(@times, (1 - U(idx2)), h3) + epsilon(idx2, :);
            
            % Class 3: U*h2 + (1-U)*h3 + e
            n3 = sum(idx3);
            X(idx3, :) = bsxfun(@times, U(idx3), h2) + bsxfun(@times, (1 - U(idx3)), h3) + epsilon(idx3, :);
        end
        
        function [h1, h2, h3] = computeBaseWaveforms(~)
            j = 1:21;
            
            % h1(j) = max(6 - |j - 11|, 0)
            h1 = max(6 - abs(j - 11), 0);
            
            % h2(j) = h1(j - 4) = max(6 - |(j - 4) - 11|, 0) = max(6 - |j - 15|, 0)
            h2 = max(6 - abs(j - 15), 0);
            
            % h3(j) = h1(j + 4) = max(6 - |(j + 4) - 11|, 0) = max(6 - |j - 7|, 0)
            h3 = max(6 - abs(j - 7), 0);
        end
    end
end
