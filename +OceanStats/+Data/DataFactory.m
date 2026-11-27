classdef DataFactory < handle
    % DATAFACTORY Context/Runner for the Data Generator Strategy Pattern.
    %
    % The DataFactory delegates data generation to dynamically loaded
    % strategy objects from the `+OceanStats/+Data/+Generators` package.
    % This ensures zero breaking changes when adding new data types.
    %
    % == SUPPORTED DATA TYPES ==
    %   1. "Regression":        y = f(x) + epsilon
    %   2. "Classification":    Gaussian Mixture Models (GMM)
    %   3. "HighDim":           Sparse Regression (p >> N)
    %   4. "Shapes":            Non-linear manifolds (Circles, XOR, Moons, Spirals)
    %   5. "TimeSeries":        AR processes and Random Walks
    %   6. "ComplexFunctions":  MARS Benchmarks (Tensor, Friedman)
    %
    % == USAGE EXAMPLES ==
    %   % 1. Legacy API identical signature
    %   factory = OceanStats.Data.DataFactory('Seed', 2026);
    %   [X, Y, Truth] = factory.create('HighDim', 'N', 100, 'Features', 500);
    %
    %   % 2. Complex MARS Benchmarks
    %   [X, Y, mu] = factory.create('ComplexFunctions', 'Function', 'Friedman1');
    %
    % Author: Ocean Library System
    % Date: 2026-02-22
    
    properties
        Seed (1,1) double {mustBeNonnegative, mustBeInteger} = 0
        DefaultNoise (1,1) double {mustBeNonnegative} = 1.0
    end
    
    methods
        function obj = DataFactory(options)
            arguments
                options.Seed (1,1) double = 0
                options.DefaultNoise (1,1) double = 1.0
            end
            obj.Seed = options.Seed;
            obj.DefaultNoise = options.DefaultNoise;
            
            % Initialize the Mersenne Twister generator
            rng(obj.Seed, 'twister');
        end
        
        function [X, Y, Meta] = create(obj, type, varargin)
            % CREATE Dynamically injects the strategy and generates data.
            
            arguments
                obj
                type (1,1) string {mustBeMember(type, ["Regression", "Classification", "HighDim", "Shapes", "TimeSeries", "ComplexFunctions"])}
            end
            arguments (Repeating)
                varargin
            end
            
            generator = obj.resolveStrategy(type);
            
            % Push factory-level configuration down to the strategy
            generator.Seed = obj.Seed;
            generator.DefaultNoise = obj.DefaultNoise;
            
            [X, Y, Meta] = generator.generate(varargin{:});
        end
        
        function setSeed(obj, newSeed)
            obj.Seed = newSeed;
            rng(obj.Seed, 'twister');
        end
    end
    
    methods (Access = private)
        function strategy = resolveStrategy(~, type)
            % Maps the generic string to the specific Object Instantiation
            switch type
                case "Regression"
                    strategy = OceanStats.Data.Generators.RegressionGenerator();
                case "Classification"
                    strategy = OceanStats.Data.Generators.ClassificationGenerator();
                case "HighDim"
                    strategy = OceanStats.Data.Generators.HighDimGenerator();
                case "Shapes"
                    strategy = OceanStats.Data.Generators.ShapesGenerator();
                case "TimeSeries"
                    strategy = OceanStats.Data.Generators.TimeSeriesGenerator();
                case "ComplexFunctions"
                    strategy = OceanStats.Data.Generators.ComplexFunctionsGenerator();
            end
        end
    end
end