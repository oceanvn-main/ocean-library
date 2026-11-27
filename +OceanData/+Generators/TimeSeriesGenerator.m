classdef TimeSeriesGenerator < OceanData.Generators.IDataGenerator
    % TIMESERIESGENERATOR Creates stochastic temporal processes.
    % Models: AR1 processes and Random Walks with Trends.
    % 
    % Note: Returns [X, Y] where X is Time steps and Y is the Sequence.
    
    properties
        Model      (1,1) string {mustBeMember(Model, ["AR1", "RandomWalk"])} = "AR1"
        Coeffs     (1,1) double = 0.8
        TrendSlope (1,1) double = 0.05
        Sigma      (1,1) double = -1 % -1 means use DefaultNoise
    end
    
    methods
        function obj = TimeSeriesGenerator(opts)
            arguments
                opts.N          (1,1) double = 200
                opts.Model      (1,1) string = "AR1"
                opts.Coeffs     (1,1) double = 0.8
                opts.TrendSlope (1,1) double = 0.05
                opts.Sigma      (1,1) double = -1
            end
            obj.N          = opts.N;
            obj.Model      = opts.Model;
            obj.Coeffs     = opts.Coeffs;
            obj.TrendSlope = opts.TrendSlope;
            obj.Sigma      = opts.Sigma;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            sig = obj.Sigma;
            if sig == -1, sig = obj.DefaultNoise; end

            X = (1:obj.N)'; % Time steps
            Noise = sig * randn(obj.N, 1);
            Y = zeros(obj.N, 1);
            
            % Generate Deterministic Trend
            Trend = X * obj.TrendSlope;

            switch obj.Model
                case "AR1"
                    % Use MATLAB's built-in filter for optimized recursion
                    % H(z) = 1 / (1 - phi*z^-1)
                    Y = filter(1, [1, -obj.Coeffs], Noise);
                    
                case "RandomWalk"
                    % Optimized Cumulative Sum
                    Y = cumsum(Noise);
            end
            
            Y = Y + Trend;
            
            % Save components
            obj.MetaData.Trend = Trend;
            obj.MetaData.Signal = Y - Trend; % Pure stochastic part
        end
    end
end
