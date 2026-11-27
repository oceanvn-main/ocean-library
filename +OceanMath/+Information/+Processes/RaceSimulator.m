classdef RaceSimulator < handle
    % RACESIMULATOR Simulates independent races of K horses using stochastic random draws.
    %
    % This class models a horse race where each horse has a true winning
    % probability. Winners are drawn according to these probabilities using
    % a discrete custom distribution and an inverse transform sampler.
    %
    % Properties:
    %   TrueProbabilities - (1 x K) double vector of positive winning probabilities summing to 1.
    %   NumHorses         - (1,1) double, number of horses K.
    %   Sampler           - OceanMath.Random.Samplers.ISampler strategy instance.
    %
    % Examples:
    %   sim = OceanMath.Information.Processes.RaceSimulator([0.5, 0.3, 0.2]);
    %   winner = sim.drawWinner();
    %   winners = sim.simulateRaces(100);
    %
    % See also: OceanMath.Random.Samplers.InverseTransformSampler,
    %           OceanMath.Information.Distributions.DiscreteDistribution

    properties
        TrueProbabilities (1,:) double {mustBePositive} = 1.0
        NumHorses (1,1) double {mustBeInteger, mustBePositive} = 1
        Sampler OceanMath.Random.Samplers.ISampler = OceanMath.Random.Samplers.InverseTransformSampler()
    end

    properties (Access = private)
        Distribution OceanMath.Information.Distributions.DiscreteDistribution
    end

    methods
        function obj = RaceSimulator(trueProbs)
            % RACESIMULATOR Construct a RaceSimulator.
            %
            % INPUTS:
            %   trueProbs - (1 x K) or (K x 1) vector of positive horse probabilities summing to 1.
            arguments
                trueProbs (1,:) double {mustBePositive, mustBeVector}
            end

            % Validate probabilities sum to 1
            if abs(sum(trueProbs) - 1.0) > 1e-9
                error('OceanMath:Information:RaceSimulator:InvalidProbabilities', ...
                    'True probabilities must sum to 1. Got sum = %.10f', sum(trueProbs));
            end

            obj.TrueProbabilities = trueProbs;
            obj.NumHorses = length(trueProbs);
            
            % Instantiate InverseTransformSampler and DiscreteDistribution
            obj.Sampler = OceanMath.Random.Samplers.InverseTransformSampler();
            obj.Distribution = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF( ...
                (1:obj.NumHorses)', trueProbs(:));
        end

        function winner = drawWinner(obj)
            % DRAWWINNER Draw a single winning horse index (1 to K).
            %
            % OUTPUTS:
            %   winner - scalar double, winning horse index (1-indexed).
            arguments
                obj (1,1) OceanMath.Information.Processes.RaceSimulator
            end

            samples = obj.Sampler.sample(obj.Distribution, 1);
            winner = samples(1);
        end

        function winners = simulateRaces(obj, numRaces)
            % SIMULATERACES Draw vectors of independent race winners.
            %
            % INPUTS:
            %   numRaces - (1,1) positive integer, number of races to simulate.
            %
            % OUTPUTS:
            %   winners - (numRaces x 1) double array of horse indices.
            arguments
                obj (1,1) OceanMath.Information.Processes.RaceSimulator
                numRaces (1,1) double {mustBePositive, mustBeInteger}
            end

            winners = obj.Sampler.sample(obj.Distribution, numRaces);
        end
    end
end
