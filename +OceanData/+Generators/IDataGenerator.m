classdef (Abstract) IDataGenerator < handle
    % IDATAGENERATOR Interface for robust data generation strategies.
    % Base class for the DataFactory Strategy Pattern.
    %
    % DEPARTS FROM OLD DESIGN:
    %   1. Uses the Template Method Pattern to guarantee [X, Y] signature.
    %   2. All generation settings must be public properties, not arguments
    %      to generate().
    %   3. Extra data (like true underlying functions, betas, centroids)
    %      are stored in the MetaData property after generation.
    
    properties
        N            (1,1) double {mustBePositive, mustBeInteger} = 100
        Seed         (1,1) double {mustBeNonnegative, mustBeInteger} = 0
        DefaultNoise (1,1) double {mustBeNonnegative} = 1.0
        MetaData     struct = struct()  % Populated after generate()
    end
    
    methods
        function [X, Y] = generate(obj, N)
            % GENERATE Creates [X, Y] data pairs.
            %   [X, Y] = obj.generate()
            %   [X, Y] = obj.generate(N)   % Overrides obj.N
            
            if nargin > 1
                obj.N = N;
            end
            
            if obj.Seed > 0
                rng(obj.Seed);
            end
            
            % Delegate to subclass implementation
            [X, Y] = obj.generateCore();
        end

        % ── Bayes Oracle (optional — synthetic generators should override) ──

        function tf = hasBayesOracle(~)
            % HASBAYESORACLE  Returns true if this generator can compute the
            %   Bayes optimal prediction. Override in synthetic generators.
            tf = false;
        end

        function [y_pred, err, densities] = bayesPredict(obj, X, y_true) %#ok<STOUT>
            % BAYESPREDICT  Bayes optimal classifier using true generative params.
            %   Override in synthetic generators that know the ground truth.
            %   File-based generators (e.g. ZipGenerator) cannot compute this.
            %
            %   [y_pred, err] = gen.bayesPredict(X_test, y_test)
            error('OceanData:IDataGenerator:NoBayesOracle', ...
                '%s is not a synthetic generator — no Bayes oracle available.', class(obj));
        end

        function varargout = bayesDensity(obj, X) %#ok<STOUT>
            % BAYESDENSITY  Class-conditional densities from the true model.
            %   Override in synthetic generators. See bayesPredict.
            error('OceanData:IDataGenerator:NoBayesOracle', ...
                '%s is not a synthetic generator — no Bayes oracle available.', class(obj));
        end
    end
    
    methods (Abstract, Access = protected)
        % Subclasses MUST implement this and return exactly [X, Y].
        % They should assign any additional details to obj.MetaData.
        [X, Y] = generateCore(obj)
    end
end
