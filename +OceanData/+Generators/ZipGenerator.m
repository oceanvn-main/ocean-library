classdef ZipGenerator < OceanData.Generators.IDataGenerator
    % ZIPGENERATOR Loads the ESL Digit Recognition Zip dataset.
    % 
    % Description:
    %   Loads the handwritten digit dataset. The data contains 256 pixel 
    %   values (-1 to 1) per image and their corresponding digit label.
    %
    % Usage:
    %   factory = OceanStats.Data.DataFactory('Seed', 123);
    %   factory.resolveStrategy('Zip').Type = 'test'; % optional config
    %   [X, Y, Meta] = factory.create('Zip');
    
    properties
        Type (1,1) string {mustBeMember(Type, ["train", "test", "all"])} = "train"
    end
    
    methods
        function obj = ZipGenerator(options)
            arguments
                options.Type (1,1) string = "train"
                options.Seed (1,1) double = 0
                options.DefaultNoise (1,1) double = 1.0
            end
            obj.Type = options.Type;
            obj.Seed = options.Seed;
            obj.DefaultNoise = options.DefaultNoise;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            % Resolve data root
            root = fullfile('c:\Ocean Library\Matlab code practice\Data\zip');
            
            % Load logical split
            if obj.Type == "train"
                data = loadData(fullfile(root, 'zip.train', 'zip.train'));
            elseif obj.Type == "test"
                data = loadData(fullfile(root, 'zip.test', 'zip.test'));
            elseif obj.Type == "all"
                trainData = loadData(fullfile(root, 'zip.train', 'zip.train'));
                testData = loadData(fullfile(root, 'zip.test', 'zip.test'));
                data = [trainData; testData];
            end
            
            % Column 1 is digit class (0-9)
            % Columns 2:257 are features (16x16 flattened images)
            Y = data(:, 1);
            X = data(:, 2:end);
            
            % MetaData population
            obj.MetaData.Name = 'Zip Digits';
            obj.MetaData.Categories = 0:9;
            obj.MetaData.NumFeatures = 256;
            obj.MetaData.Split = obj.Type;
        end
    end
end

function data = loadData(filepath)
    % Simple text loader for whitespace delimited file. 
    data = readmatrix(filepath, 'FileType', 'text');
end
