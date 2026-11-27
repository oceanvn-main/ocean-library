classdef ChunkerFactory
    % CHUNKERFACTORY Static factory for instantiating text chunkers.
    %
    %   Provides a centralized factory method to create concrete chunker
    %   objects based on strategy names.

    methods (Static)
        function c = create(name, options)
            % CREATE Instantiate a chunker by strategy name.
            %
            %   c = ChunkerFactory.create("sliding", WindowSize=128, Overlap=32)
            %   c = ChunkerFactory.create("sentence-window", WindowRadius=1)
            %   c = ChunkerFactory.create("recursive", MaxCharacters=1000)
            arguments
                name (1,1) string
                options.MaxCharacters (1,1) double {mustBeInteger, mustBePositive} = 1000
                options.WindowSize    (1,1) double {mustBeInteger, mustBePositive} = 128
                options.Overlap       (1,1) double {mustBeInteger, mustBeNonnegative} = 32
                options.WindowRadius  (1,1) double {mustBeInteger, mustBeNonnegative} = 1
            end

            switch lower(name)
                case "recursive"
                    c = OceanNLP.Chunking.RecursiveCharChunker('MaxCharacters', options.MaxCharacters);
                case "sliding"
                    c = OceanNLP.Chunking.SlidingWindowChunker('WindowSize', options.WindowSize, 'Overlap', options.Overlap);
                case "sentence-window"
                    c = OceanNLP.Chunking.SentenceWindowChunker('WindowRadius', options.WindowRadius);
                otherwise
                    error('OceanNLP:Chunker:UnknownStrategy', 'Unknown chunker strategy "%s".', name);
            end
        end
    end
end

