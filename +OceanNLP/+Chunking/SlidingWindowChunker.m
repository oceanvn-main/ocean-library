classdef SlidingWindowChunker < OceanNLP.Chunking.IChunker
    % SLIDINGWINDOWCHUNKER Overlapping sliding-window word chunking strategy.
    %
    %   Splits text into fixed-size word windows with configurable overlap
    %   to preserve context across chunk boundaries in NLP/RAG tasks.

    properties (SetAccess=private)
        WindowSize (1,1) double % Total words per chunk window
        Overlap    (1,1) double % Number of overlapping words between consecutive windows
    end

    methods
        function obj = SlidingWindowChunker(options)
            % SLIDINGWINDOWCHUNKER Construct a SlidingWindowChunker instance.
            %
            %   obj = SlidingWindowChunker('WindowSize', 128, 'Overlap', 32)
            arguments
                options.WindowSize (1,1) double {mustBeInteger, mustBePositive} = 128
                options.Overlap    (1,1) double {mustBeInteger, mustBeNonnegative} = 32
            end

            if options.Overlap >= options.WindowSize
                error('OceanNLP:Chunker:BadOverlap', 'Overlap must be strictly smaller than WindowSize.');
            end

            obj.WindowSize = options.WindowSize;
            obj.Overlap    = options.Overlap;
        end

        function chunks = chunk(obj, text)
            % CHUNK Segment text into sliding word windows with overlap.
            arguments
                obj
                text string
            end

            % Split text into words and handle empty input edge cases
            words = split(strtrim(text));
            if isempty(words) || any(words == "")
                chunks = strings(1, 0);
                return;
            end

            % Compute step size between window starting points (WindowSize - Overlap)
            step = obj.WindowSize - obj.Overlap;
            starts = 1:step:numel(words);
            chunks = strings(1, numel(starts));

            % Construct each sliding window chunk
            for i = 1:numel(starts)
                idxStart = starts(i);
                idxEnd = min(starts(i) + obj.WindowSize - 1, numel(words));
                chunks(i) = join(words(idxStart:idxEnd), " ");
            end
        end
    end
end

