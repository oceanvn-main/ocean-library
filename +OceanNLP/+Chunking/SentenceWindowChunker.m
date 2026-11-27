classdef SentenceWindowChunker < OceanNLP.Chunking.IChunker
    % SENTENCEWINDOWCHUNKER Sentence-level chunking strategy with context radius.
    %
    %   Detects sentence boundaries and builds a window centered on each
    %   target sentence, including WindowRadius sentences before and after it.

    properties (SetAccess=private)
        WindowRadius (1,1) double = 1 % Number of surrounding sentences to include on each side
    end

    methods
        function obj = SentenceWindowChunker(options)
            % SENTENCEWINDOWCHUNKER Construct a SentenceWindowChunker instance.
            %
            %   obj = SentenceWindowChunker('WindowRadius', 1)
            arguments
                options.WindowRadius (1,1) double {mustBeInteger, mustBeNonnegative} = 1
            end
            obj.WindowRadius = options.WindowRadius;
        end

        function chunks = chunk(obj, text)
            % CHUNK Segment text into sentence-centered context windows.
            arguments
                obj
                text string
            end

            % Split text into sentences using punctuation boundaries (. ! ?)
            sentences = split(regexprep(strtrim(text), '(?<=[.!?])\s+', '|'), '|');
            sentences = sentences(strlength(sentences) > 0);

            chunks = strings(1, numel(sentences));

            % For each sentence, build context window from (i - WindowRadius) to (i + WindowRadius)
            for i = 1:numel(sentences)
                idxStart = max(1, i - obj.WindowRadius);
                idxEnd   = min(numel(sentences), i + obj.WindowRadius);
                chunks(i) = join(sentences(idxStart:idxEnd), " ");
            end
        end
    end
end

