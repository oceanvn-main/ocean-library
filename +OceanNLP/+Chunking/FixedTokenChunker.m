classdef FixedTokenChunker < OceanNLP.Chunking.IChunker
    % FIXEDTOKENCHUNKER Fixed word-count text chunking strategy.
    %
    %   Splits input text into non-overlapping chunks containing up to
    %   ChunkSize words each.

    properties (SetAccess=private)
        ChunkSize (1,1) double = 128 % Maximum number of words per chunk
    end

    methods
        function obj = FixedTokenChunker(options)
            % FIXEDTOKENCHUNKER Construct a FixedTokenChunker instance.
            %
            %   obj = FixedTokenChunker('ChunkSize', 128)
            arguments
                options.ChunkSize (1,1) double {mustBeInteger, mustBePositive} = 128
            end
            obj.ChunkSize = options.ChunkSize;
        end

        function chunks = chunk(obj, text)
            % CHUNK Segment text into fixed-size word chunks.
            %
            %   chunks = chunk(obj, text) splits text by whitespace and groups
            %   words into contiguous, non-overlapping chunks.
            arguments
                obj
                text string
            end

            % Clean leading/trailing whitespace and split into individual word tokens
            words = split(strtrim(text));
            words = words(strlength(words) > 0);

            % Calculate total number of chunks required
            n = ceil(numel(words) / obj.ChunkSize);
            chunks = strings(1, n);

            % Slice words for each non-overlapping chunk
            for i = 1:n
                idxStart = (i - 1) * obj.ChunkSize + 1;
                idxEnd = min(i * obj.ChunkSize, numel(words));
                chunks(i) = join(words(idxStart:idxEnd), " ");
            end
        end
    end
end

