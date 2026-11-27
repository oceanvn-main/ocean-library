classdef RecursiveCharChunker < OceanNLP.Chunking.IChunker
    % RECURSIVECHARCHUNKER Structural line-by-line character length chunker.
    %
    %   Aggregates newline-separated paragraphs/lines into chunks without
    %   exceeding MaxCharacters per chunk.

    properties (SetAccess=private)
        MaxCharacters (1,1) double = 1000 % Maximum character count per chunk
    end

    methods
        function obj = RecursiveCharChunker(options)
            % RECURSIVECHARCHUNKER Construct a RecursiveCharChunker instance.
            %
            %   obj = RecursiveCharChunker('MaxCharacters', 1000)
            arguments
                options.MaxCharacters (1,1) double {mustBeInteger, mustBePositive} = 1000
            end
            obj.MaxCharacters = options.MaxCharacters;
        end

        function chunks = chunk(obj, text)
            % CHUNK Segment text by lines without exceeding character threshold.
            arguments
                obj
                text string
            end

            % Split input by newline characters
            parts = split(text, newline);
            chunks = strings(1, 0);
            current = "";

            % Accumulate lines into current chunk until MaxCharacters is reached
            for i = 1:numel(parts)
                candidate = strtrim(join([current, parts(i)], " "));
                if strlength(candidate) <= obj.MaxCharacters
                    current = candidate;
                else
                    if strlength(current) > 0
                        chunks(end+1) = current; %#ok<AGROW>
                    end
                    current = parts(i);
                end
            end

            % Append any remaining buffered text
            if strlength(strtrim(current)) > 0
                chunks(end+1) = strtrim(current); %#ok<AGROW>
            end
        end
    end
end

