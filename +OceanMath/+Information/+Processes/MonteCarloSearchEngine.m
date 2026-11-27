classdef MonteCarloSearchEngine
    % MONTECARLOSEARCHENGINE Simulates Shannon's physical book-flip search.
    %
    % This class implements the Monte Carlo search algorithm that scans forward
    % in a text corpus starting from a random index to match a given context
    % and extract the following character.
    %
    % See also: OceanMath.Information.Processes.CorpusSanitizer,
    %           OceanMath.Information.Processes.ShannonTextGenerator

    properties
        Corpus char
        Alphabet (1, 27) char = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',' ']
    end

    methods
        function obj = MonteCarloSearchEngine(corpus)
            % MONTECARLOSEARCHENGINE Constructor.
            arguments
                corpus char
            end
            % Sanitize corpus using CorpusSanitizer
            obj.Corpus = OceanMath.Information.Processes.CorpusSanitizer.sanitize(corpus);
        end

        function [nextChar, searchLog] = searchNext(obj, context)
            % SEARCHNEXT Scans corpus forward from a random index for the context.
            %
            % context: The search string key (length 1 for bigrams, length 2 for trigrams).
            % Returns:
            %   nextChar  - The character immediately following the match.
            %   searchLog - Structural trace containing:
            %                 .startIndex
            %                 .matchIndex
            %                 .scannedIndices
            %                 .scannedText
            %                 .foundChar
            arguments
                obj (1,1) OceanMath.Information.Processes.MonteCarloSearchEngine
                context char
            end

            N = length(obj.Corpus);
            L = length(context);

            if N <= L
                error('OceanMath:Information:MonteCarloSearchEngine:CorpusTooShort', ...
                    'Corpus length (%d) must be greater than context length (%d).', N, L);
            end

            % Normalize context characters to uppercase
            context = upper(context);

            % 1. Generate a random uniform index in the corpus:
            % startIndex = randi(length(corpus) - length(context))
            startIndex = double(randi(N - L));

            % 2. Scan forward character by character from startIndex
            matchIndex = -1;
            scannedIndices = double.empty(1, 0);
            scannedText = char.empty(1, 0);

            for k = 0:(N-1)
                % Compute current start index with wrap-around
                currStart = double(mod(startIndex + k - 1, N) + 1);
                
                scannedIndices(end+1) = currStart; %#ok<AGROW>
                scannedText(end+1) = obj.Corpus(currStart); %#ok<AGROW>
                
                % Extract the substring of length L starting at currStart (handling wrap-around)
                idxRange = mod((currStart : currStart + L - 1) - 1, N) + 1;
                subStr = obj.Corpus(idxRange);
                
                % Check if matches context
                if strcmp(subStr, context)
                    matchIndex = currStart;
                    break;
                end
            end

            if matchIndex == -1
                error('OceanMath:Information:MonteCarloSearchEngine:ContextNotFound', ...
                    'Context "%s" was not found in the corpus.', context);
            end

            % 4. Extract the character immediately following the match
            nextCharIdx = mod(matchIndex + L - 1, N) + 1;
            nextChar = obj.Corpus(nextCharIdx);

            % Build search log struct
            searchLog = struct();
            searchLog.startIndex = startIndex;
            searchLog.matchIndex = matchIndex;
            searchLog.scannedIndices = scannedIndices;
            searchLog.scannedText = scannedText;
            searchLog.foundChar = nextChar;
        end
    end
end
