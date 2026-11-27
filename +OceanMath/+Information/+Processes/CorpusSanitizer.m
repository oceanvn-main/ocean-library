classdef CorpusSanitizer
    % CORPUSSANITIZER Normalizes raw text to uppercase letters A-Z and spaces.
    %
    % This class provides static methods to sanitize text from character arrays
    % or directly from text files to a strict 27-state alphabet (A-Z and space).
    %
    % See also: OceanMath.Information.Processes.ShannonTextGenerator

    methods (Static)
        function cleanText = sanitize(rawText)
            % SANITIZE Normalizes a character array to 27-state uppercase space.
            %
            % Accepts a character array, converts it to uppercase, replaces 
            % carriage returns, line feeds, tabs, numbers, punctuation, and symbols
            % with space characters, strips any characters outside A-Z and space,
            % and compresses multiple consecutive spaces into a single space.
            arguments
                rawText char
            end

            if isempty(rawText)
                cleanText = char.empty(1, 0);
                return;
            end

            % Ensure it is a 1xL character row vector
            cleanText = rawText(:).';

            % 1. Convert to uppercase
            cleanText = upper(cleanText);

            % 2. Replace carriage returns, line feeds, tabs, numbers, punctuation, 
            %    and symbols with space characters.
            cleanText = regexprep(cleanText, '[\r\n\t0-9`~!@#\$%\^&\*\(\)\-_\+=\[\]\{\}\\\|;:''",<\.>\/\?]', ' ');

            % 3. Strip any characters outside A-Z and space
            % Keep only A-Z and space (' ')
            keepIdx = (cleanText >= 'A' & cleanText <= 'Z') | (cleanText == ' ');
            cleanText = cleanText(keepIdx);

            % 4. Compress multiple consecutive spaces into a single space character
            cleanText = regexprep(cleanText, ' +', ' ');
        end

        function cleanText = sanitizeFile(filePath)
            % SANITIZEFILE Reads a text file and returns sanitized contents.
            arguments
                filePath (1, :) char
            end

            fid = fopen(filePath, 'r', 'n', 'UTF-8');
            if fid == -1
                error('OceanMath:Information:CorpusSanitizer:FileNotFound', ...
                    'Could not open file: %s', filePath);
            end
            
            cleanUp = onCleanup(@() fclose(fid));
            rawText = fread(fid, '*char').';
            
            cleanText = OceanMath.Information.Processes.CorpusSanitizer.sanitize(rawText);
        end
    end
end
