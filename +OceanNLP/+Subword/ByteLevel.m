classdef ByteLevel
    % BYTELEVEL  GPT-2's reversible byte <-> printable-character alphabet mapping.
    %
    % Every UTF-8 BYTE (0..255) is mapped to a distinct printable Unicode character so that a
    % byte-level BPE vocabulary needs only 256 base symbols and can encode ANY
    % text in any language without producing unknown (`[UNK]`) tokens.
    %
    % Mapping Rules:
    %   1. Bytes that are already printable Latin-1 (33-126, 161-172, 174-255) map to themselves.
    %   2. The remaining 68 unprintable/whitespace bytes map to U+0100 (256) upward in byte order:
    %      - Space 0x20 (32)   -> U+0120 ('Ġ' - Latin Capital Letter G with Dot Above)
    %      - Newline 0x0A (10) -> U+010A ('Ċ' - Latin Capital Letter C with Dot Above)
    %      - Tab 0x09 (9)      -> U+0109 ('Ĉ' - Latin Capital Letter C with Circumflex)
    %
    % Examples:
    %   encoded = ByteLevel.encode(" Hello World!");  % -> "ĠHelloĠWorld!"
    %   decoded = ByteLevel.decode("ĠHelloĠWorld!");  % -> " Hello World!"
    %
    % Reference: GPT-2 encoder.py bytes_to_unicode().

    methods (Static)
        function t = table()
            % TABLE Generate or return persistent 1x256 code point lookup table.
            % table(b + 1) returns the 32-bit Unicode code point mapped from UTF-8 byte b (0..255).
            persistent T
            if isempty(T)
                % Printable Latin-1 byte ranges:
                %   33:126  (ASCII printable characters '!' through '~')
                %   161:172 (Latin-1 Supplement printable range 1)
                %   174:255 (Latin-1 Supplement printable range 2)
                bs = [33:126, 161:172, 174:255];
                cs = bs; % Printable bytes map 1:1 to themselves
                n = 0;
                % Map remaining 68 unprintable/whitespace bytes sequentially to U+0100 (256) upwards
                for b = 0:255
                    if ~any(bs == b)
                        bs(end+1) = b; %#ok<AGROW>
                        cs(end+1) = 256 + n; %#ok<AGROW> % Map to U+0100 + n
                        n = n + 1;
                    end
                end
                T = zeros(1, 256);
                T(bs + 1) = cs; % Fill 1x256 lookup table indexed by (byte_value + 1)
            end
            t = T;
        end

        function a = alphabet()
            % ALPHABET Return the 1x256 string array of printable base symbols in byte order (0..255).
            % Used as the initial seed vocabulary for training byte-level BPE algorithms.
            t = OceanNLP.Subword.ByteLevel.table();
            a = string(num2cell(char(t)));
            a = reshape(a, 1, []);
        end

        function mapped = encode(text)
            % ENCODE Convert a raw text string into a printable ByteLevel string.
            %
            % Workflow:
            %   1. Convert text to UTF-8 raw byte values (0..255).
            %   2. Map each byte through table() into a visible character.
            %
            % Example:
            %   ByteLevel.encode(" Hello") -> "ĠHello"
            arguments
                text (1,1) string
            end
            if strlength(text) == 0, mapped = ""; return; end
            bytes = double(unicode2native(char(text), 'UTF-8')); % Extract UTF-8 bytes (0..255)
            t = OceanNLP.Subword.ByteLevel.table();
            mapped = string(char(t(bytes + 1))); % Map bytes to 100% printable characters
        end

        function text = decode(mapped)
            % DECODE Convert a printable ByteLevel string back to exact original text.
            %
            % Workflow:
            %   1. Reverse-lookup printable characters back to raw UTF-8 uint8 bytes.
            %   2. Decode UTF-8 byte array back into original Unicode text.
            %
            % Example:
            %   ByteLevel.decode("ĠHello") -> " Hello"
            arguments
                mapped (1,1) string
            end
            if strlength(mapped) == 0, text = ""; return; end
            t = OceanNLP.Subword.ByteLevel.table();
            [known, loc] = ismember(double(char(mapped)), t);
            if ~all(known)
                error('OceanNLP:Subword:ByteLevel:NotMapped', ...
                    'decode() received characters outside the byte-level alphabet.');
            end
            % Reconstruct original text from raw UTF-8 bytes
            text = string(native2unicode(uint8(loc - 1), 'UTF-8')); 
        end
    end
end
