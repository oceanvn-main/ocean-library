classdef BertBasicPreTokenizer < OceanNLP.Text.IPreTokenizer
    % BERTBASICPRETOKENIZER  BERT's BasicTokenizer (Devlin et al. 2018; HF
    % transformers BasicTokenizer), Unicode-aware via OceanNLP.Text.Unicode:
    %   1. clean: drop NUL, U+FFFD and control characters; map whitespace to space
    %   2. put spaces around CJK ideographs
    %   3. whitespace split
    %   4. per word (unless in NeverSplit): if Lowercase -> lowercase and, if
    %      StripAccents, remove accents (NFD, drop combining marks);
    %      then split OFF every punctuation character as its own word
    %
    % This closes the disclosed BasicTokenize limits of the older
    % WordPieceTokenizer.BasicTokenize (ASCII punctuation only).
    %
    %   pre = BertBasicPreTokenizer()                              % uncased BERT
    %   pre = BertBasicPreTokenizer('Lowercase', false)            % cased BERT
    %
    % StripAccents defaults to Lowercase (HF behaviour when strip_accents=None).
    properties (SetAccess = private)
        Lowercase (1,1) logical = true
        StripAccents (1,1) logical = true
        NeverSplit (1,:) string = ["[CLS]" "[SEP]" "[MASK]" "[PAD]" "[UNK]"]
    end
    properties (Constant)
        Reference = "BERT BasicTokenizer (Devlin et al. 2018; HF transformers BasicTokenizer)"
        InDomain = "raw-text"
        OutDomain = "token-sequence"
    end
    methods (Access = public)
        function obj = BertBasicPreTokenizer(options)
            % BERTBASICPRETOKENIZER Construct an instance of BertBasicPreTokenizer.
            %
            % Parameters:
            %   Lowercase (logical)    - Perform lowercasing on all tokens (default: true).
            %   StripAccents (logical) - Decompose and strip diacritics/accents (default: matches Lowercase).
            %   NeverSplit (string)    - Array of special tokens that should never be split (e.g. "[CLS]").
            arguments
                options.Lowercase (1,1) logical = true
                options.StripAccents (1,1) logical
                options.NeverSplit (1,:) string = ["[CLS]" "[SEP]" "[MASK]" "[PAD]" "[UNK]"]
            end
            obj.Lowercase = options.Lowercase;
            if isfield(options, 'StripAccents')
                obj.StripAccents = options.StripAccents;
            else
                % HF Transformers default: strip_accents defaults to matches uncased setting
                obj.StripAccents = options.Lowercase;
            end
            obj.NeverSplit = options.NeverSplit;
        end

        function words = forward(obj, text)
            % FORWARD Process an input text string into BERT pre-tokenized words.
            %
            % Pipeline Execution Steps:
            %   Step 1: Clean text (drop NUL, replacement char U+FFFD, and unprintable controls; normalize whitespace).
            %   Step 2: Add spaces around CJK (Chinese, Japanese, Korean) ideographs to isolate them.
            %   Step 3: Split into raw word units using whitespace.
            %   Step 4: Per word, apply lowercasing & accent stripping, then separate punctuation as standalone tokens.
            arguments
                obj
                text (1,1) string
            end
            U = OceanNLP.Text.Unicode();
            cps = U.codepoints(text); % Convert string into 1xN array of numeric Unicode Code Point IDs

            % --- STEP 1: CLEAN TEXT ---
            % BERT/HF clean_text rule: drop NUL (0), replacement char U+FFFD (65533), and control characters.
            % Unicode Category Code IDs:
            %   15 = Cc (Control)
            %   16 = Cf (Format)
            %   18 = Co (Private Use)
            %   19 = Cs (Surrogate)
            % Exception: preserve standard whitespace controls \t (9), \n (10), \r (13).
            cat = U.category(cps);
            control = cat == 15 | cat == 16 | cat == 18 | cat == 19; % Logical mask for unprintable control chars
            drop = cps == 0 | cps == 65533 | (control & cps ~= 9 & cps ~= 10 & cps ~= 13);
            cps = cps(~drop); % Remove all marked control/invalid characters
            cps(U.isSpace(cps)) = 32; % Standardize all Unicode whitespace characters to ASCII space (32)

            % --- STEP 2: SPACE PADDING AROUND CJK IDEOGRAPHS ---
            % Insert spaces (code point 32) before and after Chinese/Japanese/Korean characters
            % so that CJK characters are treated as independent tokens during whitespace splitting.
            cjk = obj.isCjk_(cps);
            if any(cjk)
                out = zeros(1, 0);
                for k = 1:numel(cps)
                    if cjk(k)
                        out = [out 32 cps(k) 32]; % Pad with spaces before & after CJK char
                    else
                        out = [out cps(k)];
                    end %#ok<AGROW>
                end
                cps = out;
            end

            % --- STEP 3: WHITESPACE SPLIT ---
            % Split the cleaned & padded code point array into raw word chunks on whitespace runs
            raw = OceanNLP.Text.PreTokenize.WhitespacePreTokenizer().forward(U.fromCodepoints(cps));

            % --- STEP 4: PER-WORD PROCESSING & PUNCTUATION ISOLATION ---
            words = strings(1, 0);
            for w = raw
                % Protect special Transformer control tokens (e.g. "[CLS]", "[SEP]") from being split or modified
                if any(w == obj.NeverSplit)
                    words(end+1) = w; %#ok<AGROW>
                    continue
                end

                % Optional lowercasing (uncased model setting)
                if obj.Lowercase, w = U.lowerText(w); end %#ok<FXSET>

                % Optional accent stripping: NFD decompose and drop Category 6 (Mn - Non-spacing mark)
                % Example: "e" + "´" (U+0301) -> drops U+0301, keeping base letter "e"
                if obj.StripAccents
                    d = U.codepoints(U.normalize(w, "NFD"));
                    w = U.fromCodepoints(d(U.category(d) ~= 6)); % Strip accents (Mn), keep base letters
                end

                % Isolate punctuation: split off every punctuation character into a standalone token
                words = [words, obj.splitPunctuation_(w)]; %#ok<AGROW> "cafe!" → ["cafe", "!"]
            end
        end

        function s = name(obj)
            % NAME Return unique identifier for this pre-tokenizer instance.
            s = "bertbasic";
            if ~obj.Lowercase, s = "bertbasic-cased"; end
        end
    end

    methods (Access = private)
        function parts = splitPunctuation_(~, word)
            % SPLITPUNCTUATION_ Splits a word into tokens by isolating punctuation.
            % Any character identified as Unicode Punctuation becomes its own 1-character token.
            % Non-punctuation sequences are grouped into single word tokens.
            %
            % Example: "hello,world!" -> ["hello", ",", "world", "!"]
            cps = OceanNLP.Text.Unicode.codepoints(word);
            if isempty(cps), parts = strings(1, 0); return; end
            isP = OceanNLP.Text.Unicode.isPunctuation(cps);
            parts = strings(1, 0);
            i = 1;  n = numel(cps);
            while i <= n
                if isP(i)
                    % Punctuation character: emit as individual 1-character token
                    parts(end+1) = OceanNLP.Text.Unicode.fromCodepoints(cps(i)); %#ok<AGROW>
                    i = i + 1;
                else
                    % Non-punctuation character sequence: scan contiguous non-punctuation span
                    j = i;
                    while j < n && ~isP(j + 1), j = j + 1; end
                    parts(end+1) = OceanNLP.Text.Unicode.fromCodepoints(cps(i:j)); %#ok<AGROW>
                    i = j + 1;
                end
            end
        end

        function tf = isCjk_(~, cps)
            % ISCJK_ Determine whether code points belong to CJK Unicode ranges.
            % Includes main CJK Unified Ideographs block, Extensions A-E, Compatibility Ideographs,
            % matching HuggingFace Transformers / BERT reference implementation ranges:
            %   U+4E00..U+9FFF   (CJK Unified Ideographs)
            %   U+3400..U+4DBF   (CJK Extension A)
            %   U+20000..U+2A6DF (CJK Extension B)
            %   U+2A700..U+2B73F (CJK Extension C)
            %   U+2B740..U+2B81F (CJK Extension D)
            %   U+2B820..U+2CEAF (CJK Extension E)
            %   U+F900..U+FAFF   (CJK Compatibility Ideographs)
            %   U+2F800..U+2FA1F (CJK Compatibility Supplement)
            tf = (cps >= hex2dec('4E00') & cps <= hex2dec('9FFF')) | (cps >= hex2dec('3400') & cps <= hex2dec('4DBF')) | ...
                 (cps >= hex2dec('20000') & cps <= hex2dec('2A6DF')) | (cps >= hex2dec('2A700') & cps <= hex2dec('2B73F')) | ...
                 (cps >= hex2dec('2B740') & cps <= hex2dec('2B81F')) | (cps >= hex2dec('2B820') & cps <= hex2dec('2CEAF')) | ...
                 (cps >= hex2dec('F900') & cps <= hex2dec('FAFF')) | (cps >= hex2dec('2F800') & cps <= hex2dec('2FA1F'));
        end
    end
end
