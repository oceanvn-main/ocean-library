classdef CodeTable
    % CODETABLE  Immutable result of a source-coding algorithm.
    %
    % Stores the complete mapping from symbols to binary codewords, along
    % with pre-computed or lazily derived information-theoretic metrics.
    % Created by any ISourceCoder (e.g., HuffmanCoder, ShannonFanoCoder).
    %
    % Construction — use the static factory:
    %   table = OceanMath.Information.Coding.CodeTable.build(symbols, probs, codewords)
    %
    % Metric methods:
    %   L   = table.expectedLength()   % L_avg = sum(p_k * len_k)
    %   R   = table.redundancy()       % R = L_avg - H(X) / log2(Base)
    %   eta = table.efficiency()       % eta = H(X) / L_avg  in [0, 1]
    %   K   = table.kraftSum()         % K = sum(Base^(-len_k)), should be <= 1
    %   tf  = table.isComplete()       % K == 1 (prefix-free complete code)
    %   gap = table.codingGap()        % L_avg - H(X) in same base units
    %
    % Display:
    %   table.printSummary()           % formatted table to command window
    %   s = table.toStruct()           % serializable plain struct
    %
    % Usage:
    %   coder = OceanMath.Information.Coding.HuffmanCoder();
    %   table = coder.encode({'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
    %   table.printSummary()
    %   L = table.expectedLength()     % ~1.9 bits
    %
    % See also: OceanMath.Information.Coding.HuffmanCoder
    %           OceanMath.Information.Coding.ISourceCoder

    % ------------------------------------------------------------------ %
    properties (SetAccess = immutable)
        % Symbols_     : (K x 1) cell of string symbol labels
        % Probs_        : (1 x K) double, PMF values, sum = 1
        % Codewords_    : (K x 1) cell of char vectors (e.g. '101')
        % Lengths_      : (1 x K) double, length of each codeword
        % Base_         : code alphabet size (2 = binary)
        % CoderName_    : name of the algorithm that produced this table
        Symbols_   (:,1) cell
        Probs_     (1,:) double
        Codewords_ (:,1) cell
        Lengths_   (1,:) double
        Base_      (1,1) double
        CoderName_ (1,:) char
    end

    % ------------------------------------------------------------------ %
    methods (Static)
        function obj = build(symbols, probs, codewords, options)
            % BUILD  Construct a validated CodeTable.
            %
            % INPUTS:
            %   symbols   - (K x 1) or (1 x K) cell of char/string labels
            %   probs     - (K x 1) or (1 x K) non-negative values summing to 1
            %   codewords - (K x 1) or (1 x K) cell of char vectors
            %
            % Name-Value Options:
            %   Base      (double): code alphabet size. Default: 2 (binary).
            %   CoderName (char):   name of the algorithm. Default: 'Unknown'.
            arguments
                symbols   (:,1) cell
                probs     (1,:) double {mustBeNonnegative}
                codewords (:,1) cell
                options.Base      (1,1) double {mustBePositive, mustBeInteger} = 2
                options.CoderName (1,:) char = 'Unknown'
            end
            % Validate
            K = numel(symbols);
            if K ~= numel(probs)
                error('OceanMath:Information:Coding:CodeTable:SizeMismatch', ...
                    'symbols and probs must have the same length. Got %d vs %d.', ...
                    K, numel(probs));
            end
            if K ~= numel(codewords)
                error('OceanMath:Information:Coding:CodeTable:SizeMismatch', ...
                    'symbols and codewords must have the same length. Got %d vs %d.', ...
                    K, numel(codewords));
            end
            if abs(sum(probs) - 1.0) > 1e-9
                error('OceanMath:Information:Coding:CodeTable:InvalidPMF', ...
                    'Probabilities must sum to 1. Got %.10f.', sum(probs));
            end
            % Ensure column cells
            symbols_col   = symbols(:);
            codewords_col = codewords(:);
            % Compute lengths from codewords
            lens = cellfun(@numel, codewords_col)';   % (1 x K)
            % Construct
            obj = OceanMath.Information.Coding.CodeTable( ...
                symbols_col, probs(:)', codewords_col, lens, ...
                options.Base, options.CoderName);
        end
    end

    % ------------------------------------------------------------------ %
    methods
        % ---- Core information-theoretic metrics ---- %

        function L = expectedLength(obj)
            % EXPECTEDLENGTH  L_avg = sum_k p_k * len_k  (in code symbols per source symbol).
            %
            % OUTPUTS:
            %   L - scalar double, average codeword length
            L = sum(obj.Probs_ .* obj.Lengths_);
        end

        function H = entropyBits(obj)
            % ENTROPYBITS  H(X) in bits (log base 2) of the source distribution.
            %
            % Used internally by redundancy(), efficiency(), codingGap().
            p = obj.Probs_(obj.Probs_ > 0);
            H = -sum(p .* log2(p));
        end

        function R = redundancy(obj)
            % REDUNDANCY  R = L_avg - H(X)/log2(Base)  (in code symbols per source symbol).
            %
            % For binary codes (Base=2): R = L_avg - H_bits.
            % Shannon's theorem guarantees R >= 0, and Huffman gives R < 1.
            %
            % OUTPUTS:
            %   R - scalar double >= 0
            H_base = obj.entropyBits() / log2(obj.Base_);
            R = obj.expectedLength() - H_base;
        end

        function eta = efficiency(obj)
            % EFFICIENCY  eta = H(X) / (L_avg * log2(Base))  in [0, 1].
            %
            % eta = 1 means the code achieves the entropy bound exactly.
            % eta < 1 means there is redundancy.
            %
            % OUTPUTS:
            %   eta - scalar double in (0, 1]
            L = obj.expectedLength();
            if L < 1e-12
                eta = 1.0;   % degenerate single-symbol case
                return;
            end
            eta = obj.entropyBits() / (L * log2(obj.Base_));
            eta = max(0.0, min(1.0, eta));   % numerical clamp
        end

        function K = kraftSum(obj)
            % KRAFTSUM  K = sum_k Base^(-len_k).
            %
            % Kraft inequality: a uniquely decodable code must have K <= 1.
            % A complete prefix-free code has K = 1.
            %
            % OUTPUTS:
            %   K - scalar double in (0, 1]
            K = sum(obj.Base_ .^ (-obj.Lengths_));
        end

        function tf = isComplete(obj)
            % ISCOMPLETE  True if the code is a complete prefix-free code (K = 1).
            tf = abs(obj.kraftSum() - 1.0) < 1e-9;
        end

        function gap = codingGap(obj)
            % CODINGGAP  L_avg - H(X) in bits (for binary codes).
            %
            % Always >= 0 (Shannon's theorem). Huffman achieves gap < 1 bit.
            % Identical to redundancy() for Base=2.
            %
            % OUTPUTS:
            %   gap - scalar double >= 0
            gap = obj.expectedLength() - obj.entropyBits() / log2(obj.Base_);
        end

        % ---- Accessors ---- %

        function syms = getSymbols(obj)
            % GETSYMBOLS  Return (K x 1) cell of symbol labels.
            syms = obj.Symbols_;
        end

        function p = getProbabilities(obj)
            % GETPROBABILITIES  Return (1 x K) probability vector.
            p = obj.Probs_;
        end

        function cw = getCodewords(obj)
            % GETCODEWORDS  Return (K x 1) cell of codeword strings.
            cw = obj.Codewords_;
        end

        function l = getLengths(obj)
            % GETLENGTHS  Return (1 x K) codeword length vector.
            l = obj.Lengths_;
        end

        function l_opt = getOptimalLengths(obj)
            % GETOPTIMALLENGTHS  Return (1 x K) theoretical optimal lengths -log_Base(p_k).
            %
            % These are the Shannon ideal lengths. Huffman codes round up
            % to the nearest integer, creating the < 1 bit redundancy gap.
            p = obj.Probs_;
            l_opt = -log(p) / log(obj.Base_);
        end

        function n = getSize(obj)
            % GETSIZE  Number of symbols in the alphabet.
            n = numel(obj.Symbols_);
        end

        % ---- Display ---- %

        function printSummary(obj)
            % PRINTSUMMARY  Print a formatted code table to the command window.
            K = obj.getSize();
            fprintf('\n=== CodeTable [%s] — %d symbols, Base-%d ===\n', ...
                obj.CoderName_, K, obj.Base_);
            fprintf('%-12s  %-8s  %-12s  %-6s  %-10s\n', ...
                'Symbol', 'Prob', 'Codeword', 'Len', 'Opt.Len');
            fprintf('%s\n', repmat('-', 1, 56));
            l_opt = obj.getOptimalLengths();
            for k = 1:K
                sym = obj.Symbols_{k};
                if ~ischar(sym), sym = num2str(sym); end
                fprintf('%-12s  %-8.4f  %-12s  %-6d  %-10.3f\n', ...
                    sym, obj.Probs_(k), obj.Codewords_{k}, ...
                    obj.Lengths_(k), l_opt(k));
            end
            fprintf('%s\n', repmat('-', 1, 56));
            fprintf('  H(X)             = %.4f bits\n', obj.entropyBits());
            fprintf('  L_avg            = %.4f code symbols/symbol\n', obj.expectedLength());
            fprintf('  Redundancy R     = %.4f\n', obj.redundancy());
            fprintf('  Efficiency eta   = %.4f (%.1f%%)\n', obj.efficiency(), obj.efficiency()*100);
            fprintf('  Kraft sum K      = %.6f  (complete: %s)\n', ...
                obj.kraftSum(), mat2str(obj.isComplete()));
            fprintf('\n');
        end

        function s = toStruct(obj)
            % TOSTRUCT  Export as a plain serializable struct.
            s = struct( ...
                'Symbols',    {obj.Symbols_}, ...
                'Probs',      obj.Probs_, ...
                'Codewords',  {obj.Codewords_}, ...
                'Lengths',    obj.Lengths_, ...
                'Base',       obj.Base_, ...
                'CoderName',  obj.CoderName_, ...
                'ExpectedLength', obj.expectedLength(), ...
                'Entropy',    obj.entropyBits(), ...
                'Redundancy', obj.redundancy(), ...
                'Efficiency', obj.efficiency(), ...
                'KraftSum',   obj.kraftSum());
        end

        function disp(obj)
            % DISP  Compact one-line display.
            fprintf('CodeTable  [%s, %d symbols, Base-%d] — L=%.4f, H=%.4f, eta=%.2f%%\n', ...
                obj.CoderName_, obj.getSize(), obj.Base_, ...
                obj.expectedLength(), obj.entropyBits(), obj.efficiency()*100);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function obj = CodeTable(symbols, probs, codewords, lengths, base, coderName)
            % Private constructor — use CodeTable.build() static factory.
            obj.Symbols_   = symbols;
            obj.Probs_     = probs;
            obj.Codewords_ = codewords;
            obj.Lengths_   = lengths;
            obj.Base_      = base;
            obj.CoderName_ = coderName;
        end
    end
end
