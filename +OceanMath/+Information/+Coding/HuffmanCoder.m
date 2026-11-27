classdef HuffmanCoder < OceanMath.Information.Coding.ISourceCoder
    % HUFFMANCODER  Huffman (1952) optimal prefix-free source coder.
    %
    % Implements the classic bottom-up greedy merge algorithm that builds
    % an optimal prefix-free binary (or k-ary) code minimizing L_avg.
    %
    % Shannon's theorem guarantees: H(X) <= L_avg < H(X) + 1 for binary.
    % No other uniquely decodable code has a smaller expected length.
    %
    % Construction:
    %   coder = OceanMath.Information.Coding.HuffmanCoder()
    %   coder = OceanMath.Information.Coding.HuffmanCoder('Base', 2)
    %
    % Usage:
    %   % Encode — returns a CodeTable
    %   table = coder.encode({'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
    %   table.printSummary()
    %
    %   % Build and inspect the tree
    %   tree = coder.buildTree({'a','b','c'}, [0.5, 0.3, 0.2]);
    %   tree.depth()           % max codeword length
    %   tree.numLeaves()       % alphabet size
    %
    % Name-Value Options (constructor):
    %   Base (double): code alphabet size. Default: 2 (binary).
    %                  Use 3 for ternary, 4 for quaternary, etc.
    %                  Note: for k-ary Huffman (k>2), the alphabet may need
    %                  (K-1) mod (k-1) == 0 dummy symbols for full balance.
    %                  This coder adds them automatically if needed.
    %
    % See also: OceanMath.Information.Coding.ISourceCoder
    %           OceanMath.Information.Coding.HuffmanTree
    %           OceanMath.Information.Coding.CodeTable

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Base_  (1,1) double = 2   % code alphabet size
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = HuffmanCoder(options)
            % HUFFMANCODER  Construct with optional code base.
            %
            % Name-Value Options:
            %   Base (double): code alphabet size. Default: 2.
            arguments
                options.Base (1,1) double {mustBePositive, mustBeInteger} = 2
            end
            if options.Base < 2
                error('OceanMath:Information:Coding:HuffmanCoder:InvalidBase', ...
                    'Code base must be >= 2. Got %d.', options.Base);
            end
            obj.Base_ = options.Base;
        end

        % ---- ISourceCoder interface ------------------------------------ %

        function table = encode(obj, symbols, probs)
            % ENCODE  Build Huffman code and return a CodeTable.
            %
            % INPUTS:
            %   symbols - (K x 1) or (1 x K) cell of char/string labels
            %   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1
            %
            % OUTPUTS:
            %   table - OceanMath.Information.Coding.CodeTable
            %
            % For k-ary codes: dummy symbols with probability 0 are added
            % automatically to pad the alphabet to satisfy (K-1) mod (k-1)==0.
            arguments
                obj
                symbols (:,1) cell
                probs   (1,:) double {mustBeNonnegative}
            end
            obj.validateInputs_(symbols, probs);

            % Pad for k-ary if needed
            [symbols_padded, probs_padded] = obj.padForKAry_(symbols, probs);

            tree = OceanMath.Information.Coding.HuffmanTree('Base', obj.Base_);
            tree.build(symbols_padded, probs_padded);

            % Extract CodeTable for original symbols only (no dummy padding)
            fullTable = tree.getCodeTable();
            K_orig = numel(symbols);

            if numel(symbols_padded) == K_orig
                table = fullTable;
            else
                % Remove dummy entries from padded table
                allSyms = fullTable.getSymbols();
                allProbs = fullTable.getProbabilities();
                allCodes = fullTable.getCodewords();
                table = OceanMath.Information.Coding.CodeTable.build( ...
                    allSyms(1:K_orig), allProbs(1:K_orig) / sum(allProbs(1:K_orig)), ...
                    allCodes(1:K_orig), ...
                    'Base', obj.Base_, ...
                    'CoderName', sprintf('Huffman (base-%d)', obj.Base_));
            end
        end

        function tree = buildTree(obj, symbols, probs)
            % BUILDTREE  Build and return the full HuffmanTree object.
            %
            % Useful for visualization via HuffmanTreeStrategy.
            %
            % INPUTS:
            %   symbols - (K x 1) or (1 x K) cell of char/string labels
            %   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1
            %
            % OUTPUTS:
            %   tree - OceanMath.Information.Coding.HuffmanTree (built)
            arguments
                obj
                symbols (:,1) cell
                probs   (1,:) double {mustBeNonnegative}
            end
            obj.validateInputs_(symbols, probs);
            [symbols_padded, probs_padded] = obj.padForKAry_(symbols, probs);

            tree = OceanMath.Information.Coding.HuffmanTree('Base', obj.Base_);
            tree.build(symbols_padded, probs_padded);
        end

        function name = getName(obj)
            % GETNAME  Return algorithm name.
            name = sprintf('Huffman (base-%d)', obj.Base_);
        end

        function tf = isOptimal(~)
            % ISOPTIMAL  Huffman is provably optimal among prefix-free codes.
            tf = true;
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function [syms_out, probs_out] = padForKAry_(obj, symbols, probs)
            % PADFORKARRY_  Add zero-probability dummy symbols to satisfy k-ary balance.
            %
            % For k-ary Huffman (k>2), valid if (K-1) mod (k-1) == 0.
            % If not, add ceil((k-1 - rem(K-1, k-1)) / 1) dummy symbols.
            k = obj.Base_;
            K = numel(symbols);

            if k == 2
                % Binary: no padding needed
                syms_out  = symbols(:);
                probs_out = probs(:)';
                return;
            end

            r = mod(K - 1, k - 1);
            if r == 0
                syms_out  = symbols(:);
                probs_out = probs(:)';
                return;
            end

            % Need to add (k-1-r) dummy symbols with probability 0
            nDummy = k - 1 - r;
            dummySyms  = arrayfun(@(i) sprintf('__dummy%d__', i), 1:nDummy, 'UniformOutput', false)';
            dummyProbs = zeros(1, nDummy);

            syms_out  = [symbols(:); dummySyms];
            probs_out = [probs(:)', dummyProbs];

            % Note: probs may no longer sum to 1 — re-normalize
            % Actually they still sum to 1 since we add 0's
        end
    end
end
