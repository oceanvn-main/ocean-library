classdef HuffmanTree < handle
    % HUFFMANTREE  k-ary Huffman trie built during source code construction.
    %
    % Represents the complete Huffman trie as a DigraphCus graph.
    % Supports binary (Base=2), ternary (Base=3), and any k-ary code.
    % Each node stores metadata fields:
    %   .weight   (double)  — probability weight (sum of descendants' probs)
    %   .symbol   (char)    — symbol label for leaf nodes; '' for internal nodes
    %   .isLeaf   (logical) — true for leaf nodes, false for internal nodes
    %   .codeword (char)    — assigned codeword (leaf nodes only)
    %
    % The tree is directed: edges go from parent → child.
    % Child edges are labelled '0', '1', ..., 'k-1' (left to right by weight).
    %
    % Construction:
    %   tree = OceanMath.Information.Coding.HuffmanTree()
    %   tree = OceanMath.Information.Coding.HuffmanTree('Base', 3)
    %   tree.build(symbols, probs)         % run Huffman merge
    %   table = tree.getCodeTable()        % extract CodeTable
    %   g = tree.toDigraph()               % get underlying DigraphCus
    %
    % Usage (internal — called by HuffmanCoder):
    %   tree = OceanMath.Information.Coding.HuffmanTree('Base', 3);
    %   tree.build({'a','b','c'}, [0.5, 0.3, 0.2]);
    %   tree.getCodeTable().printSummary()
    %
    % See also: OceanMath.Information.Coding.HuffmanCoder
    %           OceanMath.Information.Coding.CodeTable
    %           OceanMath.Graph.DigraphCus

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Graph_     % OceanMath.Graph.DigraphCus — directed tree
        RootId_    (1,1) double = 0   % Node ID of root (0 = not built yet)
        CodeTable_ % OceanMath.Information.Coding.CodeTable (derived after build)
        Base_      (1,1) double = 2   % code alphabet size (branching factor)
        Built_     (1,1) logical = false
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = HuffmanTree(options)
            % HUFFMANTREE  Construct an empty tree shell.
            %
            % Name-Value Options:
            %   Base (double): code alphabet size. Default: 2 (binary).
            arguments
                options.Base (1,1) double {mustBePositive, mustBeInteger} = 2
            end
            obj.Base_   = options.Base;
            obj.Graph_  = OceanMath.Graph.DigraphCus('Directed', true);
        end

        function build(obj, symbols, probs)
            % BUILD  Run the Huffman bottom-up greedy merge algorithm (k-ary).
            %
            % Constructs the full tree in obj.Graph_ and derives the
            % CodeTable. After this call, getCodeTable() is available.
            %
            % INPUTS:
            %   symbols - (K x 1) cell of char/string symbol labels
            %   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1
            %
            % Algorithm (k-ary, k = obj.Base_):
            %   1. Create one leaf node per symbol (priority = prob)
            %   2. While > 1 node in queue:
            %      a. Extract k nodes with smallest weight
            %      b. Merge into a new internal node (weight = sum)
            %      c. Connect parent → each child with labels '0','1',...,'k-1'
            %   3. Traverse from root to assign codewords
            %
            % Note: For k > 2, caller (HuffmanCoder) pads the alphabet so
            % that (K-1) mod (k-1) == 0 before calling build().
            arguments
                obj
                symbols (:,1) cell
                probs   (1,:) double {mustBeNonnegative}
            end
            K = numel(symbols);
            k = obj.Base_;   % branching factor

            if K ~= numel(probs)
                error('OceanMath:Information:Coding:HuffmanTree:SizeMismatch', ...
                    'symbols and probs must have the same length. Got %d vs %d.', ...
                    K, numel(probs));
            end
            if K == 0
                error('OceanMath:Information:Coding:HuffmanTree:Empty', ...
                    'symbols must not be empty.');
            end
            if K == 1
                % Degenerate: single symbol gets codeword '0'
                sym = symbols{1};
                if ~ischar(sym), sym = char(string(sym)); end
                meta = struct('weight', probs(1), 'symbol', sym, ...
                              'isLeaf', true, 'codeword', '0');
                nid = obj.Graph_.addNode(sym, [0, 0], meta);
                obj.RootId_ = nid;
                codewords = {'0'};
                obj.CodeTable_ = OceanMath.Information.Coding.CodeTable.build( ...
                    symbols, probs, codewords, ...
                    'Base', k, 'CoderName', sprintf('Huffman (base-%d)', k));
                obj.Built_ = true;
                return;
            end

            % Pre-allocate graph nodes (upper bound)
            maxNodes = 2 * K;
            obj.Graph_.reserveNodes(maxNodes);

            % ---- Step 1: Create leaf nodes ----------------------------- %
            nodeIds = zeros(1, K);
            weights = probs(:)';           % (1 x K)

            for i = 1:K
                sym = symbols{i};
                if ~ischar(sym), sym = char(string(sym)); end
                meta = struct('weight', probs(i), 'symbol', sym, ...
                              'isLeaf', true, 'codeword', '');
                nodeIds(i) = obj.Graph_.addNode(sym, [0, 0], meta);
            end

            % ---- Step 2: Priority-queue k-ary Huffman merge ------------ %
            % Each merge step extracts k smallest and combines them.
            % For k=2 this is classic binary Huffman.
            activeIds = nodeIds;
            activeWts = weights;

            while numel(activeIds) > 1
                % Sort ascending (stable: ties broken by insertion order)
                [activeWts, sortIdx] = sort(activeWts, 'ascend');
                activeIds = activeIds(sortIdx);

                % Extract min(k, numel) nodes to merge
                nMerge = min(k, numel(activeIds));
                mergeIds = activeIds(1:nMerge);
                mergeWts = activeWts(1:nMerge);

                newWeight    = sum(mergeWts);
                internalName = sprintf('[%.4g]', newWeight);
                meta = struct('weight', newWeight, 'symbol', '', ...
                              'isLeaf', false, 'codeword', '');
                newId = obj.Graph_.addNode(internalName, [0, 0], meta);

                % Connect parent → each child with label '0','1','2',...
                for ci = 1:nMerge
                    lbl = num2str(ci - 1);    % '0' for lightest, '1', '2', ...
                    obj.Graph_.addEdge(newId, mergeIds(ci), newWeight, 'Label', lbl);
                end

                % Replace merged nodes with new internal node
                activeIds = [newId, activeIds(nMerge+1:end)];
                activeWts = [newWeight,  activeWts(nMerge+1:end)];
            end

            obj.RootId_ = activeIds(1);   % last node standing is root

            % ---- Step 3: Traverse tree to assign codewords ------------- %
            % Use containers.Map (handle class) so recursive mutations propagate.
            cwMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
            obj.assignCodes_(obj.RootId_, '', cwMap);

            % Collect codewords in original symbol order
            codewords = cell(K, 1);
            for i = 1:K
                sym = symbols{i};
                if ~ischar(sym), sym = char(string(sym)); end
                if cwMap.isKey(sym)
                    codewords{i} = cwMap(sym);
                else
                    codewords{i} = '';   % should not happen for a valid tree
                end
            end

            % ---- Step 4: Build CodeTable ------------------------------- %
            obj.CodeTable_ = OceanMath.Information.Coding.CodeTable.build( ...
                symbols, probs(:)', codewords, ...
                'Base', k, 'CoderName', sprintf('Huffman (base-%d)', k));

            obj.Built_ = true;
        end

        % ---- Public accessors ----------------------------------------- %

        function table = getCodeTable(obj)
            % GETCODETABLE  Return the derived CodeTable.
            %
            % Requires build() to have been called first.
            obj.assertBuilt_();
            table = obj.CodeTable_;
        end

        function g = toDigraph(obj)
            % TODIGRAPH  Return the underlying DigraphCus (for visualization).
            %
            % Nodes contain metadata: weight, symbol, isLeaf, codeword.
            % Edge labels encode the bit ('0' or '1').
            obj.assertBuilt_();
            g = obj.Graph_;
        end

        function id = getRootId(obj)
            % GETROOTID  Return the integer ID of the root node.
            obj.assertBuilt_();
            id = obj.RootId_;
        end

        function d = depth(obj)
            % DEPTH  Maximum root-to-leaf path length (= max codeword length).
            obj.assertBuilt_();
            d = max(obj.CodeTable_.getLengths());
        end

        function n = numLeaves(obj)
            % NUMLEAVES  Number of leaf nodes (= alphabet size K).
            obj.assertBuilt_();
            n = obj.CodeTable_.getSize();
        end

        function n = numInternal(obj)
            % NUMINTERNAL  Number of internal (non-leaf) nodes.
            obj.assertBuilt_();
            totalNodes = obj.Graph_.getNodeCount();
            n = totalNodes - obj.numLeaves();
        end

        function disp(obj)
            % DISP  Compact one-line summary.
            if ~obj.Built_
                fprintf('HuffmanTree  [not built]\n');
            else
                fprintf('HuffmanTree  [%d leaves, %d internal, depth=%d, Base-%d]\n', ...
                    obj.numLeaves(), obj.numInternal(), obj.depth(), obj.Base_);
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function assignCodes_(obj, nodeId, prefix, cwMap)
            % ASSIGNCODES_  Recursive DFS to assign codewords to leaf nodes.
            %
            % Uses a containers.Map (handle class) as accumulator so that
            % assignments inside recursive calls are visible to the caller.
            % This avoids MATLAB's pass-by-value semantics on cell arrays.
            %
            % INPUTS:
            %   nodeId - integer node ID in obj.Graph_
            %   prefix - char, accumulated bit string so far
            %   cwMap  - containers.Map(symbol -> codeword), mutated in-place
            node = obj.Graph_.getNode(nodeId);
            if isempty(node), return; end
            meta = node.Metadata;

            if meta.isLeaf
                % Store codeword: use symbol as key
                cwMap(meta.symbol) = prefix;
                return;
            end

            % Get children (out-neighbors in directed graph)
            children = obj.Graph_.neighbors(nodeId);
            for ci = 1:numel(children)
                childId  = children(ci);
                edgeIdx  = obj.Graph_.getEdgeId(nodeId, childId);
                if edgeIdx > 0
                    edgeLabel = obj.Graph_.getEdge(edgeIdx).Label;
                    obj.assignCodes_(childId, [prefix, edgeLabel], cwMap);
                end
            end
        end

        function assertBuilt_(obj)
            if ~obj.Built_
                error('OceanMath:Information:Coding:HuffmanTree:NotBuilt', ...
                    'Call build(symbols, probs) before accessing tree results.');
            end
        end
    end
end
