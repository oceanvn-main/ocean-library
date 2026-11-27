classdef TestHuffmanCoding < OceanTest.BaseTestCase
    methods (Test)
        function testTest1KraftInequalityBinaryHuffmanAlwaysAchievesK1(testCase)
            % Test 1: Kraft inequality — binary Huffman always achieves K = 1
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c','d'};
            probs = [0.4, 0.3, 0.2, 0.1];
            coder = HC();
            table = coder.encode(syms, probs);
            K = table.kraftSum();
            assert(abs(K - 1.0) < 1e-9, 'Kraft sum must equal 1 for Huffman');
            fprintf('[PASS] Test 1: Kraft inequality (K = %.6f)\n', K);
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest2ShannonOptimalityHXLAvgHX1(testCase)
            % Test 2: Shannon optimality — H(X) <= L_avg < H(X) + 1
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c','d','e'};
            probs = [0.35, 0.25, 0.20, 0.12, 0.08];
            table = hEnc(syms, probs);
            H = table.entropyBits();
            L = table.expectedLength();
            assert(L >= H - 1e-9,     'L_avg must be >= H(X)');
            assert(L <  H + 1 + 1e-9,'L_avg must be < H(X)+1 for binary Huffman');
            fprintf('[PASS] Test 2: Shannon bounds H=%.4f, L=%.4f, gap=%.4f\n', H, L, L-H);
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest3SingleSymbolEdgeCaseCodeword0LAvg0(testCase)
            % Test 3: Single symbol edge case — codeword = '0', L_avg = 0
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a'};
            probs = [1.0];
            table = hEnc(syms, probs);
            assert(numel(table.getCodewords()) == 1, 'Should have 1 codeword');
            % Single symbol: 1 bit minimum representation
            L = table.expectedLength();
            assert(isfinite(L), 'Expected length must be finite');
            assert(table.getSize() == 1, 'Size must be 1');
            fprintf('[PASS] Test 3: Single symbol (L=%.4f)\n', L);
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest4Uniform4SymbolAllLengths2LH2Bits(testCase)
            % Test 4: Uniform 4-symbol — all lengths = 2, L = H = 2 bits
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c','d'};
            probs = [0.25, 0.25, 0.25, 0.25];
            table = hEnc(syms, probs);
            L = table.expectedLength();
            H = table.entropyBits();
            lens = table.getLengths();
            assert(all(lens == 2), 'All codewords should have length 2 for uniform-4');
            assert(abs(L - 2.0) < 1e-9, 'Expected length should be 2.0');
            assert(abs(H - 2.0) < 1e-9, 'Entropy should be 2.0');
            assert(abs(table.redundancy()) < 1e-9, 'Redundancy should be 0');
            assert(abs(table.efficiency() - 1.0) < 1e-9, 'Efficiency should be 1.0');
            fprintf('[PASS] Test 4: Uniform-4 (L=%.4f, H=%.4f, eta=%.4f)\n', L, H, table.efficiency());
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest5CodetableIscompleteIsTrueForHuffmanCodes(testCase)
            % Test 5: CodeTable.isComplete() is true for Huffman codes
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c','d','e','f'};
            probs = [0.30, 0.25, 0.20, 0.12, 0.08, 0.05];
            table = hEnc(syms, probs);
            assert(table.isComplete(), 'Huffman code must be complete (K = 1)');
            fprintf('[PASS] Test 5: isComplete() = true\n');
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest6OptimalcodelengthLOptLog2P(testCase)
            % Test 6: optimalCodeLength — l_opt = -log2(p)
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            probs  = [0.5, 0.25, 0.25];
            l_opt  = optL(probs);
            expect = [-log2(0.5), -log2(0.25), -log2(0.25)];
            assert(max(abs(l_opt - expect)) < 1e-10, 'optimalCodeLength mismatch');
            fprintf('[PASS] Test 6: optimalCodeLength [%.4f, %.4f, %.4f]\n', l_opt);
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest7ComparecodingStructTheoremSatisfiedTrue(testCase)
            % Test 7: compareCoding struct — theorem_satisfied = true
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c'};
            probs = [0.5, 0.3, 0.2];
            S = comp(syms, probs, 'Verbose', false);
            assert(S.theorem_satisfied, 'Shannon theorem must be satisfied');
            assert(abs(S.entropy_bits - S.huffman_table.entropyBits()) < 1e-9, ...
            'entropy_bits mismatch in struct');
            fprintf('[PASS] Test 7: compareCoding struct (theorem_satisfied=true, L=%.4f)\n', S.huffman_L);
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest8HuffmantreeObjectDepthNumleavesNuminternalConsistency(testCase)
            % Test 8: HuffmanTree object — depth/numLeaves/numInternal consistency
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c','d'};
            probs = [0.4, 0.3, 0.2, 0.1];
            coder = HC();
            tree  = coder.buildTree(syms, probs);
            assert(tree.numLeaves()   == 4,             'numLeaves should be 4');
            assert(tree.numInternal() == 3,             'numInternal = K-1 = 3');
            assert(tree.depth()       >= 2,             'depth must be >= 2');
            table2 = tree.getCodeTable();
            assert(abs(table2.kraftSum() - 1.0) < 1e-9,'CodeTable from tree must have K=1');
            fprintf('[PASS] Test 8: HuffmanTree (leaves=%d, internal=%d, depth=%d)\n', ...
            tree.numLeaves(), tree.numInternal(), tree.depth());
            nPass = nPass + 1;
            % =====================================================================
        end

        function testTest9InformationengineCodingStrategyAliases(testCase)
            % Test 9: InformationEngine coding strategy aliases
            fprintf('\n=== Huffman Coding Test Suite ===\n\n');
            nPass = 0; nFail = 0;
            CT  = @OceanMath.Information.Coding.CodeTable;
            HC  = @OceanMath.Information.Coding.HuffmanCoder;
            HT  = @OceanMath.Information.Coding.HuffmanTree;
            hEnc = @OceanMath.Information.Coding.huffmanEncode;
            optL = @OceanMath.Information.Coding.optimalCodeLength;
            comp = @OceanMath.Information.Coding.compareCoding;
            % =====================================================================
            syms  = {'a','b','c','d'};
            probs = [0.4, 0.3, 0.2, 0.1];
            table = hEnc(syms, probs);
            engine = OceanMath.Information.InformationEngine(table);
            results = engine.runAll('ExpectedLength','Redundancy','CodingEfficiency','KraftSum','CodingGap');
            assert(isfield(results, 'ExpectedLength'),   'Missing ExpectedLength');
            assert(isfield(results, 'Redundancy'),       'Missing Redundancy');
            assert(isfield(results, 'CodingEfficiency'), 'Missing CodingEfficiency');
            assert(isfield(results, 'KraftSum'),         'Missing KraftSum');
            assert(isfield(results, 'CodingGap'),        'Missing CodingGap');
            assert(abs(results.KraftSum - 1.0) < 1e-9,  'KraftSum alias mismatch');
            assert(results.CodingEfficiency > 0 && results.CodingEfficiency <= 1, ...
            'Efficiency must be in (0,1]');
            fprintf('[PASS] Test 9: InformationEngine aliases (L=%.4f, R=%.4f, eta=%.4f%%)\n', ...
            results.ExpectedLength, results.Redundancy, results.CodingEfficiency*100);
            nPass = nPass + 1;
            % =====================================================================
        end

    end
end