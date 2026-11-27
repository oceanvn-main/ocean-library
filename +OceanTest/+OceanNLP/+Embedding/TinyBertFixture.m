classdef TinyBertFixture
    % TINYBERTFIXTURE  Shared synthetic BERT-family fixture for tests
    % (d=4, 2 heads, d_ff=8, 1 layer, 12-token vocab). Weights are seeded
    % random, not trained: tests assert structure and independently
    % recomputed values, never model quality.
    %
    % vocab ids: 0 [PAD] 1 [UNK] 2 [CLS] 3 [SEP] 4 [MASK] 5 [unused0]
    %            6 [unused1] 7 cat 8 dog 9 , 10 . 11 a

    methods (Static)
        function vocabFile = writeVocab()
            vocabFile = string(tempname) + ".txt";
            toks = ["[PAD]" "[UNK]" "[CLS]" "[SEP]" "[MASK]" "[unused0]" "[unused1]" "cat" "dog" "," "." "a"];
            fid = fopen(vocabFile, 'w');
            fprintf(fid, '%s\n', toks);
            fclose(fid);
        end

        function W = weights(seed, numLayers)
            arguments
                seed (1,1) double = 1
                numLayers (1,1) double = 1
            end
            rng(seed, 'twister');
            d = 4; dff = 8; V = 12; P = 16;
            W.TokEmbedding.WordEmbeddings = randn(V, d);
            W.TokEmbedding.PositionEmbeddings = randn(P, d);
            W.TokEmbedding.TokenTypeEmbeddings = randn(2, d);
            W.TokEmbedding.LayerNormGamma = 1 + 0.1 * randn(d, 1);
            W.TokEmbedding.LayerNormBeta = 0.1 * randn(d, 1);
            for i = 1:numLayers
                L.WQ = randn(d, d); L.bQ = 0.1 * randn(d, 1);
                L.WK = randn(d, d); L.bK = 0.1 * randn(d, 1);
                L.WV = randn(d, d); L.bV = 0.1 * randn(d, 1);
                L.WO = randn(d, d); L.bO = 0.1 * randn(d, 1);
                L.NumHeads = 2;
                L.W1 = randn(dff, d); L.b1 = 0.1 * randn(dff, 1);
                L.W2 = randn(d, dff); L.b2 = 0.1 * randn(d, 1);
                L.LN1Gamma = 1 + 0.1 * randn(d, 1); L.LN1Beta = 0.1 * randn(d, 1);
                L.LN2Gamma = 1 + 0.1 * randn(d, 1); L.LN2Beta = 0.1 * randn(d, 1);
                Layers(i) = L; %#ok<AGROW>
            end
            W.Layers = Layers;
            W.Pooler.W = randn(d, d);
            W.Pooler.b = 0.1 * randn(d, 1);
            W.Classifier.W = randn(1, d);
            W.Classifier.b = 0.1;
            W.Projection.W = randn(3, d);
        end
    end
end
