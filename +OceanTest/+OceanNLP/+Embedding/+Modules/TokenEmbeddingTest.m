classdef TokenEmbeddingTest < OceanTest.BaseTestCase
    % TOKENEMBEDDINGTEST  Unit tests for OceanNLP.Embedding.Modules.TokenEmbedding.
    %
    % HAND-WORKED LOOKUP-AND-SUM EXAMPLE (d=2, VocabSize=3, MaxPositions=2,
    % NumSegmentTypes=2):
    %   W_tok (3x2): row1(id0)=[1 0], row2(id1)=[0 1], row3(id2)=[2 2]
    %   W_pos (2x2): row1(pos0)=[10 10], row2(pos1)=[20 20]
    %   W_seg (2x2): row1(seg0)=[100 0], row2(seg1)=[0 100]
    %
    %   ids = [0 2] (0-indexed BERT ids), segmentIds = [0 1]
    %
    %   position t=1 (id=0, pos=0, seg=0):
    %     E(:,1) = W_tok(1,:)' + W_pos(1,:)' + W_seg(1,:)'
    %            = [1;0] + [10;10] + [100;0] = [111;10]
    %   position t=2 (id=2, pos=1, seg=1):
    %     E(:,2) = W_tok(3,:)' + W_pos(2,:)' + W_seg(2,:)'
    %            = [2;2] + [20;20] + [0;100] = [22;122]
    %
    %   Expected E = [111 22; 10 122]   (d x T = 2 x 2)

    properties
        WTok
        WPos
        WSeg
    end

    methods (TestClassSetup)
        function buildWeights(testCase)
            testCase.WTok = [1 0; 0 1; 2 2];
            testCase.WPos = [10 10; 20 20];
            testCase.WSeg = [100 0; 0 100];
        end
    end

    methods (Test)
        function testHandWorkedLookupAndSum(testCase)
            import OceanNLP.Embedding.Modules.TokenEmbedding
            emb = TokenEmbedding(testCase.WTok, testCase.WPos, testCase.WSeg);

            E = emb.forward([0 2], [0 1]);

            expected = [111 22; 10 122];
            testCase.verifyEqual(E, expected, 'AbsTol', 1e-12);
        end

        function testSegmentIdsDefaultToAllZeros(testCase)
            import OceanNLP.Embedding.Modules.TokenEmbedding
            emb = TokenEmbedding(testCase.WTok, testCase.WPos, testCase.WSeg);

            Eexplicit = emb.forward([0 2], [0 0]);
            Edefault  = emb.forward([0 2]);

            testCase.verifyEqual(Edefault, Eexplicit, 'AbsTol', 1e-12);
        end

        function testZeroIndexedIdConvention(testCase)
            % BERT id 0 must read MATLAB row 1 (not row 0, which doesn't
            % exist, and not some other row) -- verify directly against
            % the raw weight table.
            import OceanNLP.Embedding.Modules.TokenEmbedding
            emb = TokenEmbedding(testCase.WTok, testCase.WPos, testCase.WSeg);

            E = emb.forward(0, 0);   % single token, id=0, pos=0 (implicit), seg=0
            expected = testCase.WTok(1, :)' + testCase.WPos(1, :)' + testCase.WSeg(1, :)';
            testCase.verifyEqual(E, expected, 'AbsTol', 1e-12);
        end

        function testPositionConventionUsesSequencePositionNotTokenId(testCase)
            % Position is determined by the token's 0-indexed SEQUENCE slot
            % (t-1), not by its id -- feed the SAME id twice at two
            % different positions and confirm the position component differs.
            import OceanNLP.Embedding.Modules.TokenEmbedding
            emb = TokenEmbedding(testCase.WTok, testCase.WPos, testCase.WSeg);

            E = emb.forward([1 1], [0 0]);   % same token id at position 0 and 1
            col1 = testCase.WTok(2, :)' + testCase.WPos(1, :)' + testCase.WSeg(1, :)';
            col2 = testCase.WTok(2, :)' + testCase.WPos(2, :)' + testCase.WSeg(1, :)';
            testCase.verifyEqual(E, [col1 col2], 'AbsTol', 1e-12);
            testCase.verifyNotEqual(E(:,1), E(:,2));
        end

        function testIdOutOfRangeErrors(testCase)
            import OceanNLP.Embedding.Modules.TokenEmbedding
            emb = TokenEmbedding(testCase.WTok, testCase.WPos, testCase.WSeg);
            testCase.verifyError(@() emb.forward(3, 0), ...   % VocabSize=3 -> max valid id = 2
                'OceanNLP:Embedding:Modules:TokenEmbedding:IdOutOfRange');
        end

        function testHiddenDimMismatchErrorsAtConstruction(testCase)
            import OceanNLP.Embedding.Modules.TokenEmbedding
            badWPos = [1 2 3; 4 5 6];   % 3 columns, WTok has 2
            testCase.verifyError(@() TokenEmbedding(testCase.WTok, badWPos, testCase.WSeg), ...
                'OceanNLP:Embedding:Modules:TokenEmbedding:HiddenDimMismatch');
        end

        function testFeatureMajorOutputShape(testCase)
            import OceanNLP.Embedding.Modules.TokenEmbedding
            emb = TokenEmbedding(testCase.WTok, testCase.WPos, testCase.WSeg);
            E = emb.forward([0 2], [0 1]);
            testCase.verifySize(E, [2 2]);   % (d x T)
        end
    end
end
