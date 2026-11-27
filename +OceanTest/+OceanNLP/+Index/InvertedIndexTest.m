classdef InvertedIndexTest < OceanTest.BaseTestCase
    % INVERTEDINDEXTEST  Unit tests for OceanNLP.Index.InvertedIndex.
    %
    % HAND-COUNTED FIXTURE CORPUS (WhitespaceTokenizer, lowercases,
    % strips only edge punctuation -- none present here so tokenization is
    % literal whitespace-split):
    %   doc1: "the cat sat on the mat"  -> the,cat,sat,on,the,mat  (6 tokens, "the" x2)
    %   doc2: "the dog sat"             -> the,dog,sat             (3 tokens)
    %   doc3: "a cat and a dog"         -> a,cat,and,a,dog         (5 tokens, "a" x2)
    %
    % Pooled counts: the=3, cat=2, sat=2, dog=2, a=2, on=1, mat=1, and=1.
    % Vocabulary.fit() orders by descending frequency, ties alphabetical
    % ascending (see Vocabulary's own DETERMINISTIC ORDERING policy):
    %   id1 = <unk> (OOV, reserved)
    %   id2 = the   (freq 3)
    %   id3 = a     (freq 2, alphabetically first among the freq-2 group)
    %   id4 = cat   (freq 2)
    %   id5 = dog   (freq 2)
    %   id6 = sat   (freq 2)
    %   id7 = and   (freq 1, alphabetically first among the freq-1 group)
    %   id8 = mat   (freq 1)
    %   id9 = on    (freq 1)
    % VocabSize = 9 (includes the reserved OOV slot).
    %
    % Document frequencies: the=2 (doc1,doc2), a=1 (doc3), cat=2 (doc1,doc3),
    % dog=2 (doc2,doc3), sat=2 (doc1,doc2), and=1 (doc3), mat=1 (doc1), on=1 (doc1).
    % Document lengths: doc1=6, doc2=3, doc3=5. Average = 14/3.

    properties
        Corpus
        Index
    end

    methods (TestClassSetup)
        function buildIndex(testCase)
            import OceanNLP.Index.InvertedIndex
            testCase.Corpus = [ ...
                "the cat sat on the mat", ...
                "the dog sat", ...
                "a cat and a dog"];
            testCase.Index = InvertedIndex();
            testCase.Index.fit(testCase.Corpus);
        end
    end

    methods (Test)
        function testVocabularySizeIncludesOOVSlot(testCase)
            testCase.verifyEqual(testCase.Index.vocabularySize(), 9);
        end

        function testNumDocuments(testCase)
            testCase.verifyEqual(testCase.Index.numDocuments(), 3);
        end

        function testPostingsForCatMatchesHandCount(testCase)
            catId = testCase.Index.Vocabulary.encode("cat");
            [docIds, counts] = testCase.Index.postings(catId);
            testCase.verifyEqual(docIds, [1 3]);
            testCase.verifyEqual(counts, [1 1]);
        end

        function testPostingsForTheMatchesHandCountIncludingRepeatedTerm(testCase)
            % "the" occurs twice in doc1 -- verify the posting count
            % reflects the raw TERM COUNT (2), not just document presence.
            theId = testCase.Index.Vocabulary.encode("the");
            [docIds, counts] = testCase.Index.postings(theId);
            testCase.verifyEqual(docIds, [1 2]);
            testCase.verifyEqual(counts, [2 1]);   % doc1 has "the" twice, doc2 once
        end

        function testDocumentFrequencyMatchesHandCounts(testCase)
            idx = testCase.Index;
            v = idx.Vocabulary;
            testCase.verifyEqual(idx.documentFrequency(v.encode("the")), 2);
            testCase.verifyEqual(idx.documentFrequency(v.encode("a")),   1);
            testCase.verifyEqual(idx.documentFrequency(v.encode("cat")), 2);
            testCase.verifyEqual(idx.documentFrequency(v.encode("dog")), 2);
            testCase.verifyEqual(idx.documentFrequency(v.encode("sat")), 2);
            testCase.verifyEqual(idx.documentFrequency(v.encode("and")), 1);
            testCase.verifyEqual(idx.documentFrequency(v.encode("mat")), 1);
            testCase.verifyEqual(idx.documentFrequency(v.encode("on")),  1);
        end

        function testDocumentLengthMatchesHandCounts(testCase)
            idx = testCase.Index;
            testCase.verifyEqual(idx.documentLength(1), 6);
            testCase.verifyEqual(idx.documentLength(2), 3);
            testCase.verifyEqual(idx.documentLength(3), 5);
        end

        function testAverageDocumentLength(testCase)
            testCase.verifyEqual(testCase.Index.averageDocumentLength(), 14/3, 'AbsTol', 1e-12);
        end

        function testSparseByConstructionPostingLengthEqualsDocFrequencyNotNumDocuments(testCase)
            % The sparse-safety rule for this non-matrix structure: a
            % term's posting list has EXACTLY documentFrequency(termId)
            % entries, never numDocuments() entries with zero-padding.
            idx = testCase.Index;
            theId = idx.Vocabulary.encode("the");
            [docIds, ~] = idx.postings(theId);
            testCase.verifyEqual(numel(docIds), idx.documentFrequency(theId));
            testCase.verifyNotEqual(numel(docIds), idx.numDocuments());   % 2 ~= 3
        end

        function testOOVTermIdReturnsEmptyPostingsAndZeroDocFrequency(testCase)
            idx = testCase.Index;
            oovId = idx.Vocabulary.OOVId;
            [docIds, counts] = idx.postings(oovId);
            testCase.verifyEmpty(docIds);
            testCase.verifyEmpty(counts);
            testCase.verifyEqual(idx.documentFrequency(oovId), 0);
        end

        function testTermIdOutOfRangeErrorsDistinctFromOOV(testCase)
            % OOV (in-range, id 1) returns empty; a termId beyond
            % vocabularySize() is a caller bug and must ERROR instead --
            % these are two distinct policies, verified as distinct here.
            idx = testCase.Index;
            badId = idx.vocabularySize() + 1;
            testCase.verifyError(@() idx.postings(badId), ...
                'OceanNLP:Index:InvertedIndex:TermIdOutOfRange');
        end

        function testDocIdOutOfRangeErrors(testCase)
            idx = testCase.Index;
            badDocId = idx.numDocuments() + 1;
            testCase.verifyError(@() idx.documentLength(badDocId), ...
                'OceanNLP:Index:InvertedIndex:DocIdOutOfRange');
        end

        function testQueryBeforeFitErrors(testCase)
            import OceanNLP.Index.InvertedIndex
            freshIdx = InvertedIndex();
            testCase.verifyError(@() freshIdx.postings(2), ...
                'OceanNLP:Index:InvertedIndex:NotFitted');
        end

        function testIsFitted(testCase)
            import OceanNLP.Index.InvertedIndex
            freshIdx = InvertedIndex();
            testCase.verifyFalse(freshIdx.isFitted());
            testCase.verifyTrue(testCase.Index.isFitted());
        end
    end
end
