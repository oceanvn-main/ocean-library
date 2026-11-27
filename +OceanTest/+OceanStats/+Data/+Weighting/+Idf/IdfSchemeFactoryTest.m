classdef IdfSchemeFactoryTest < OceanTest.BaseTestCase
    % IDFSCHEMEFACTORYTEST  Unit tests for the IIdfScheme Strategy family:
    %   IdfSchemeFactory, SmoothIdf, ClassicIdf, BM25Idf.

    properties (Constant)
        Df = [2 2 1 1]   % document frequency per term, 4-term toy vocab
        N  = 3           % total documents
    end

    methods (Test)
        function testSmoothIdfMatchesHandFormula(testCase)
            import OceanStats.Data.Weighting.Idf.*
            scheme = IdfSchemeFactory.create('smooth');
            idf = scheme.compute(testCase.Df, testCase.N);
            expected = log((1 + testCase.N) ./ (1 + testCase.Df)) + 1;

            testCase.verifyEqual(idf, expected, 'AbsTol', 1e-12);
            testCase.verifyEqual(scheme.name(), "smooth");
            testCase.verifyClass(scheme, 'OceanStats.Data.Weighting.Idf.SmoothIdf');
        end

        function testClassicIdfMatchesTextbookFormula(testCase)
            % Sparck Jones (1972) / Manning-Raghavan-Schutze eq. 11.5.
            import OceanStats.Data.Weighting.Idf.*
            scheme = IdfSchemeFactory.create('classic');
            idf = scheme.compute(testCase.Df, testCase.N);
            expected = log10(testCase.N ./ testCase.Df);

            testCase.verifyEqual(idf, expected, 'AbsTol', 1e-12);
            testCase.verifyEqual(scheme.name(), "classic");
            % Un-smoothed: a term in every document gets idf exactly 0.
            testCase.verifyEqual(log10(3/3), 0, 'AbsTol', 1e-12);
        end

        function testBm25IdfMatchesLuceneFormula(testCase)
            import OceanStats.Data.Weighting.Idf.*
            scheme = IdfSchemeFactory.create('bm25');
            idf = scheme.compute(testCase.Df, testCase.N);
            expected = log((testCase.N - testCase.Df + 0.5) ./ (testCase.Df + 0.5) + 1);

            testCase.verifyEqual(idf, expected, 'AbsTol', 1e-12);
            testCase.verifyEqual(scheme.name(), "bm25");
        end

        function testCaseInsensitiveMatching(testCase)
            import OceanStats.Data.Weighting.Idf.*
            scheme = IdfSchemeFactory.create('  ClAsSiC  ');
            testCase.verifyClass(scheme, 'OceanStats.Data.Weighting.Idf.ClassicIdf');
        end

        function testUnknownNameErrors(testCase)
            import OceanStats.Data.Weighting.Idf.*
            testCase.verifyError(@() IdfSchemeFactory.create('bogus'), ...
                'OceanStats:Data:Weighting:Idf:IdfSchemeFactory:Unknown');
        end

        function testValidNames(testCase)
            import OceanStats.Data.Weighting.Idf.*
            names = IdfSchemeFactory.validNames();

            testCase.verifyClass(names, 'cell');
            testCase.verifyTrue(all(ismember({'smooth', 'classic', 'bm25'}, names)));
        end

        function testEverySchemeHasNonEmptySymbolicLatex(testCase)
            import OceanStats.Data.Weighting.Idf.*
            for name = IdfSchemeFactory.validNames()
                scheme = IdfSchemeFactory.create(name{1});
                testCase.verifyClass(scheme.symbolicLatex(), 'string');
                testCase.verifyGreaterThan(strlength(scheme.symbolicLatex()), 0);
            end
        end

        function testIIdfSchemeCannotBeInstantiatedDirectly(testCase)
            testCase.verifyError(@() OceanStats.Data.Weighting.Idf.IIdfScheme(), ...
                'MATLAB:class:abstract');
        end

        function testWeightingClassesAcceptAnyIdfSchemeViaConstructor(testCase)
            % Integration: the Strategy is genuinely decoupled -- both
            % weighting classes accept ANY IIdfScheme, not just their own
            % default, without their source being touched to add a new one.
            import OceanStats.Data.Weighting.*
            import OceanStats.Data.Weighting.Idf.*
            counts = [2 1 0 0; 0 1 1 0; 1 0 0 3];

            tfidfClassic = TfidfWeighting('IdfScheme', IdfSchemeFactory.create('classic'));
            tfidfClassic.fit(counts);
            testCase.verifyEqual(tfidfClassic.IdfScheme.name(), "classic");
            testCase.verifyEqual(tfidfClassic.IDF, log10(3 ./ sum(counts > 0, 1)), 'AbsTol', 1e-12);

            bm25Smooth = BM25Weighting('IdfScheme', IdfSchemeFactory.create('smooth'));
            bm25Smooth.fit(counts);
            testCase.verifyEqual(bm25Smooth.IdfScheme.name(), "smooth");
            testCase.verifyEqual(bm25Smooth.IDF, log((1+3) ./ (1+sum(counts > 0, 1))) + 1, 'AbsTol', 1e-12);
        end

        function testDefaultsPreserveOriginalValidatedBehavior(testCase)
            % Regression guard: swapping in the Strategy pattern must not
            % change either class's DEFAULT numeric output -- these are the
            % exact hand-worked values validated against scikit-learn and a
            % Lucene-formula reference earlier in this project.
            import OceanStats.Data.Weighting.*
            counts = [2 1 0 0; 0 1 1 0; 1 0 0 3];

            tfidf = TfidfWeighting();
            tfidf.fit(counts);
            testCase.verifyEqual(tfidf.IdfScheme.name(), "smooth");

            bm25 = BM25Weighting();
            bm25.fit(counts);
            testCase.verifyEqual(bm25.IdfScheme.name(), "bm25");
            W = bm25.forward(counts);
            testCase.verifyEqual(full(W(1,1)), 0.646255, 'AbsTol', 1e-5);
            testCase.verifyEqual(full(W(3,4)), 1.438550, 'AbsTol', 1e-5);
        end
    end
end
