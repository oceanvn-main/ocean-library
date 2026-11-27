classdef AnalyzerAndFiltersTest < OceanTest.BaseTestCase
    % ANALYZERANDFILTERSTEST  Char filters, token filters, Unicode helpers and
    % the Analyzer composition. All expectations are written by hand here.
    % Non-ASCII characters are built from code points so the file stays ASCII.

    methods (Test)
        % ---------------- Unicode helpers ----------------
        function testCodepointRoundTripWithSurrogatePair(testCase)
            emoji = string(char([55357 56832]));            % U+1F600 as a UTF-16 surrogate pair
            cps = OceanNLP.Text.Unicode.codepoints("a" + emoji + "b");
            testCase.verifyEqual(cps, [97 128512 98]);
            testCase.verifyEqual(OceanNLP.Text.Unicode.fromCodepoints(cps), "a" + emoji + "b");
        end

        function testCategoriesComeFromJavaNotMatlabRegex(testCase)
            testCase.verifyTrue(all(OceanNLP.Text.Unicode.isLetter([97 246 1076])));      % a, o-diaeresis, Cyrillic de
            testCase.verifyFalse(any(OceanNLP.Text.Unicode.isLetter([48 33 32])));
            testCase.verifyTrue(OceanNLP.Text.Unicode.isMark(769));                        % combining acute
            testCase.verifyTrue(all(OceanNLP.Text.Unicode.isNumber([48 178 1633])));      % 0, superscript 2, Arabic-Indic 1
            testCase.verifyTrue(all(OceanNLP.Text.Unicode.isPunctuation([33 46 8217 8212])));   % ! . right quote, em-dash
            testCase.verifyFalse(any(OceanNLP.Text.Unicode.isPunctuation([97 48 32])));
            testCase.verifyTrue(all(OceanNLP.Text.Unicode.isSpace([32 9 10 160 12288])));
        end

        function testCategoryTableIsUnicode15NotTheJavaEightTables(testCase)
            U = OceanNLP.Text.Unicode;
            testCase.verifyEqual(U.tableVersion(), "15.0.0");
            % letters/numbers/punctuation added after Unicode 6.2 (the bundled JVM's version)
            testCase.verifyTrue(all(U.isLetter([hex2dec('1E900') hex2dec('1C90') hex2dec('30000')])));   % Adlam, Georgian Mtavruli, CJK ext G
            testCase.verifyTrue(U.isNumber(hex2dec('1D7CE')));                                            % mathematical bold digit zero
            testCase.verifyEqual(U.category([65 97 48 32 hex2dec('E000') hex2dec('378')]), [1 2 9 12 18 0]);   % Lu Ll Nd Zs Co Cn
            testCase.verifyEqual(U.category(-1), 0);                                                       % out of range -> unassigned
        end

        function testLowerTextUsesUnicodeTableNotMatlabLower(testCase)
            U = OceanNLP.Text.Unicode;
            adlamUpper = U.fromCodepoints(hex2dec('1E906'));
            testCase.verifyEqual(double(char(U.lowerText(adlamUpper))), [55354 56616]);   % U+1E928 as a surrogate pair
            testCase.verifyEqual(U.lowerText("Hello WORLD"), "hello world");
            testCase.verifyEqual(strlength(U.lowerText(string(char(304)))), 2);               % U+0130 -> i + combining dot (full mapping)
            % per-character, no final-sigma rule (as HF BertNormalizer)
            testCase.verifyEqual(double(char(U.lowerText(string(char([931]))))), 963);
        end

        function testWhitespaceIsWhiteSpacePropertyNotControlSeparators(testCase)
            testCase.verifyFalse(any(OceanNLP.Text.Unicode.isSpace([28 29 30 31])));      % not White_Space (Python re and Rust agree)
            testCase.verifyTrue(all(OceanNLP.Text.Unicode.isSpace([11 12 133 160 8232 12288])));
        end

        function testHtmlStripAstralAndInvalidNumericEntities(testCase)
            f = OceanNLP.Text.CharFilter.HtmlStripCharFilter();
            emoji = string(char([55357 56832]));
            testCase.verifyEqual(f.forward("&#128512;"), emoji);
            testCase.verifyEqual(f.forward("&#x1F600;"), emoji);
            testCase.verifyEqual(f.forward("&#x110000;"), string(char(65533)));           % out of range
            testCase.verifyEqual(f.forward("&#55296;"), string(char(65533)));             % surrogate
            testCase.verifyEqual(f.forward("&#0;"), string(char(65533)));
            testCase.verifyEqual(f.forward("&#99999999999;"), string(char(65533)));       % absurdly long
        end

        function testBertBasicCleanStepMatchesReferenceRules(testCase)
            b = OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('NeverSplit', strings(1, 0));
            testCase.verifyEqual(b.forward("a" + string(char(11)) + "b"), "ab");         % VT is a control char: deleted, not a space
            testCase.verifyEqual(b.forward("a" + string(char(160)) + "b"), ["a" "b"]);   % NBSP is whitespace
            testCase.verifyEqual(b.forward("a" + string(char(57344)) + "b"), "ab");      % private use is dropped
            testCase.verifyEqual(b.forward("a" + string(char(888)) + "b"), "a" + string(char(888)) + "b");   % unassigned kept (HF Rust)
        end

        function testNormalizeForms(testCase)
            e = string(char(233));
            decomposed = string(char([101 769]));
            testCase.verifyEqual(OceanNLP.Text.Unicode.normalize(e, "NFD"), decomposed);
            testCase.verifyEqual(OceanNLP.Text.Unicode.normalize(decomposed, "NFC"), e);
            testCase.verifyEqual(OceanNLP.Text.Unicode.normalize(string(char(65313)), "NFKC"), "A");   % full-width A
        end

        % ---------------- char filters ----------------
        function testAsciiFolding(testCase)
            f = OceanNLP.Text.CharFilter.AsciiFoldingCharFilter();
            in = "caf" + string(char(233)) + " na" + string(char(239)) + "ve " + string(char(223)) + " " + string(char(322)) + "odz";
            testCase.verifyEqual(f.forward(in), "cafe naive ss lodz");
            testCase.verifyEqual(f.forward("plain"), "plain");
        end

        function testHtmlStrip(testCase)
            f = OceanNLP.Text.CharFilter.HtmlStripCharFilter();
            testCase.verifyEqual(strtrim(f.forward("<p>Hello <b>big</b> world</p>")), "Hello  big  world");
            testCase.verifyEqual(f.forward("a&amp;b &lt;c&gt; &#65;&#x42;"), "a&b <c> AB");
            testCase.verifyEqual(strtrim(f.forward("x<script>var a=1;</script>y<style>p{}</style>z")), "x y z");
            testCase.verifyEqual(f.forward("&amp;lt;"), "&lt;");           % decoded once only
        end

        function testUnicodeNormalizeCharFilter(testCase)
            f = OceanNLP.Text.CharFilter.UnicodeNormalizeCharFilter();     % NFKC
            testCase.verifyEqual(f.forward(string(char([65313 65314]))), "AB");
            g = OceanNLP.Text.CharFilter.UnicodeNormalizeCharFilter('Form', "NFD");
            testCase.verifyEqual(strlength(g.forward(string(char(233)))), 2);
        end

        % ---------------- token filters ----------------
        function testStopWordFilterDefaultAndCustom(testCase)
            f = OceanNLP.Text.Filter.StopWordFilter();
            testCase.verifyEqual(f.forward(["the" "cat" "is" "on" "a" "mat"]), ["cat" "mat"]);
            testCase.verifyEqual(numel(f.Words), 33);
            g = OceanNLP.Text.Filter.StopWordFilter('Words', ["cat"]);
            testCase.verifyEqual(g.forward(["the" "cat"]), "the");
            testCase.verifyEqual(f.forward("The"), "The");                 % case-sensitive: lowercase first
        end

        function testLengthFilter(testCase)
            f = OceanNLP.Text.Filter.LengthFilter('Min', 2, 'Max', 4);
            testCase.verifyEqual(f.forward(["a" "ab" "abcd" "abcde"]), ["ab" "abcd"]);
            testCase.verifyError(@() OceanNLP.Text.Filter.LengthFilter('Min', 5, 'Max', 2), 'OceanNLP:Text:LengthFilter:BadRange');
        end

        function testLowercaseFilter(testCase)
            testCase.verifyEqual(OceanNLP.Text.Filter.LowercaseFilter().forward(["ABC" "dEf"]), ["abc" "def"]);
        end

        % ---------------- analyzer ----------------
        function testBasicAnalyzerEqualsItsTokenizer(testCase)
            % The basic version is unchanged: an empty-chain Analyzer is exactly its tokenizer.
            tok = OceanNLP.Tokenize.RegexTokenizer();
            a = OceanNLP.Text.Analyzer('Tokenizer', tok);
            txt = "The Quick, brown fox; 42 times!";
            testCase.verifyEqual(a.forward(txt), tok.forward(txt));
            testCase.verifyEqual(OceanNLP.Text.Analyzer.preset("basic").forward(txt), tok.forward(txt));
        end

        function testEnglishPresetHandWorked(testCase)
            a = OceanNLP.Text.Analyzer.preset("english");
            % regex \w+ lowercases -> the running dogs ponies were hopping quickly
            % stop "the" -> running dogs ponies were hopping quickly ; porter -> run dog poni were hop quickli
            testCase.verifyEqual(a.forward("The running dogs' ponies were hopping quickly!"), ...
                ["run" "dog" "poni" "were" "hop" "quickli"]);
        end

        function testChainOrderMatters(testCase)
            % "ins" is not a stop word, but its Porter stem "in" is: it survives stop-then-stem and is
            % removed by stem-then-stop.
            P = OceanNLP.Text.Stem.PorterStemmer();
            tok = OceanNLP.Tokenize.RegexTokenizer();
            a1 = OceanNLP.Text.Analyzer('Tokenizer', tok, 'TokenFilters', {OceanNLP.Text.Filter.StopWordFilter(), OceanNLP.Text.Filter.StemFilter('Stemmer', P)});
            a2 = OceanNLP.Text.Analyzer('Tokenizer', tok, 'TokenFilters', {OceanNLP.Text.Filter.StemFilter('Stemmer', P), OceanNLP.Text.Filter.StopWordFilter()});
            testCase.verifyEqual(a1.forward("ins cats"), ["in" "cat"]);
            testCase.verifyEqual(a2.forward("ins cats"), "cat");
        end

        function testCharFilterRunsBeforeTokenizer(testCase)
            a = OceanNLP.Text.Analyzer.fromSpec(struct('CharFilters', ["htmlstrip" "asciifolding"], 'Tokenizer', "regex"));
            in = "<b>Caf" + string(char(233)) + "s</b>";
            testCase.verifyEqual(a.forward(in), "cafes");
        end

        function testFromSpecIsDataDriven(testCase)
            a = OceanNLP.Text.Analyzer.fromSpec(struct('TokenFilters', ["lowercase" "length" "stem:porter"]));
            testCase.verifyEqual(a.forward("Running RUNS"), ["run" "run"]);
            testCase.verifyEqual(a.name(), "analyzer(regex>lowercase>length>stem:porter)");
            testCase.verifyError(@() OceanNLP.Text.Analyzer.preset("nope"), 'OceanNLP:Text:Analyzer:UnknownPreset');
            testCase.verifyError(@() OceanNLP.Text.Analyzer.fromSpec(struct('TokenFilters', "bogus")), 'OceanNLP:Text:StrategyResolver:Unknown');
        end

        function testAnalyzerRejectsWrongStageTypes(testCase)
            testCase.verifyError(@() OceanNLP.Text.Analyzer('TokenFilters', {OceanNLP.Text.CharFilter.NoneCharFilter()}), ...
                'OceanNLP:Text:Analyzer:BadStage');
            testCase.verifyError(@() OceanNLP.Text.Analyzer('CharFilters', {OceanNLP.Text.Filter.LowercaseFilter()}), ...
                'OceanNLP:Text:Analyzer:BadStage');
        end

        function testAnalyzerThreePillars(testCase)
            a = OceanNLP.Text.Analyzer.preset("english");
            m = a.getMetadata();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyGreaterThan(strlength(m.ComposedPartReference.TokenizerReference), 0);
            testCase.verifyEqual(numel(m.ComposedPartReference.StageReferences), 3);
            p = a.getPipelineSpec();
            testCase.verifyEqual(p.Kind, "VIRTUAL");
            testCase.verifyEqual(p.ComposesFrom, {'OceanNLP.Tokenize.RegexTokenizer', 'OceanNLP.Text.Filter.LowercaseFilter', ...
                'OceanNLP.Text.Filter.StopWordFilter', 'OceanNLP.Text.Filter.StemFilter'});
        end

        function testAnalyzerPlugsIntoBM25RankerAsTokenizer(testCase)
            corpus = ["cats are running", "the dog runs fast", "birds fly high"];
            plain = OceanNLP.Retrieval.BM25Ranker();  plain.fit(corpus);
            stem = OceanNLP.Retrieval.BM25Ranker('Tokenizer', OceanNLP.Text.Analyzer.preset("english"));  stem.fit(corpus);
            [~, sPlain] = plain.rank("run");
            [iStem, sStem] = stem.rank("run");
            testCase.verifyEqual(max(sPlain), 0);                 % no exact "run" token without stemming
            testCase.verifyGreaterThan(sStem(1), 0);              % stemming matches running + runs
            testCase.verifyEqual(sort(iStem(1:2)), [1 2]);
        end
    end
end
