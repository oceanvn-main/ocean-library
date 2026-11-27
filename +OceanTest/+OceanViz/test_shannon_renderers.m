classdef test_shannon_renderers < OceanTest.BaseTestCase
    % TEST_SHANNON_RENDERERS Unit tests validating CorpusSearchRenderer and EntropySpectrumRenderer.
    %
    % Compliant with plain Unicode/ASCII requirements (No LaTeX).
    
    properties
        FigureHandle
        AxSearch
        AxProb
        AxEntropy
    end
    
    methods (TestMethodSetup)
        function setupGraphics(testCase)
            % Create an invisible figure and UIAxes for headless test verification.
            testCase.FigureHandle = uifigure('Visible', 'off');
            testCase.AxSearch = uiaxes(testCase.FigureHandle);
            testCase.AxProb = uiaxes(testCase.FigureHandle);
            testCase.AxEntropy = uiaxes(testCase.FigureHandle);
        end
    end
    
    methods (TestMethodTeardown)
        function teardownGraphics(testCase)
            % Cleans up graphic containers.
            if isgraphics(testCase.FigureHandle)
                delete(testCase.FigureHandle);
            end
        end
    end
    
    methods (Test)
        function test_search_renderer_lifecycle(testCase)
            % 1. Instantiate CorpusSearchRenderer
            renderer = OceanViz.Strategies.CorpusSearchRenderer(testCase.AxSearch);
            testCase.verifyNotEmpty(renderer);
            testCase.verifyEqual(renderer.AxesHandle, testCase.AxSearch);
            
            % 2. Initialize grid
            rows = 10;
            cols = 25;
            renderer.initializeGrid(rows, cols);
            
            testCase.verifyEqual(renderer.Rows, rows);
            testCase.verifyEqual(renderer.Cols, cols);
            testCase.verifyEqual(size(renderer.TextHandles), [rows, cols]);
            
            % 3. Verify renderSearchStep with scan, match, and drawn markers
            displayText = repmat('A', 1, rows * cols);
            scanPos = 12;
            matchPos = 40;
            contextLen = 3;
            drawnCharPos = 43;
            
            % Should execute without crash
            renderer.renderSearchStep(displayText, scanPos, matchPos, contextLen, drawnCharPos);
            
            % Check that patches are visible
            testCase.verifyEqual(char(renderer.HighlightPatches.Cursor.Visible), 'on');
            testCase.verifyEqual(char(renderer.HighlightPatches.Match.Visible), 'on');
            testCase.verifyEqual(char(renderer.HighlightPatches.Drawn.Visible), 'on');
            
            % 4. Verify animateScan lifecycle with dummy corpus and searchLog
            corpus = 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG';
            searchLog = struct(...
                'startIndex', 5, ...
                'matchIndex', 15, ...
                'scannedIndices', 5:15, ...
                'scannedText', 'QUICK BROWN', ...
                'foundChar', 'N'...
            );
            
            % Test standard animation run with fast speedMs to keep unit tests fast
            renderer.animateScan(corpus, searchLog, 0.001);
            
            % 5. Test edge case: renderSearchStep with empty values
            renderer.renderSearchStep(displayText, [], [], [], []);
            testCase.verifyEqual(char(renderer.HighlightPatches.Cursor.Visible), 'off');
            testCase.verifyEqual(char(renderer.HighlightPatches.Match.Visible), 'off');
            testCase.verifyEqual(char(renderer.HighlightPatches.Drawn.Visible), 'off');
        end
        
        function test_spectrum_renderer_lifecycle(testCase)
            % 1. Instantiate EntropySpectrumRenderer
            renderer = OceanViz.Strategies.EntropySpectrumRenderer(testCase.AxProb, testCase.AxEntropy);
            testCase.verifyNotEmpty(renderer);
            testCase.verifyEqual(renderer.AxProb, testCase.AxProb);
            testCase.verifyEqual(renderer.AxEntropy, testCase.AxEntropy);
            
            % 2. Update probability spectrum
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ';
            probabilities = ones(1, 27) / 27; % Uniform
            context = 'TH';
            selectedChar = 'E';
            
            % Update without error
            renderer.updateProbabilitySpectrum(alphabet, probabilities, context, selectedChar);
            
            % Check that bars were created in AxProb
            children = testCase.AxProb.Children;
            testCase.verifyNotEmpty(children);
            
            % Verify title is correctly set without LaTeX
            testCase.verifyEqual(char(testCase.AxProb.Title.Interpreter), 'none');
            testCase.verifySubstring(char(testCase.AxProb.Title.String), 'TH');
            
            % 3. Update entropy decay curve
            entropies = [4.755, 4.012, 3.567, 2.890];
            activeOrder = 2;
            
            % Update without error
            renderer.updateEntropyDecayCurve(entropies, activeOrder);
            
            % Check that plots are updated in AxEntropy
            testCase.verifyNotEmpty(testCase.AxEntropy.Children);
            
            % Verify title and ticks are correct
            testCase.verifyEqual(char(testCase.AxEntropy.Title.Interpreter), 'none');
            testCase.verifyEqual(char(testCase.AxEntropy.TickLabelInterpreter), 'none');
        end
    end
end
