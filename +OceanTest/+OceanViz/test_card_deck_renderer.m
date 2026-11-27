classdef test_card_deck_renderer < OceanTest.BaseTestCase
    % TEST_CARD_DECK_RENDERER Unit tests validating CardDeckRenderer.
    %
    % Compliant with plain Unicode/ASCII requirements (No LaTeX).

    properties
        FigureHandle
        AxesHandle
    end

    methods (TestMethodSetup)
        function setupGraphics(testCase)
            % Create an invisible figure and UIAxes for headless test verification.
            testCase.FigureHandle = uifigure('Visible', 'off');
            testCase.AxesHandle = uiaxes(testCase.FigureHandle);
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
        function test_renderer_initialization(testCase)
            % 1. Instantiate CardDeckRenderer
            renderer = OceanViz.Strategies.CardDeckRenderer(testCase.AxesHandle);
            testCase.verifyNotEmpty(renderer);
            testCase.verifyEqual(renderer.AxesHandle, testCase.AxesHandle);

            % 2. Test initialization with varying dimensions
            % 52 cards (standard deck)
            renderer.initializeLayout(26, 26);
            testCase.verifyEqual(renderer.R, 26);
            testCase.verifyEqual(renderer.B, 26);
            testCase.verifyEqual(length(renderer.CardPatches.Handles), 52);

            % 10 cards
            renderer.initializeLayout(5, 5);
            testCase.verifyEqual(renderer.R, 5);
            testCase.verifyEqual(renderer.B, 5);
            testCase.verifyEqual(length(renderer.CardPatches.Handles), 10);

            % 0 cards (empty deck)
            renderer.initializeLayout(0, 0);
            testCase.verifyEqual(renderer.R, 0);
            testCase.verifyEqual(renderer.B, 0);
            testCase.verifyEqual(length(renderer.CardPatches.Handles), 0);
        end

        function test_add_card_patches(testCase)
            renderer = OceanViz.Strategies.CardDeckRenderer(testCase.AxesHandle);
            renderer.initializeLayout(5, 5);

            % Add a Red card at index 1
            renderer.addCard('Red', 1, 0.5);
            testCase.verifyEqual(renderer.ProbValues(1), 0.5);
            testCase.verifyEqual(renderer.CardPatches.Types(1), 1);
            testCase.verifyTrue(isgraphics(renderer.CardPatches.Handles(1)));

            % Add a Black card at index 2
            renderer.addCard('Black', 2, 0.4);
            testCase.verifyEqual(renderer.ProbValues(2), 0.4);
            testCase.verifyEqual(renderer.CardPatches.Types(2), 2);
            testCase.verifyTrue(isgraphics(renderer.CardPatches.Handles(2)));

            % Test undrawn count update
            renderer.updateUndrawnCount(8);
            testCase.verifyEqual(length(renderer.DeckPilePatches), min(3, 8));

            % Test interactive hover check (without throw)
            % Hover inside card 1 bounds
            pos1 = renderer.CardPatches.Positions(1, :);
            x_mid = pos1(1) + pos1(3)/2;
            y_mid = pos1(2) + pos1(4)/2;
            renderer.checkHover(x_mid, y_mid, 2.0, 2.0);
            testCase.verifyEqual(char(renderer.TooltipRect.Visible), 'on');
            testCase.verifyEqual(char(renderer.TooltipText.Visible), 'on');

            % Hover outside all cards
            renderer.checkHover(0, 0, 2.0, 2.0);
            testCase.verifyEqual(char(renderer.TooltipRect.Visible), 'off');
            testCase.verifyEqual(char(renderer.TooltipText.Visible), 'off');
        end

        function test_clear(testCase)
            renderer = OceanViz.Strategies.CardDeckRenderer(testCase.AxesHandle);
            renderer.initializeLayout(5, 5);

            % Add some cards
            renderer.addCard(1, 1, 0.5);
            renderer.addCard(2, 2, 0.4);

            % Verify elements are drawn
            testCase.verifyTrue(isgraphics(renderer.CardPatches.Handles(1)));
            testCase.verifyTrue(isgraphics(renderer.CardPatches.Handles(2)));

            % Clear layout
            renderer.clear();

            % Verify graphic handles are deleted or empty
            testCase.verifyEqual(length(renderer.CardPatches.Handles), 0);
            testCase.verifyEqual(length(renderer.ProbValues), 0);
            testCase.verifyEqual(char(renderer.TooltipRect.Visible), 'off');
            testCase.verifyEqual(char(renderer.TooltipText.Visible), 'off');
        end
    end
end
