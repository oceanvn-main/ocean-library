classdef test_card_deck_sandbox < OceanTest.BaseTestCase
    % TEST_CARD_DECK_SANDBOX Integration tests for CardDeckSandbox app.
    
    methods (TestClassSetup)
        function addAppPath(testCase)
            import matlab.unittest.fixtures.PathFixture
            % Determine relative folder to include the 'Information Theory' directory
            [currentFolder, ~, ~] = fileparts(mfilename('fullpath'));
            % CurrentFolder is +OceanTest/+OceanViz
            % LibraryRoot is currentFolder/../..
            libRoot = fullfile(currentFolder, '..', '..');
            appDir = fullfile(libRoot, '+OceanViz', '+App', 'Information Theory');
            
            % Apply path fixture to temporarily add directory to path
            testCase.applyFixture(PathFixture(appDir));
        end
    end
    
    methods (Test)
        function test_app_lifecycle(testCase)
            % 1. Open the app
            app = CardDeckSandbox();
            testCase.verifyTrue(isvalid(app.FigureHandle), 'App figure should be valid.');
            testCase.verifyEqual(char(app.FigureHandle.Visible), 'off', 'App should start invisible by default.');
            
            % 2. Show the app
            app.show();
            testCase.verifyEqual(char(app.FigureHandle.Visible), 'on', 'App should become visible.');
            
            % 3. Close the app
            app.closeApp();
            testCase.verifyFalse(isvalid(app.FigureHandle), 'App figure should be deleted after closing.');
            testCase.verifyEmpty(app.TimerPlayback, 'Timer should be cleaned up and empty.');
        end
        
        function test_simulation_controls(testCase)
            % 1. Initialize
            app = CardDeckSandbox();
            testCase.addTeardown(@() app.closeApp());
            
            % Configure simple parameters to make test fast
            app.NumRedRSpinner.Value = 5;
            app.NumBlackBSpinner.Value = 5;
            app.resetSimulation();
            
            testCase.verifyEqual(app.Simulator.NumRedR, 5);
            testCase.verifyEqual(app.Simulator.NumBlackB, 5);
            testCase.verifyEqual(app.CurrentWealth, 100.0);
            testCase.verifyEqual(length(app.WealthHistory), 1);
            
            % 2. Step
            app.stepPressed();
            testCase.verifyEqual(app.Simulator.CurrentIndex, 1, 'Index should increment by 1 after step.');
            testCase.verifyEqual(length(app.WealthHistory), 2, 'Wealth history should grow to 2.');
            testCase.verifyEqual(length(app.Renderer.CardPatches), 1, 'One card patch should be drawn.');
            
            % 3. Play & Pause
            app.togglePlay();
            testCase.verifyTrue(app.IsPlaying, 'App should be playing.');
            testCase.verifyNotEmpty(app.TimerPlayback, 'Timer should be instantiated.');
            
            app.togglePlay();
            testCase.verifyFalse(app.IsPlaying, 'App should be paused.');
            testCase.verifyEmpty(app.TimerPlayback, 'Timer should be cleaned up.');
            
            % 4. Simulate 100
            app.runSimulate100();
            testCase.verifyEqual(length(app.PathsLinesHandles), 100, 'Should overlay 100 trajectories.');
            
            % 5. Reset
            app.resetSimulation();
            testCase.verifyEqual(app.Simulator.CurrentIndex, 0, 'Index should reset to 0.');
            testCase.verifyEqual(app.CurrentWealth, 100.0, 'Wealth should reset to starting wealth.');
            testCase.verifyEqual(length(app.WealthHistory), 1, 'Wealth history should reset.');
            testCase.verifyEqual(length(app.Renderer.CardPatches), 0, 'Card patches should be cleared.');
        end
        
        function test_strategy_switching(testCase)
            % Initialize
            app = CardDeckSandbox();
            testCase.addTeardown(@() app.closeApp());
            
            % 1. Switch to Constant Fraction
            app.StrategyDropdown.Value = 'Constant Fraction (b = alpha)';
            app.strategyChanged();
            testCase.verifyEqual(char(app.BeliefDeviationSlider.Enable), 'on', 'Slider should be enabled.');
            testCase.verifyEqual(app.BeliefDeviationSlider.Value, 0.5, 'Default alpha should be 0.5.');
            
            % Update alpha slider
            app.BeliefDeviationSlider.Value = 0.4;
            app.sliderChanged();
            testCase.verifyEqual(app.BeliefDeviationLabel.Text, 'alpha = 0.40', 'Label should match alpha.');
            badgeLbl = findobj(app.BadgeEquivalence, 'Type', 'uilabel');
            testCase.verifyEqual(badgeLbl.Text, 'MISMATCH DETECTED', 'Badge should report mismatch.');
            
            % 2. Switch to Subjective Belief Mismatch
            app.StrategyDropdown.Value = 'Subjective Belief Mismatch';
            app.strategyChanged();
            testCase.verifyEqual(char(app.BeliefDeviationSlider.Enable), 'on', 'Slider should be enabled.');
            testCase.verifyEqual(app.BeliefDeviationSlider.Value, 0.2, 'Default dev should be 0.2.');
            
            % Update dev slider
            app.BeliefDeviationSlider.Value = 0.15;
            app.sliderChanged();
            testCase.verifyEqual(app.BeliefDeviationLabel.Text, '15% Deviation', 'Label should match dev.');
            
            % 3. Switch back to Kelly Proportional
            app.StrategyDropdown.Value = 'Kelly Proportional (b = p)';
            app.strategyChanged();
            testCase.verifyEqual(char(app.BeliefDeviationSlider.Enable), 'off', 'Slider should be disabled.');
            testCase.verifyEqual(app.BeliefDeviationSlider.Value, 0, 'Deviation should reset to 0.');
            testCase.verifyEqual(badgeLbl.Text, 'EQUIVALENCE VERIFIED', 'Badge should report equivalence.');
        end
    end
end
