classdef test_shannon_sandbox < OceanTest.BaseTestCase
    % TEST_SHANNON_SANDBOX Integration tests for ShannonApproximationsSandbox app.
    %
    % This test suite verifies layout creation, timer cleanup, order switching,
    % step loops, and corpus file loading.
    %
    % See also: ShannonApproximationsSandbox

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
        function test_sandbox_lifecycle(testCase)
            % Verify that figure handles, layouts, and background playback timers initialize correctly
            app = ShannonApproximationsSandbox();
            
            testCase.verifyNotEmpty(app.FigureHandle);
            testCase.verifyTrue(isvalid(app.FigureHandle));
            
            % Assert core layouts are present
            testCase.verifyNotEmpty(app.LeftPanel);
            testCase.verifyNotEmpty(app.CenterPanel);
            testCase.verifyNotEmpty(app.RightPanel);
            
            % Assert visualization components exist
            testCase.verifyNotEmpty(app.BookAxes);
            testCase.verifyNotEmpty(app.AxProb);
            testCase.verifyNotEmpty(app.AxEntropy);
            
            % Confirm active state properties are initialized
            testCase.verifyEmpty(app.GeneratedText);
            testCase.verifyEmpty(app.ActiveContext);
            testCase.verifyFalse(app.IsPlaying);
            
            % Verify timer is created and configured
            testCase.verifyNotEmpty(app.PlaybackTimer);
            testCase.verifyTrue(isvalid(app.PlaybackTimer));
            
            % Perform absolute teardown to prevent resources leakage
            app.closeApp();
            testCase.verifyEmpty(app.FigureHandle);
            testCase.verifyEmpty(app.PlaybackTimer);
        end
        
        function test_order_transitions(testCase)
            % Verify switching Markov order updates properties, labels, and badges
            app = ShannonApproximationsSandbox();
            cleanup = onCleanup(@() app.closeApp());
            
            % Get buttons list
            btns = app.OrderButtonGroup.Children;
            
            % Transition to Zero-Order (UserData = 0)
            btn0 = [];
            for i = 1:length(btns)
                if btns(i).UserData == 0
                    btn0 = btns(i);
                    break;
                end
            end
            testCase.verifyNotEmpty(btn0);
            
            app.OrderButtonGroup.SelectedObject = btn0;
            app.changeOrderCallback();
            
            testCase.verifyEqual(app.getActiveOrder(), 0);
            testCase.verifyEmpty(app.ActiveContext);
            testCase.verifyEqual(app.EntropyDecayBadge.Text, 'MAXIMAL ENTROPY (CHAOS)');
            
            % Transition to Third-Order (UserData = 3)
            btn3 = [];
            for i = 1:length(btns)
                if btns(i).UserData == 3
                    btn3 = btns(i);
                    break;
                end
            end
            testCase.verifyNotEmpty(btn3);
            
            app.OrderButtonGroup.SelectedObject = btn3;
            app.changeOrderCallback();
            
            testCase.verifyEqual(app.getActiveOrder(), 3);
            testCase.verifyEqual(app.EntropyDecayBadge.Text, 'ENTROPY DECAY ACTIVE');
        end
        
        function test_playback_step_loop(testCase)
            % Verify that playing, pausing, and stepping the simulation executes without errors
            app = ShannonApproximationsSandbox();
            cleanup = onCleanup(@() app.closeApp());
            
            % Initial draw should be empty
            testCase.verifyEmpty(app.GeneratedText);
            
            % Execute single step
            app.runStep();
            testCase.verifyNotEmpty(app.GeneratedText);
            
            % Verify play toggle
            testCase.verifyFalse(app.IsPlaying);
            app.togglePlayback();
            testCase.verifyTrue(app.IsPlaying);
            testCase.verifyEqual(app.PlaybackTimer.Running, 'on');
            
            % Verify pause toggle
            app.togglePlayback();
            testCase.verifyFalse(app.IsPlaying);
            testCase.verifyEqual(app.PlaybackTimer.Running, 'off');
        end
        
        function test_corpus_file_load(testCase)
            % Verify loading a custom corpus resets generators and recalculates all entropies
            app = ShannonApproximationsSandbox();
            cleanup = onCleanup(@() app.closeApp());
            
            mockCorpus = 'AAAAABBBBB AAAAABBBBB';
            app.loadCorpus(mockCorpus);
            
            % Entropies must satisfy the law of systematic decay (H0 >= H1 >= H2 >= H3)
            H = app.Generator.Entropies;
            testCase.verifyEqual(H(1), log2(27), 'AbsTol', 1e-4);
            testCase.verifyTrue(H(1) >= H(2));
            testCase.verifyTrue(H(2) >= H(3));
            testCase.verifyTrue(H(3) >= H(4));
            
            % Verify UI status labels show the calculated values
            testCase.verifyEqual(app.LblH0.Text, sprintf('%.4f bits', H(1)));
            testCase.verifyEqual(app.LblH1.Text, sprintf('%.4f bits', H(2)));
            testCase.verifyEqual(app.LblH2.Text, sprintf('%.4f bits', H(3)));
            testCase.verifyEqual(app.LblH3.Text, sprintf('%.4f bits', H(4)));
        end
    end
end
