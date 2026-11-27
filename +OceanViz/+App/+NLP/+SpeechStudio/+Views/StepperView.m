classdef StepperView < handle
    % STEPPERVIEW  Fixed stage stepper ("review completed steps" wizard variant): one
    %   button per stage, enabled only after a completed, non-stale run and only for stages
    %   the model supports; the current stage is bold and marked.

    properties (SetAccess = private)
        Grid
        Buttons = matlab.ui.control.Button.empty
    end

    properties (Access = private)
        ViewModel
        Listeners = event.listener.empty
    end


    methods
        function obj = StepperView(parent, viewModel)
            arguments
                parent
                viewModel (1,1) OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel
            end
            obj.ViewModel = viewModel;
            n = numel(viewModel.StageTitles);
            obj.Grid = uigridlayout(parent, [1 n], 'ColumnWidth', repmat({'1x'}, 1, n), ...
                'Padding', [8 10 8 10], 'ColumnSpacing', 4);
            for k = 1:n
                num = string(k) + ".";
                obj.Buttons(k) = uibutton(obj.Grid, 'Text', char(num + " " + viewModel.StageTitles(k)), ...
                    'WordWrap', 'on', 'ButtonPushedFcn', @(~, ~) obj.ViewModel.selectStage(k));
            end
            for p = ["StageEnabled", "CurrentStage", "IsStale", "HasResult"]
                obj.Listeners(end+1) = addlistener(viewModel, p, 'PostSet', @(~, ~) obj.sync());
            end
            obj.sync();
        end

        function delete(obj)
            delete(obj.Listeners);
        end

        function sync(obj)
            vm = obj.ViewModel;
            for k = 1:numel(obj.Buttons)
                b = obj.Buttons(k);
                b.Enable = vm.StageEnabled(k);
                cur = k == vm.CurrentStage && vm.HasResult;
                b.FontWeight = obj.iff(cur, 'bold', 'normal');
                tip = "";
                if ~vm.HasResult, tip = "Press Run to compute the pipeline";
                elseif vm.IsStale, tip = "Settings changed: press Run";
                elseif ~vm.StageEnabled(k) && vm.NoSpeech, tip = "No speech detected: nothing was recognised";
                elseif ~vm.StageEnabled(k), tip = "Not applicable to this model";
                end
                b.Tooltip = char(tip);
                num = string(k) + ".";
                txt = num + " " + vm.StageTitles(k);
                if cur, txt = "▶ " + txt; end
                b.Text = char(txt);
            end
        end
    end

    methods (Static, Access = private)
        function v = iff(c, a, b)
            if c, v = a; else, v = b; end
        end
    end
end
