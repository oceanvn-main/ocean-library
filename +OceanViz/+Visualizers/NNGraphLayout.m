classdef NNGraphLayout < OceanViz.Visualizers.IReplayLayout
    % NNGRAPHLAYOUT  Default dashboard layout: 3-panel left column +
    %   large NNGraph right panel. Standard for classification experiments.
    %
    % See also: OceanViz.Visualizers.IReplayLayout

    methods
        function [axCost, axAcc, axSample, axPred] = buildLayout(~, fig, ck, colors)
            axCost   = subplot(3, 3, 1,           'Parent', fig);
            axAcc    = subplot(3, 3, 4,           'Parent', fig);
            axSample = subplot(3, 3, 7,           'Parent', fig);
            axPred   = subplot(3, 3, [2 3 5 6 8 9], 'Parent', fig);

            set(axCost, 'Color', colors.Background, 'XColor', colors.Grid, 'YColor', colors.Grid);
            xlim(axCost, [1 ck.numEpochs]);
            ylim(axCost, [0 max(ck.costHistory) * 1.1 + 0.01]);
            grid(axCost, 'on');
            ylabel(axCost, 'Loss', 'Color', colors.Text);
            title(axCost, sprintf('Cross-Entropy  (%d samples)', ck.nSamples), ...
                'Color', colors.Text, 'FontSize', 9);

            set(axAcc, 'Color', colors.Background, 'XColor', colors.Grid, 'YColor', colors.Grid);
            xlim(axAcc, [1 ck.numEpochs]);
            ylim(axAcc, [0 1.05]);
            grid(axAcc, 'on');
            ylabel(axAcc, 'Accuracy', 'Color', colors.Text);
            title(axAcc, 'Training Accuracy', 'Color', colors.Text, 'FontSize', 9);

            axis(axPred, 'off');
        end
    end
end
