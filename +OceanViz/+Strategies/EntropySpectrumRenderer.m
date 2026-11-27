classdef EntropySpectrumRenderer < handle
    % ENTROPYSPECTRUMRENDERER Renders conditional probabilities and entropy decay curves.
    %
    % This class manages the plotting of the character conditional probabilities
    % (sorted descending) and the entropy decay curve representing Shannon's
    % structural decay from maximal entropy (chaos) down to lower entropy.
    %
    % See also: OceanMath.Information.Processes.ShannonTextGenerator

    properties
        AxProb (1, 1) matlab.ui.control.UIAxes
        AxEntropy (1, 1) matlab.ui.control.UIAxes
    end

    methods
        function obj = EntropySpectrumRenderer(axProb, axEntropy)
            % ENTROPYSPECTRUMRENDERER Constructor.
            arguments
                axProb matlab.ui.control.UIAxes
                axEntropy matlab.ui.control.UIAxes
            end
            obj.AxProb = axProb;
            obj.AxEntropy = axEntropy;

            % Configure dark mode styling for both axes
            obj.configureAxes(obj.AxProb);
            obj.configureAxes(obj.AxEntropy);
        end

        function configureAxes(obj, ax)
            % CONFIGUREAXES Applies modern dark-mode aesthetic styling to axes.
            arguments
                obj
                ax matlab.ui.control.UIAxes
            end
            ax.BackgroundColor = [0.17, 0.19, 0.23];
            ax.XColor = [0.85, 0.88, 0.93];
            ax.YColor = [0.85, 0.88, 0.93];
            ax.GridColor = [0.30, 0.35, 0.40];
            ax.MinorGridColor = [0.30, 0.35, 0.40];
            ax.XGrid = 'on';
            ax.YGrid = 'on';
            ax.FontName = 'Helvetica';
            ax.FontSize = 9;
            ax.Title.Interpreter = 'none';
            ax.XLabel.Interpreter = 'none';
            ax.YLabel.Interpreter = 'none';
            ax.TickLabelInterpreter = 'none';
        end

        function updateProbabilitySpectrum(obj, alphabet, probabilities, context, selectedChar)
            % UPDATEPROBABILITYSPECTRUM Plots the conditional probability distribution.
            arguments
                obj
                alphabet (1, 27) char
                probabilities (1, 27) double
                context char
                selectedChar char
            end

            % Sort probabilities in descending order
            [sortedProbs, sortIdx] = sort(probabilities, 'descend');
            sortedAlphabet = alphabet(sortIdx);

            % Clear probability axes
            cla(obj.AxProb);

            % Create base bar chart
            hold(obj.AxProb, 'on');
            bar(obj.AxProb, 1:27, sortedProbs, 'FaceColor', [0.30, 0.45, 0.65], 'EdgeColor', 'none');

            % Highlight the selected character bar in glowing neon cyan
            if ~isempty(selectedChar)
                selIdx = find(sortedAlphabet == selectedChar, 1, 'first');
                if ~isempty(selIdx)
                    bar(obj.AxProb, selIdx, sortedProbs(selIdx), 'FaceColor', [0.30, 0.65, 0.95], 'EdgeColor', 'none');
                end
            end
            hold(obj.AxProb, 'off');

            % Configure labels & limits
            obj.AxProb.XTick = 1:27;
            
            % Generate ticks where space is represented as '_'
            tickLabels = cell(1, 27);
            for idx = 1:27
                c = sortedAlphabet(idx);
                if c == ' '
                    tickLabels{idx} = '_';
                else
                    tickLabels{idx} = c;
                end
            end
            obj.AxProb.XTickLabel = tickLabels;
            obj.AxProb.XLim = [0.5, 27.5];
            
            % Ensure robust Y limits
            maxVal = max(sortedProbs);
            if maxVal <= 0
                maxVal = 1.0;
            end
            obj.AxProb.YLim = [0, maxVal * 1.15];

            % Title displaying the conditioning context
            if isempty(context)
                ctxStr = "empty";
            else
                ctxStr = "'" + string(context) + "'";
            end
            title(obj.AxProb, "Conditional Probability p(x | context) = " + ctxStr, ...
                'Color', [0.85, 0.88, 0.93], 'FontSize', 10);
            xlabel(obj.AxProb, "Alphabet Characters (Sorted Descending)", 'Color', [0.85, 0.88, 0.93]);
            ylabel(obj.AxProb, "Probability", 'Color', [0.85, 0.88, 0.93]);
        end

        function updateEntropyDecayCurve(obj, entropies, activeOrder)
            % UPDATEENTROPYDECAYCURVE Plots structural decay of Shannon conditional entropy.
            arguments
                obj
                entropies (1, 4) double
                activeOrder (1, 1) double {mustBeMember(activeOrder, [0, 1, 2, 3])}
            end

            cla(obj.AxEntropy);

            orders = 0:3;
            hold(obj.AxEntropy, 'on');

            % Plot decay curve path
            plot(obj.AxEntropy, orders, entropies, '-o', ...
                'Color', [0.30, 0.65, 0.95], ...
                'LineWidth', 2.0, ...
                'MarkerFaceColor', [0.30, 0.65, 0.95], ...
                'MarkerSize', 5);

            % Plot vertical indicator at the active Markov order
            plot(obj.AxEntropy, [activeOrder, activeOrder], [0, entropies(activeOrder + 1)], ...
                '--', 'Color', [0.99, 0.40, 0.10], 'LineWidth', 1.5);
            plot(obj.AxEntropy, activeOrder, entropies(activeOrder + 1), ...
                'o', 'Color', [0.99, 0.40, 0.10], ...
                'MarkerFaceColor', [0.99, 0.40, 0.10], ...
                'MarkerSize', 8);

            hold(obj.AxEntropy, 'off');

            % Ticks and Labels
            obj.AxEntropy.XTick = 0:3;
            obj.AxEntropy.XTickLabel = {'H0 (Uniform)', 'H1 (Unigram)', 'H2 (Bigram)', 'H3 (Trigram)'};
            obj.AxEntropy.XLim = [-0.3, 3.3];
            
            % Robust y limits
            maxVal = max(entropies);
            if maxVal <= 0
                maxVal = 5.0;
            end
            obj.AxEntropy.YLim = [0, maxVal * 1.15];

            title(obj.AxEntropy, "Information Entropy Decay H(N)", ...
                'Color', [0.85, 0.88, 0.93], 'FontSize', 10);
            xlabel(obj.AxEntropy, "Markov Order N", 'Color', [0.85, 0.88, 0.93]);
            ylabel(obj.AxEntropy, "Entropy (bits/symbol)", 'Color', [0.85, 0.88, 0.93]);
        end
    end
end
