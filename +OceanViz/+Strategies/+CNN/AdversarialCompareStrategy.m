classdef AdversarialCompareStrategy < OceanViz.Strategies.BaseStrategy
    % ADVERSARIALCOMPARESTRATEGY  Three-panel original / perturbation /
    %   adversarial-image comparison. "Which tiny input direction most
    %   changes the prediction?"
    %
    %   r = AdversarialPerturbation.fgsm(engine, X, trueClass, epsilon);
    %   strat.plot(ax, r, struct('Image', X));
    %
    % See also: OceanDL.Interpretability.AdversarialPerturbation

    methods
        function plot(obj, ax, data, options) %#ok<INUSL>
            if nargin < 4, options = struct(); end
            import OceanViz.Strategies.CNN.CNNVizUtils

            cla(ax, 'reset');
            imgs = {};
            labels = {};
            if isfield(options, 'Image') && ~isempty(options.Image)
                orig = reshape(options.Image, size(options.Image,1), size(options.Image,2), size(options.Image,3));
                imgs{end+1} = CNNVizUtils.normalize01(orig); %#ok<AGROW>
                labels{end+1} = sprintf('original (%.3f)', data.OriginalScore(1)); %#ok<AGROW>
            end
            pert = reshape(data.Perturbation, size(data.Perturbation,1), size(data.Perturbation,2), size(data.Perturbation,3));
            imgs{end+1} = CNNVizUtils.applyColormap(CNNVizUtils.normalize01(mean(pert, 3)), redblue256()); %#ok<AGROW>
            labels{end+1} = 'perturbation'; %#ok<AGROW>

            adv = reshape(data.AdversarialImage, size(data.AdversarialImage,1), size(data.AdversarialImage,2), size(data.AdversarialImage,3));
            imgs{end+1} = CNNVizUtils.normalize01(adv); %#ok<AGROW>
            labels{end+1} = sprintf('adversarial (%.3f)', data.AdversarialScore(1)); %#ok<AGROW>

            CNNVizUtils.drawTileGrid(ax, imgs, labels);
            ax.Color = CNNVizUtils.COLOR_BG;
            axis(ax, 'off');
            title(ax, 'Adversarial Perturbation', ...
                'Color', CNNVizUtils.COLOR_TEXT, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end

function cmap = redblue256()
    % Local helper: a simple diverging blue-white-red colormap, dependency-free.
    n = 128;
    top = [linspace(0,1,n)', linspace(0,1,n)', ones(n,1)];      % blue -> white
    bot = [ones(n,1), linspace(1,0,n)', linspace(1,0,n)'];      % white -> red
    cmap = [top; bot];
end
