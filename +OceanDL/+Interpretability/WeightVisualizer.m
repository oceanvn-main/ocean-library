classdef WeightVisualizer
    % WEIGHTVISUALIZER  "What patterns have the filters learned?"
    %   Extracts a Conv2D layer's filters as displayable, per-filter
    %   min-max-normalized tiles. Pure forward-only: reads Weight, needs
    %   no forward pass through any image.
    %
    %   USAGE:
    %     conv = engine.getLayerObject('conv1');
    %     tiles = OceanDL.Interpretability.WeightVisualizer.extractTiles(conv);
    %     img   = OceanDL.Interpretability.WeightVisualizer.toDisplayImage(tiles{1});

    methods (Static)
        function tiles = extractTiles(convLayer)
            % EXTRACTTILES  One (KH x KW x Cin) raw-weight tile per output
            %   filter, as a 1 x Cout cell array.
            W = convLayer.Weight;
            Cout = size(W, 4);
            tiles = cell(1, Cout);
            for o = 1:Cout
                tiles{o} = W(:, :, :, o);
            end
        end

        function img = toDisplayImage(tile)
            % TODISPLAYIMAGE  (KH x KW x Cin) -> a [0,1]-normalized
            %   displayable image: RGB kept as-is (Cin==3), single-channel
            %   kept as-is (Cin==1), otherwise averaged across input
            %   channels (the common fallback for Cin not in {1,3}).
            Cin = size(tile, 3);
            if Cin == 1 || Cin == 3
                img = tile;
            else
                img = mean(tile, 3);
            end
            lo = min(img(:)); hi = max(img(:));
            if hi > lo
                img = (img - lo) / (hi - lo);
            else
                img = zeros(size(img));
            end
        end
    end
end
