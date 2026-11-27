classdef ActivationMapExtractor
    % ACTIVATIONMAPEXTRACTOR  "Where does a feature respond?"
    %   Forward pass + read a chosen layer's cached (H x W x C x N)
    %   feature maps. Pure forward-only.
    %
    %   USAGE:
    %     maps = ActivationMapExtractor.extract(engine, X, 'conv1');   % (H,W,C,N)
    %     img  = ActivationMapExtractor.channelToImage(maps, 3, 1);    % [0,1] image

    methods (Static)
        function maps = extract(engine, X, layerName)
            engine.forward(X);
            maps = engine.getActivations(layerName);
        end

        function img = channelToImage(maps, channelIdx, sampleIdx)
            arguments
                maps (:,:,:,:) double
                channelIdx (1,1) double {mustBePositive, mustBeInteger} = 1
                sampleIdx (1,1) double {mustBePositive, mustBeInteger} = 1
            end
            img = maps(:, :, channelIdx, sampleIdx);
            lo = min(img(:)); hi = max(img(:));
            if hi > lo
                img = (img - lo) / (hi - lo);
            else
                img = zeros(size(img));
            end
        end
    end
end
