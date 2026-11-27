classdef HvaeDecoderAdapter < OceanDL.nn.Module
    % HVAEDECODERADAPTER  Test helper: the generative path of a ONE-latent
    %   HierarchicalVAE as an ordinary VAE decoder (z -> likelihood parameters).

    properties (SetAccess = private)
        Model
        ImageSize (1,2) double
    end

    methods
        function obj = HvaeDecoderAdapter(model, imageSize)
            obj.Model = model;
            obj.ImageSize = imageSize;
        end

        function out = forward(obj, z)
            out = obj.Model.decodeLatents({z}, obj.ImageSize);
        end
    end
end
