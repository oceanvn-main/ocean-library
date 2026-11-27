classdef HvaeEncoderAdapter < OceanDL.nn.Module
    % HVAEENCODERADAPTER  Test helper: exposes the posterior of a ONE-latent
    %   HierarchicalVAE as an ordinary VAE encoder (x -> [Mu; LogVar] channels),
    %   so OceanDL.Generative.VAE can be run on exactly the same network.

    properties (SetAccess = private)
        Model
    end

    methods
        function obj = HvaeEncoderAdapter(model)
            obj.Model = model;
        end

        function h = forward(obj, X)
            m = obj.Model;
            [~, ~, ~, qs] = m.topDown(m.bottomUp(X), size(X, 4), size(X, 1:2), "mode");
            h = cat(3, qs{1}.Mu, qs{1}.LogVar);
        end
    end
end
