classdef VAEHandle < OceanViz.App.ModelLab.Models.IModelHandle
    % VAEHANDLE  A VAE (OceanDL.Generative.VAE) or hierarchical VAE
    %   (HierarchicalVAE) for the Model Lab, behind one latent interface:
    %     zs = encodeMeans(img)      posterior means, one cell per latent layer (top first)
    %     img = decode(zs)           decoder mean for the given latents
    %     imgs = sample(n, T, rs)    prior samples at temperature T
    %   Images in [0, 1] (Bernoulli / Gaussian-mean decoders).

    properties (SetAccess = private)
        Model
    end

    methods
        function obj = VAEHandle(model, opts)
            arguments
                model
                opts.ImageSize (1,3) double
                opts.Name (1,1) string = "VAE"
                opts.Meta struct = struct()
                opts.ClassNames (1,:) string = strings(1, 0)
            end
            obj.Model = model; obj.ImageSize = opts.ImageSize; obj.Name = opts.Name;
            obj.Meta = opts.Meta; obj.ClassNames = opts.ClassNames; obj.DataRange = [0 1];
            obj.Capabilities = ["vae", "encode", "decode", "sample"];
            if obj.numLayers() > 1, obj.Capabilities(end+1) = "hierarchical"; end   % a 1-level HierarchicalVAE is a VAE
            if isa(model, 'OceanDL.Interpret.ITraceable'), obj.Capabilities(end+1) = "traceable"; end   % "Inside the VAE" tab
        end

        function tf = isHierarchical(obj)
            tf = isa(obj.Model, 'OceanDL.Generative.HierarchicalVAE');
        end

        function n = numLayers(obj)
            if obj.isHierarchical(), n = obj.Model.numLatentLayers(); else, n = 1; end
        end

        function [zs, qs] = encodeMeans(obj, img)
            x = cast(obj.batch(img), 'like', obj.like());
            if obj.isHierarchical()
                qs = obj.Model.encode(x);
            else
                qs = {OceanDL.Autograd.GradMode.noGrad(@() obj.Model.encode(x))};
            end
            zs = cellfun(@(q) OceanDL.Autograd.Tensor.raw(q.Mu), qs, 'UniformOutput', false);
        end

        function img = decode(obj, zs)
            m = obj.Model;
            if obj.isHierarchical()
                out = OceanDL.Autograd.GradMode.noGrad(@() m.mean(m.decodeLatents(zs, obj.ImageSize(1:2))));
            else
                out = OceanDL.Autograd.GradMode.noGrad(@() m.mean(m.decode(zs{1})));
            end
            img = double(gather(OceanDL.Autograd.Tensor.raw(out)));
        end

        function img = reconstruct(obj, img)
            img = obj.decode(obj.encodeMeans(img));
        end

        function imgs = sample(obj, n, temperature, stream)
            m = obj.Model;
            if obj.isHierarchical()
                imgs = m.sample(n, obj.ImageSize(1:2), 'Temperature', temperature, 'Stream', stream);
            else
                zs = obj.encodeMeans(zeros(obj.ImageSize));
                z = temperature * cast(randn(stream, [size(zs{1}, 1:3) n]), 'like', obj.like());
                imgs = obj.decode({z});
            end
            imgs = double(gather(imgs));
        end
    end

    methods (Access = private)
        function x = batch(obj, img)
            x = reshape(img, obj.ImageSize(1), obj.ImageSize(2), obj.ImageSize(3), []);
        end

        function z = like(obj)
            P = obj.Model.parameters(); z = P{1}.Data([]);
        end
    end
end
