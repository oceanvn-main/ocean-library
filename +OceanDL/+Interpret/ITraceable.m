classdef (Abstract) ITraceable < handle
    % ITRACEABLE  A model whose intermediate activations can be recorded by
    %   name -- the entry point of OceanDL.Interpret (feature visualisation,
    %   Szeliski 2nd ed. Sec. 5.4 (pp. 309-311, Figs. 5.47-5.50)). Any model that implements it gets the
    %   whole toolkit (top-activating patches, projections to pixels,
    %   activation maximisation, occlusion) and the Model Lab "Inside" tab.
    %
    %   [acts, order] = model.trace(X, 'Parts', "all" | "encoder")
    %       acts   containers.Map: layer name -> activation (H x W x C x N) as a
    %              Tensor (OceanDL.Autograd.Tensor.raw for the array); it carries a
    %              graph when X is a Tensor requiring grad
    %       order  string array of the layer names in computation order
    %   T = model.traceNames(inputSize)   table Name, Size, Section
    %       Section: "encoder" | "latent" | "decoder" (models without a
    %       latent code use "encoder" for all layers).
    %
    %   NAMING: "<section>.<level/part>.<block>", e.g. "enc.L1.block1",
    %   "z1.mu", "dec.L2.td1", "dec.out". Activations are (H x W x C x N).

    methods (Abstract)
        [acts, order] = trace(obj, X, varargin)
    end

    methods
        function T = traceNames(obj, inputSize)
            % Names, sizes and sections, from one trace of a zero input.
            X = zeros([inputSize(:).' 1], 'like', OceanDL.Interpret.ITraceable.likeOf(obj));
            [acts, order] = OceanDL.Autograd.GradMode.noGrad(@() obj.trace(X));
            Name = order(:);
            Size = strings(numel(order), 1); Section = strings(numel(order), 1);
            for k = 1:numel(order)
                a = OceanDL.Autograd.Tensor.raw(acts(char(order(k))));
                s = size(a, 1:3);
                Size(k) = sprintf('%dx%dx%d', s);
                Section(k) = OceanDL.Interpret.ITraceable.sectionOf(order(k));
            end
            T = table(Name, Size, Section);
        end
    end

    methods (Static)
        function s = sectionOf(name)
            name = string(name);
            if startsWith(name, "z"), s = "latent";
            elseif startsWith(name, "dec."), s = "decoder";
            else, s = "encoder";
            end
        end

        function like = likeOf(model)
            like = 0;
            if isa(model, 'OceanDL.nn.Module')
                P = model.parameters();
                if ~isempty(P), like = P{1}.Data(1); end
            end
        end
    end
end
