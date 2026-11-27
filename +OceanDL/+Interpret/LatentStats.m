classdef LatentStats
    % LATENTSTATS  What the latent code of a (hierarchical) VAE carries, over a
    %   dataset (Kingma & Welling, "An Introduction to VAEs", Ch. 2; Ch. 4 for
    %   several latent layers).
    %
    %   KL PER UNIT: the ELBO's KL term, KL(q_phi(z|x) || p(z)) (Sec. 2.2), split
    %   over every latent number z_j and averaged over the data. Units with
    %   ~0 KL carry no information about x: the "inactive" units of Sec. 2.8
    %   (posterior equals prior). For hierarchical models the KL of layer i is
    %   KL(q(z_i | x, z_<i) || p(z_i | z_<i)), evaluated with the upper layers
    %   at their posterior means (as in reconstruction).
    %
    %   kl = LatentStats.klPerUnit(model, X)    cell, top layer first: (h x w x c) nats
    %   mu = LatentStats.meanCodes(model, X)    cell: average posterior mean mu_phi(x)
    %   n  = LatentStats.numLayers(model)       latent layers ("z<i>.mu" in the trace)
    %   Z  = LatentStats.codes(model, X, layers)  (N x D) posterior means mu_phi(x), one row per
    %                                            image, the chosen latent layers concatenated

    methods (Static)
        function kl = klPerUnit(model, X, opts)
            arguments
                model
                X
                opts.Batch (1,1) double = 256
            end
            kl = OceanDL.Interpret.LatentStats.average(model, X, "kl", opts.Batch);
        end

        function mu = meanCodes(model, X, opts)
            arguments
                model
                X
                opts.Batch (1,1) double = 256
            end
            mu = OceanDL.Interpret.LatentStats.average(model, X, "mu", opts.Batch);
        end

        function Z = codes(model, X, layers, opts)
            arguments
                model
                X
                layers double = []                % [] = all latent layers
                opts.Batch (1,1) double = 256
            end
            N = size(X, 4); Z = [];
            for b = 1:opts.Batch:N
                k = b:min(N, b + opts.Batch - 1);
                acts = OceanDL.Autograd.GradMode.noGrad(@() model.trace(X(:, :, :, k)));
                i = 1; rows = [];
                while isKey(acts, sprintf('z%d.mu', i))
                    if isempty(layers) || any(layers == i)
                        m = double(OceanDL.Autograd.Tensor.raw(acts(sprintf('z%d.mu', i))));
                        rows = [rows, reshape(m, [], numel(k)).']; %#ok<AGROW>
                    end
                    i = i + 1;
                end
                Z = [Z; rows]; %#ok<AGROW>
            end
        end

        function n = numLayers(model, inputSize)
            T = model.traceNames(inputSize);
            n = nnz(endsWith(T.Name, ".mu") & startsWith(T.Name, "z"));
        end
    end

    methods (Static, Access = private)
        function out = average(model, X, what, batch)
            N = size(X, 4); out = {};
            for b = 1:batch:N
                k = b:min(N, b + batch - 1);
                acts = OceanDL.Autograd.GradMode.noGrad(@() model.trace(X(:, :, :, k)));
                i = 1;
                while isKey(acts, sprintf('z%d.%s', i, what))
                    s = sum(double(OceanDL.Autograd.Tensor.raw(acts(sprintf('z%d.%s', i, what)))), 4);
                    if numel(out) < i, out{i} = s; else, out{i} = out{i} + s; end %#ok<AGROW>
                    i = i + 1;
                end
            end
            out = cellfun(@(s) s / N, out, 'UniformOutput', false);
        end
    end
end
