classdef DecoderAtoms
    % DECODERATOMS  How the decoder p_theta(x|z) BUILDS an image from the code.
    %
    %   ATOMS (not in the texts -- a standard way to read a generative decoder):
    %   what ONE latent unit z_j draws: the change of the decoded mean image when
    %   only z_j moves, at a reference code z0 (the data-average posterior mean):
    %       atom_j = d E[x | z] / d z_j  at z0          (a column of the decoder's Jacobian)
    %   estimated by central differences (decode(z0 + d e_j) - decode(z0 - d e_j)) / 2d,
    %   all positions of one latent channel in a single batch. Other latent units
    %   (and layers) are held fixed at z0.
    %
    %   LAYER CONTRIBUTIONS (hierarchical VAE, Kingma & Welling Ch. 4): decode
    %   the posterior means of the top layers only, the remaining layers at their
    %   prior mean p(z_i | z_<i) -- what each additional layer adds to the image.
    %
    %   [A, info] = DecoderAtoms.atoms(model, z0, layer, channel, imageSize)
    %       A (H x W x Cout x P): one atom per position of latent channel `channel`
    %       of latent layer `layer` (1 = top); info.Rows, info.Cols
    %   [imgs, labels] = DecoderAtoms.contributions(model, x)
    %       imgs (H x W x Cout x L+1): x, then "top 1 layer", ..., "all L layers"

    methods (Static)
        function [A, info] = atoms(model, z0, layer, channel, imageSize, opts)
            arguments
                model
                z0 cell
                layer (1,1) double
                channel (1,1) double
                imageSize (1,2) double
                opts.Delta (1,1) double = 0.1
            end
            [h, w, ~] = size(z0{layer});
            P = h * w;
            zs = cellfun(@(z) repmat(z, 1, 1, 1, 2 * P), z0, 'UniformOutput', false);
            [rows, cols] = ndgrid(1:h, 1:w);
            for p = 1:P
                zs{layer}(rows(p), cols(p), channel, 2 * p - 1) = zs{layer}(rows(p), cols(p), channel, 2 * p - 1) + opts.Delta;
                zs{layer}(rows(p), cols(p), channel, 2 * p)     = zs{layer}(rows(p), cols(p), channel, 2 * p)     - opts.Delta;
            end
            like = OceanDL.Interpret.ITraceable.likeOf(model);
            zs = cellfun(@(z) cast(z, 'like', like), zs, 'UniformOutput', false);
            m = OceanDL.Interpret.DecoderAtoms.decodeMean(model, zs, imageSize);
            A = (m(:, :, :, 1:2:end) - m(:, :, :, 2:2:end)) / (2 * opts.Delta);
            info.Rows = rows(:); info.Cols = cols(:);
        end

        function [imgs, labels] = contributions(model, x)
            [H, W, C] = size(x, 1:3);
            qs = OceanDL.Autograd.GradMode.noGrad(@() model.encode(reshape(x, H, W, C, 1)));
            mu = cellfun(@(q) OceanDL.Autograd.Tensor.raw(q.Mu), qs, 'UniformOutput', false);
            L = numel(mu);
            imgs = zeros(H, W, C, L + 1);
            imgs(:, :, :, 1) = x;
            labels = strings(1, L + 1); labels(1) = "x";
            for k = 1:L
                zs = mu; zs(k + 1:end) = {[]};                                   % lower layers: prior mean
                imgs(:, :, :, k + 1) = OceanDL.Interpret.DecoderAtoms.decodeMean(model, zs, [H W]);
                if k < L, labels(k + 1) = sprintf("top %d layer(s)", k); else, labels(k + 1) = sprintf("all %d layers", L); end
            end
        end

        function m = decodeMean(model, zs, imageSize)
            out = OceanDL.Autograd.GradMode.noGrad(@() model.mean(model.decodeLatents(zs, imageSize)));
            m = double(OceanDL.Autograd.Tensor.raw(out));
        end
    end
end
