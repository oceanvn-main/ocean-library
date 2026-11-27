classdef VAELayout < OceanViz.App.Replay.Layouts.IReplayStudioLayout
    % VAELAYOUT  A (hierarchical) VAE recording (OceanDL.Generative.Inspect.VAERecordable):
    %   per image x -- the reconstruction, the decoder mean at a sampled z, and for
    %   every latent layer the posterior mean, standard deviation and KL per unit;
    %   the numbers of the ELBO decomposition (Kingma & Welling 2019, Ch. 2) read
    %   from the file for the current image.

    properties (Constant)
        Needed = ["x" "xhat" "xhatSample" "mu_1" "sigma_1" "klUnit_1" "recon" "kl" "negElbo"]
    end

    methods
        function n = name(~), n = "VAE"; end

        function n = timeline(~, ~), n = "negElbo"; end

        function tf = appliesTo(obj, rec)
            tf = all(ismember(obj.Needed, rec.names()));
        end

        function tiles = tiles(~, rec)
            T = @OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile;
            tiles = [T("Input image", "x"), T("Reconstruction (decoder mean at the posterior means)", "xhat"), ...
                     T("Decoder mean at a sampled z", "xhatSample")];
            L = OceanViz.App.Replay.Layouts.VAELayout.numLatent(rec);
            for i = 1:L
                tiles = [tiles, T(sprintf("Posterior mean, latent layer %d", i), sprintf("mu_%d", i)), ...
                    T(sprintf("Posterior std, latent layer %d", i), sprintf("sigma_%d", i)), ...
                    T(sprintf("KL per unit, latent layer %d", i), sprintf("klUnit_%d", i))]; %#ok<AGROW>
            end
        end

        function lines = notes(~, rec, coords)
            c = struct('image', coords.image);
            g = @(n) double(rec.get(n, c));
            [~, at] = rec.get("negElbo", c);
            L = OceanViz.App.Replay.Layouts.VAELayout.numLatent(rec);
            lines = {sprintf('VARIATIONAL AUTOENCODER -- Kingma & Welling (2019) Ch. 2   image n = %g', at.image), '', ...
                '-ELBO = E_q[ -log p(x|z) ] + KL( q(z|x) || p(z) )   (one-sample estimate)', ...
                'z = mu + sigma .* eps,  eps ~ N(0, I)   (reparameterisation)', ''};
            if any(rec.names() == "label")
                lines{end+1} = sprintf('label y = %g', g("label"));
            end
            lines{end+1} = sprintf('-log p(x|z)   = %10.3f nats', g("recon"));
            for i = 1:L
                if any(rec.names() == sprintf("kl_%d", i))
                    lines{end+1} = sprintf('KL layer %d    = %10.3f nats', i, g(sprintf("kl_%d", i))); %#ok<AGROW>
                end
            end
            lines{end+1} = sprintf('KL total      = %10.3f nats', g("kl"));
            lines{end+1} = sprintf('-ELBO         = %10.3f nats', g("negElbo"));
            if any(rec.names() == "bitsPerDim")
                lines{end+1} = sprintf('bits / dim    = %10.4f', g("bitsPerDim"));
            end
            if L > 1
                lines = [lines, {'', 'Layer 1 = top. For i > 1: q(z_i | x, z_<i) and p(z_i | z_<i)', ...
                    'depend on the sampled upper latents (NVAE-style; beyond the Ch. 2 VAE).'}];
            end
            lines = [lines, {'', 'KL per unit ~ 0: that unit is inactive (posterior = prior).'}];
        end
    end

    methods (Static)
        function L = numLatent(rec)
            n = rec.names();
            L = 0;
            while any(n == sprintf("mu_%d", L + 1)), L = L + 1; end
        end
    end
end
