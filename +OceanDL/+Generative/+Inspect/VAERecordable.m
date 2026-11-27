classdef VAERecordable < OceanDL.Interpret.TraceableRecordable
    % VAERECORDABLE  Replay adapter for (hierarchical) VAEs: for each input image,
    %   the quantities of Kingma & Welling (2019) Ch. 2 -- the approximate posterior
    %   q(z|x) = N(mu, diag(sigma^2)), a reparameterised sample z = mu + sigma .* eps,
    %   the KL term per latent unit, the decoder mean p(x|z) and the ELBO -- plus the
    %   generic layer activations of TraceableRecordable.
    %
    %   Model: OceanDL.Generative.HierarchicalVAE (a plain VAE is its 1-level case).
    %   Latent layer i (top first) is an h x w x c map. For i > 1 the posterior is
    %   q(z_i | x, z_<i) and the prior p(z_i | z_<i): they depend on the SAMPLED upper
    %   latents (NVAE-style hierarchy; Vahdat & Kautz 2020 -- beyond the Ch. 2 VAE).
    %
    %   Channels (per image), in addition to x, label, act_*:
    %     mu_i, sigma_i, z_i, klUnit_i   posterior mean / std, sample, KL per unit   image (h x w x c)
    %     xhat                           decoder mean at the posterior MEANS (the reconstruction)   image
    %     xhatSample                     decoder mean at the sampled z                               image
    %     recon                          -log p(x|z) at the sampled z (nats)                         scalar
    %     kl_i, kl                       KL of layer i, total KL (nats)                              scalar
    %     negElbo                        recon + kl = -ELBO (one-sample estimate, nats)              scalar
    %     bitsPerDim                     negElbo / (D log 2)                                         scalar
    %   Randomness: image k draws eps from its own substream k of a stream seeded with
    %   'Seed', so every number is reproducible and independent of which channels are
    %   recorded.
    %
    %   Example:
    %     [model, card] = OceanDL.Training.ModelCard.load("usps_hvae.mat");
    %     r = OceanDL.Generative.Inspect.VAERecordable(model, X, 'Ids', idx, 'Labels', y, 'Seed', 1);
    %     OceanReplay.Record.Recorder.run(r, spec, "hvae.h5");

    properties (SetAccess = private)
        Seed (1,1) double
        NumLatent (1,1) double
        LatentShape cell                     % {i} = [h w c]
        MuRange (1,:) double
        SigmaRange (1,:) double
        KLRange (1,:) double
    end

    methods
        function obj = VAERecordable(model, X, opts)
            arguments
                model (1,1) OceanDL.Generative.HierarchicalVAE
                X {mustBeNumeric}
                opts.Seed (1,1) double = 1
                opts.Ids (1,:) double = []
                opts.Labels = []
                opts.ClassNames (1,:) string = strings(1, 0)
                opts.Layers = []
                opts.DataRange (1,2) double = [0 1]
                opts.ModelName (1,1) string = ""
                opts.DatasetName (1,1) string = ""
            end
            c = namedargs2cell(rmfield(opts, 'Seed'));
            obj@OceanDL.Interpret.TraceableRecordable(model, X, c{:});
            obj.Seed = opts.Seed;
            % shapes and display ranges from the posterior-mean pass over all images
            [qs] = model.encode(X);
            obj.NumLatent = numel(qs);
            obj.LatentShape = cell(1, obj.NumLatent);
            mu = []; sg = [];
            for i = 1:obj.NumLatent
                m = double(gather(OceanDL.Autograd.Tensor.raw(qs{i}.Mu)));
                s = exp(0.5 * double(gather(OceanDL.Autograd.Tensor.raw(qs{i}.LogVar))));
                obj.LatentShape{i} = size(m, 1:3);
                mu = [mu; m(:)]; sg = [sg; s(:)]; %#ok<AGROW>
            end
            a = max(abs(mu)); if a == 0, a = 1; end
            obj.MuRange = [-a a];
            obj.SigmaRange = [0 max(max(sg), eps)];
            kls = obj.forwardSample(X, 0);                             % KL display range (sampled pass, all images)
            k = 0; for i = 1:numel(kls), k = max(k, max(double(gather(OceanDL.Autograd.Tensor.raw(kls{i}))), [], 'all')); end
            obj.KLRange = [0 max(k, eps)];
        end

        function [c, v] = catalog(obj)
            [c, v] = catalog@OceanDL.Interpret.TraceableRecordable(obj);
            C = @OceanReplay.Core.ChannelInfo; A = "image"; S = size(obj.X, 1:3); R = obj.DataRange;
            c = [c, C("xhat", "image", 'Axes', A, 'ValueShape', S, 'DisplayRange', R, 'Symbol', "xhat", ...
                    'SymbolLatex', '$\hat{x} = \mathrm{E}[\,p(x \mid z = \mu)\,]$', ...
                    'Description', "the reconstruction: decoder mean at the posterior means (no sampling)"), ...
                 C("xhatSample", "image", 'Axes', A, 'ValueShape', S, 'DisplayRange', R, 'Symbol', "E[p(x|z)]", ...
                    'SymbolLatex', '$\mathrm{E}[\,p(x \mid z)\,]$', 'Description', "decoder mean at the SAMPLED latents z")];
            for i = 1:obj.NumLatent
                L = obj.LatentShape{i}; t = sprintf('latent layer %d (%dx%dx%d, top = 1)', i, L);
                c = [c, ...
                    C(sprintf("mu_%d", i), "image", 'Axes', A, 'ValueShape', L, 'DisplayRange', obj.MuRange, 'Symbol', sprintf("mu_%d", i), ...
                      'SymbolLatex', sprintf('$\\mu_{%d}$', i), 'Description', "posterior mean of " + t + "; display range +-max |mu|"), ...
                    C(sprintf("sigma_%d", i), "image", 'Axes', A, 'ValueShape', L, 'DisplayRange', obj.SigmaRange, 'Symbol', sprintf("sigma_%d", i), ...
                      'SymbolLatex', sprintf('$\\sigma_{%d}$', i), 'Description', "posterior standard deviation of " + t + " (prior: 1)"), ...
                    C(sprintf("z_%d", i), "image", 'Axes', A, 'ValueShape', L, 'DisplayRange', obj.MuRange + [-3 3] * obj.SigmaRange(2), ...
                      'Symbol', sprintf("z_%d", i), 'SymbolLatex', sprintf('$z_{%d} = \\mu_{%d} + \\sigma_{%d} \\odot \\epsilon$', i, i, i), ...
                      'Description', "the reparameterised sample of " + t), ...
                    C(sprintf("klUnit_%d", i), "image", 'Axes', A, 'ValueShape', L, 'DisplayRange', obj.KLRange, 'Symbol', sprintf("KL_%d per unit", i), ...
                      'SymbolLatex', sprintf('$\\mathrm{KL}(q(z_{%d})\\,\\|\\,p(z_{%d}))$', i, i), ...
                      'Description', "KL of each unit of " + t + " (nats); ~0 = an inactive unit (posterior equals prior)"), ...
                    C(sprintf("kl_%d", i), "scalar", 'Axes', A, 'Symbol', sprintf("KL_%d", i), 'SymbolLatex', sprintf('$\\mathrm{KL}_{%d}$', i), ...
                      'Description', "KL of " + t + " (nats)")]; %#ok<AGROW>
            end
            c = [c, C("recon", "scalar", 'Axes', A, 'Symbol', "-log p(x|z)", 'SymbolLatex', '$-\log p(x \mid z)$', ...
                    'Description', "reconstruction term at the sampled z (nats)"), ...
                 C("kl", "scalar", 'Axes', A, 'Symbol', "KL", 'SymbolLatex', '$\mathrm{KL}$', 'Description', "total KL, all latent layers (nats)"), ...
                 C("negElbo", "scalar", 'Axes', A, 'Symbol', "-ELBO", 'SymbolLatex', '$-\mathrm{ELBO}$', ...
                    'Description', "-ELBO = -log p(x|z) + KL (one-sample estimate, nats); an upper bound on -log p(x)"), ...
                 C("bitsPerDim", "scalar", 'Axes', A, 'Symbol', "bits/dim", 'SymbolLatex', '$\mathrm{bits/dim}$', ...
                    'Description', "-ELBO / (D log 2), D = pixels")];
        end

        function d = describe(obj)
            d = describe@OceanDL.Interpret.TraceableRecordable(obj);
            d.model.seed = obj.Seed; d.model.latentLayers = obj.NumLatent; d.model.likelihood = obj.Model.Likelihood;
            d.references = {'Kingma & Welling (2019), An Introduction to Variational Autoencoders, Ch. 2', ...
                'Hierarchical posterior / prior (i > 1): NVAE, Vahdat & Kautz (2020) -- beyond the Ch. 2 VAE'};
            d.notation = struct('negElbo', "-ELBO = E_q[-log p(x|z)] + KL(q(z|x) || p(z)), one sample", ...
                'latents', "z_1 = top layer; q(z_i | x, z_<i), p(z_i | z_<i) for i > 1");
            d.layout = "vae";
        end
    end

    methods (Access = protected)
        function [state, value] = addState(obj, state, x, k)
            m = obj.Model;
            cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
            get = @(name) obj.sampled(cache, x, k, name);
            state.xhat = @() double(gather(m.reconstruct(x)));
            state.xhatSample = @() get('xhatSample');
            for i = 1:obj.NumLatent
                ii = i;
                state.(sprintf('mu_%d', i)) = @() obj.layerValue(get('mu'), ii);
                state.(sprintf('sigma_%d', i)) = @() obj.layerValue(get('sigma'), ii);
                state.(sprintf('z_%d', i)) = @() obj.layerValue(get('z'), ii);
                state.(sprintf('klUnit_%d', i)) = @() obj.layerValue(get('klUnit'), ii);
                state.(sprintf('kl_%d', i)) = @() sum(obj.layerValue(get('klUnit'), ii), 'all');
            end
            state.recon = @() get('recon');
            state.kl = @() get('kl');
            state.negElbo = @() get('negElbo');
            state.bitsPerDim = @() get('negElbo') / (numel(x) * log(2));
            value = get('negElbo');                                 % the run's output (invariance check)
        end

        function names = defaultLayers(~, order)
            names = order(~startsWith(order, "z"));                 % latents are recorded as mu/sigma/z/KL
        end
    end

    methods (Access = private)
        function [kls, out, zs, qs] = forwardSample(obj, X, k)
            % One reparameterised pass; eps from substream k (k = 0: display-range pass).
            s = RandStream('mrg32k3a', 'Seed', obj.Seed); s.Substream = k + 1;
            m = obj.Model;
            [out, kls, zs, qs] = OceanDL.Autograd.GradMode.noGrad(@() ...
                m.topDown(m.bottomUp(X), size(X, 4), size(X, 1:2), "sample", {}, 1, s));
        end

        function v = sampled(obj, cache, x, k, name)
            if ~isKey(cache, 'done')
                [kls, out, zs, qs] = obj.forwardSample(x, k);
                R = @(t) double(gather(OceanDL.Autograd.Tensor.raw(t)));
                cache('mu') = cellfun(@(q) R(q.Mu), qs, 'UniformOutput', false);
                cache('sigma') = cellfun(@(q) exp(0.5 * R(q.LogVar)), qs, 'UniformOutput', false);
                cache('z') = cellfun(R, zs, 'UniformOutput', false);
                cache('klUnit') = cellfun(R, kls, 'UniformOutput', false);
                recon = R(OceanDL.Generative.VAE.nll(out, x, obj.Model.Likelihood, obj.Model.LogSigma));
                kl = sum(cellfun(@(a) sum(a, 'all'), cache('klUnit')));
                cache('recon') = recon; cache('kl') = kl; cache('negElbo') = recon + kl;
                cache('xhatSample') = R(obj.Model.mean(out));
                cache('done') = true;
            end
            v = cache(name);
        end
    end

    methods (Static, Access = private)
        function v = layerValue(list, i)
            v = list{i};
        end
    end
end
