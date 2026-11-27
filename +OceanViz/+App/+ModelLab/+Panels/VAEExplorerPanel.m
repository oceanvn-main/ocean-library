classdef VAEExplorerPanel < OceanViz.App.ModelLab.Panels.PanelBase
    % VAEEXPLORERPANEL  Look inside a VAE / hierarchical VAE (Kingma & Welling,
    %   "An Introduction to Variational Autoencoders", Ch. 2; hierarchical: Ch. 4):
    %   the input x, the encoder's posterior mean mu_phi(x) of q_phi(z|x) (Sec. 2.1,
    %   2.5), the reconstruction = decoder p_theta(x|z) at z = mu_phi(x), and the KL
    %   term of the ELBO (Sec. 2.2) per latent layer. The latent-traversal slider
    %   (shift one latent channel, decode) is a diagnostic NOT in the reference text.

    properties (SetAccess = private)
        LayerDropdown
        ChannelField
        ValueSlider
        Views = struct()
        KLAxes
        Info
        Zs = {}                 % posterior means of the current image (per layer)
    end

    methods
        function t = title(~), t = "VAE explorer"; end
        function c = requires(~), c = "vae"; end
        function r = reference(~), r = "Kingma & Welling, An Introduction to VAEs: Ch. 2 (VAE, ELBO), Ch. 4 (hierarchical)"; end

        function build(obj, parent, ctx)
            obj.layout(parent, ctx, 2, 3);
            m = ctx.Model; s = obj.Side;
            s.heading('VAE explorer');
            s.label('x = the current image.');
            layers = "layer " + string(1:m.numLayers());
            if m.numLayers() > 1, layers(1) = layers(1) + " (top)"; end
            obj.LayerDropdown = s.dropdown('Latent layer:', cellstr(layers));
            obj.LayerDropdown.ValueChangedFcn = @(~, ~) obj.traverse();
            obj.ChannelField = s.number('Latent channel:', 1, [1 Inf]);
            obj.ChannelField.ValueChangedFcn = @(~, ~) obj.traverse();
            obj.ValueSlider = s.slider('Shift that channel by (prior sd = 1):', [-3 3], 0, @(~, ~) obj.traverse());
            obj.ValueSlider.ValueChangingFcn = @(~, e) obj.traverse(e.Value);
            obj.Views.A = obj.view(1, 1, 'x (input)');
            obj.Views.Recon = obj.view(1, 2, 'reconstruction: mean of p_\theta(x|z) at z = \mu_\phi(x)');
            obj.Views.Traverse = obj.view(1, 3, 'latent traversal (diagnostic, not in the text)');
            obj.Views.Latent = obj.view(2, 1, '\mu_\phi(x) of q_\phi(z|x), chosen channel');
            obj.KLAxes = obj.plotAxes(2, [2 3]);
            obj.refresh();
        end

        function onImageChanged(obj, ~)
            obj.refresh();
        end

        function refresh(obj)
            m = obj.Ctx.Model;
            obj.Zs = m.encodeMeans(obj.Ctx.Image);
            obj.Views.A.set(obj.Ctx.Image);
            obj.Views.Recon.set(m.decode(obj.Zs));
            obj.traverse();
            obj.drawKL();
        end

        function img = traverse(obj, value)
            if nargin < 2, value = obj.ValueSlider.Value; end
            m = obj.Ctx.Model;
            layer = find(string(obj.LayerDropdown.Items) == string(obj.LayerDropdown.Value));
            z = obj.Zs;
            c = min(size(z{layer}, 3), max(1, round(obj.ChannelField.Value)));
            z{layer}(:, :, c, :) = z{layer}(:, :, c, :) + value;       % prior sd = 1 per latent unit
            img = m.decode(z);
            obj.Views.Traverse.set(img);
            obj.Views.Traverse.setTitle(sprintf('layer %d, channel %d shifted by %+.2f', layer, c, value));
            zm = double(gather(obj.Zs{layer}(:, :, c, 1)));
            obj.Views.Latent.set((zm - min(zm(:))) / max(eps, max(zm(:)) - min(zm(:))));
        end

    end

    methods (Access = private)
        function drawKL(obj)
            % KL per latent layer (nats), averaged over dataset test images.
            m = obj.Ctx.Model; ax = obj.KLAxes; c = obj.Ctx.Theme;
            cla(ax);
            D = obj.Ctx.Dataset;
            if isempty(D) || ~m.isHierarchical()
                title(ax, 'KL per layer: hierarchical VAEs with a dataset'); return;
            end
            X = single(D.Xte(:, :, :, 1:min(200, end)));
            net = m.Model;
            OceanDL.Autograd.GradMode.noGrad(@() net.forward(X));
            kl = net.LastStats.KLPerLayer;
            bar(ax, kl, 'FaceColor', c.Primary);
            xlabel(ax, 'latent layer (1 = top)'); ylabel(ax, 'KL (nats / image)');
            title(ax, sprintf('KL(q_\\phi(z|x) || p(z)) per layer, ELBO term (total %.1f nats)', sum(kl)));
        end
    end
end
