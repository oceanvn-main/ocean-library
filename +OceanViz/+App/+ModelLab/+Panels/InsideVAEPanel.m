classdef InsideVAEPanel < OceanViz.App.ModelLab.Panels.PanelBase
    % INSIDEVAEPANEL  What a (hierarchical) VAE learns in the middle, with the
    %   feature-visualisation methods of Szeliski 2nd ed. Sec. 5.4 (pp. 309-311, Figs. 5.47-5.50) applied to
    %   the encoder, the latent code (Kingma & Welling, Ch. 2 / Ch. 4) and the
    %   decoder. Library: OceanDL.Interpret (any ITraceable model).
    %
    %   ENCODER (what each layer DETECTS)
    %     activation maps of every channel for the current x
    %     top-9 dataset patches per channel                 Fig. 5.47 right (Zeiler & Fergus)
    %     projections of those responses to pixels          Fig. 5.47 left: guided backprop /
    %                                                       deconvnet / saliency
    %     activation maximisation                           Fig. 5.49 (Olah et al.)
    %     occlusion map for the current x                   Fig. 5.48
    %   LATENT (what the code CARRIES, and what each unit DRAWS)
    %     KL per latent unit over the dataset               ELBO KL term, Sec. 2.2; inactive units Sec. 2.8
    %     top-9 digits per latent channel, mu maps, occlusion
    %     decoder atoms of a latent channel  -- NOT in the texts (announced in the panel)
    %   DECODER (how the image is BUILT)
    %     activation maps at every resolution; layer contributions (Ch. 4)
    %   NOT in the texts, labelled in the panel: decoder atoms; 2-D t-SNE map of the codes;
    %   sparse-autoencoder dictionary features of a layer (Bricken et al. 2023).
    %   Tools that are not in the texts are labelled with a warning sign in the panel.

    properties (SetAccess = private)
        SectionDropdown
        LayerDropdown
        ChannelField
        ViewDropdown
        ChannelHint            % says whether the current view uses the Channel field
        Main1
        Info
        Names                  % table from traceNames
        Cache                  % containers.Map: dataset scans, receptive fields, statistics
        Last = struct()        % last shown result (Image, Text) -- for tests / export
    end

    properties (Constant, Access = private)
        Views = struct( ...
            'encoder', {{'Activation maps (current x)', 'Top-9 patches (dataset)', ...
                         'Top-9 + projections: guided backprop', 'Top-9 + projections: deconvnet', ...
                         'Top-9 + projections: saliency', 'Activation maximisation', ...
                         'Activation maximisation: accentuate best real digit', 'Occlusion map (current x)', ...
                         'Sparse features: overview', 'Sparse feature # (= Channel field)', ...
                         'Hinton diagram: weights into channel #'}}, ...
            'latent',  {{'KL per latent unit (dataset)', 'Top-9 digits (dataset)', 'mu maps (current x)', ...
                         'Occlusion map (current x)', 'Decoder atoms', '2-D map of all codes (t-SNE)'}}, ...
            'decoder', {{'Activation maps (current x)', 'Layer contributions', ...
                         'Sparse features: overview', 'Sparse feature # (= Channel field)', ...
                         'Hinton diagram: weights into channel #'}})
    end

    methods
        function t = title(~), t = "Inside the VAE"; end
        function c = requires(~), c = "traceable"; end
        function r = reference(~)
            r = "Szeliski 2nd ed. Sec. 5.4 (pp. 309-311, Figs. 5.47-5.49) applied to a VAE; Kingma & Welling Ch. 2 and Ch. 4";
        end

        function build(obj, parent, ctx)
            obj.layout(parent, ctx, 2, 1);
            obj.Main.RowHeight = {'1x', 120};
            m = ctx.Model; s = obj.Side;
            obj.Cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.Names = m.Model.traceNames(m.ImageSize);
            s.heading('Inside the VAE');
            obj.SectionDropdown = s.dropdown('Section:', {'encoder', 'latent', 'decoder'});
            obj.SectionDropdown.ValueChangedFcn = @(~, ~) obj.onSection();
            obj.LayerDropdown = s.dropdown('Layer:', {'-'});
            obj.LayerDropdown.ValueChangedFcn = @(~, ~) obj.onView();
            obj.ChannelField = s.number('Channel:', 1, [1 Inf]);
            obj.ChannelField.ValueChangedFcn = @(~, ~) obj.onChannel();
            obj.ViewDropdown = s.dropdown('View:', {'-'});
            obj.ViewDropdown.ValueChangedFcn = @(~, ~) obj.onView();
            obj.ChannelHint = s.label('');
            s.button('Show', @(~, ~) obj.show(), true);
            s.label(['x = the current image (top bar). Dataset views use the test set ' ...
                     '(binarised like the training data when the model card says so).']);
            obj.Main1 = obj.view(1, 1, 'choose a view and press Show');
            obj.Info = obj.textBox(2, 1);
            obj.onSection();
        end

        function onImageChanged(obj, ~)
            v = string(obj.ViewDropdown.Value);
            if contains(v, "current x") || contains(v, "contributions"), obj.show(); end
        end

        function select(obj, section, layer, view, channel)
            % Scriptable selection (tests, reports).
            obj.SectionDropdown.Value = char(section); obj.onSection();
            if nargin > 2 && ~isempty(layer), obj.LayerDropdown.Value = char(layer); end
            if nargin > 3 && ~isempty(view), obj.ViewDropdown.Value = char(view); end
            obj.onView();
            if nargin > 4 && ~isempty(channel), obj.ChannelField.Value = min(channel, obj.ChannelField.Limits(2)); end
        end

        function R = show(obj)
            section = string(obj.SectionDropdown.Value);
            layer = string(obj.LayerDropdown.Value);
            view = string(obj.ViewDropdown.Value);
            obj.Ctx.status(sprintf('Computing %s ...', view)); drawnow;
            try
                [img, txt] = obj.compute(section, layer, view);
            catch err
                img = zeros(8); txt = {['Could not compute: ' err.message]};
            end
            obj.Main1.set(img);
            obj.Main1.setTitle(sprintf('%s | %s', obj.friendly(layer), view));
            obj.Info.Value = txt;
            obj.Last = struct('Image', img, 'Text', {txt});
            R = obj.Last;
            obj.Ctx.status(sprintf('%s: done.', view));
        end
    end

    methods (Access = private)
        function onSection(obj)
            section = string(obj.SectionDropdown.Value);
            N = obj.Names;
            switch section
                case "latent", layers = N.Name(startsWith(N.Name, "z") & endsWith(N.Name, ".mu"));
                case "encoder", layers = N.Name(N.Section == "encoder");
                otherwise, layers = N.Name(N.Section == "decoder");
            end
            obj.LayerDropdown.Items = cellstr(arrayfun(@(n) obj.friendly(n), layers));
            obj.LayerDropdown.ItemsData = cellstr(layers);                 % Value = the internal name
            items = obj.LayerDropdown.ItemsData;
            switch section
                case "encoder", obj.LayerDropdown.Value = items{end};
                case "decoder", obj.LayerDropdown.Value = items{max(1, numel(items) - 1)};   % dec.out.act: last multi-channel layer
                otherwise,      obj.LayerDropdown.Value = items{1};
            end
            obj.ViewDropdown.Items = obj.Views.(char(section));
            obj.ViewDropdown.Value = obj.ViewDropdown.Items{1};
            obj.onView();
        end

        function onView(obj)
            % Channel field: valid range for the current layer + what the number means in this view.
            layer = string(obj.LayerDropdown.Value);
            k = find(obj.Names.Name == layer, 1);
            dims = sscanf(char(obj.Names.Size(k)), '%dx%dx%d').';
            C = dims(3); hw = sprintf('%dx%d', dims(1), dims(2));
            if startsWith(layer, "z")
                what = sprintf('%s has %d channels, each an %s grid of latent units (%d numbers in total)', ...
                    extractBefore(layer, ".mu"), C, hw, prod(dims));
            else
                what = sprintf('this layer has %d channels, each an %s map', C, hw);
            end
            switch obj.channelUse(string(obj.ViewDropdown.Value))
                case "one"
                    obj.setChannelRange(C);
                    obj.ChannelHint.Text = sprintf('Type 1 to %d (%s). Pick one, then Show.', C, what);
                case "highlight"
                    obj.setChannelRange(C);
                    obj.ChannelHint.Text = sprintf(['Type 1 to %d (%s). This view shows ALL of them; ' ...
                        'the chosen one gets a white frame and its numbers appear below (updates at once).'], C, what);
                case "rank"
                    obj.setChannelRange(4 * C);
                    obj.ChannelHint.Text = sprintf(['Type a feature RANK 1 to %d: the dictionary has 4 x %d = %d features, ' ...
                        'ranked by how often they fire (1 = most frequent). Then Show.'], 4 * C, C, 4 * C);
                otherwise
                    obj.ChannelField.Enable = 'off';
                    obj.ChannelHint.Text = sprintf('Channel is not used by this view (%s).', what);
            end
        end

        function setChannelRange(obj, n)
            obj.ChannelField.Enable = 'on';
            obj.ChannelField.Limits = [1 max(1, n)];
            obj.ChannelField.Value = min(max(1, round(obj.ChannelField.Value)), max(1, n));
            obj.ChannelField.RoundFractionalValues = 'on';
            obj.ChannelField.Tooltip = sprintf('1 to %d', n);
        end

        function onChannel(obj)
            % All-channel views redraw immediately with the new frame.
            if obj.channelUse(string(obj.ViewDropdown.Value)) == "highlight", obj.show(); end
        end

        function u = channelUse(~, view)
            switch view
                case {"Activation maps (current x)", "mu maps (current x)", "KL per latent unit (dataset)"}
                    u = "highlight";
                case {"Layer contributions", "2-D map of all codes (t-SNE)", "Sparse features: overview"}
                    u = "none";
                case "Sparse feature # (= Channel field)"
                    u = "rank";
                otherwise
                    u = "one";
            end
        end

        function t = friendly(obj, name)
            % Descriptive layer label, internal name in brackets.
            name = string(name); N = obj.Names;
            k = find(N.Name == name, 1);
            dims = sscanf(char(N.Size(k)), '%dx%dx%d').';
            hw = sprintf('%dx%d', dims(1), dims(2));
            nLat = nnz(startsWith(N.Name, "z") & endsWith(N.Name, ".mu"));
            tok = regexp(char(name), '^(enc|dec)\.L(\d+)\.(\w+?)(\d*)$', 'tokens', 'once');
            if startsWith(name, "z")
                i = sscanf(char(name), 'z%d');
                role = ""; if nLat > 1 && i == 1, role = "top, "; elseif nLat > 1 && i == nLat, role = "detail, "; end
                t = sprintf('Latent z%d (%s%s): posterior mean mu', i, role, N.Size(k));
            elseif name == "dec.out.act"
                t = sprintf('Decoder %s: final features', hw);
            elseif name == "dec.out"
                t = sprintf('Decoder %s: output (ink logits)', hw);
            elseif ~isempty(tok) && tok{1} == "enc"
                switch tok{3}
                    case 'in'
                        if tok{2} == "1", t = sprintf('Encoder %s: first conv (edges)', hw);
                        else, t = sprintf('Encoder %s: downsample conv (/2)', hw); end
                    otherwise, t = sprintf('Encoder %s: residual block %s', hw, tok{4});
                end
            elseif ~isempty(tok)
                switch tok{3}
                    case 'start', t = sprintf('Decoder %s: learned start (constant)', hw);
                    case 'td'
                        before = N.Name(1:k); z = before(startsWith(before, "z") & endsWith(before, ".mu"));
                        t = sprintf('Decoder %s: after adding %s', hw, extractBefore(z(end), ".mu"));
                    case 'res', t = sprintf('Decoder %s: residual block %s', hw, tok{4});
                    case 'lift', t = sprintf('Decoder %s: upsample x2 + conv', hw);
                    otherwise, t = char(name);
                end
            else
                t = char(name);
            end
            t = string(t) + "  [" + name + "]";
        end

        function [img, txt] = compute(obj, section, layer, view)
            net = obj.Ctx.Model.Model;
            c = obj.ChannelField.Value;
            x = obj.currentX();
            switch view
                case {"Activation maps (current x)", "mu maps (current x)"}
                    a = OceanDL.Interpret.Probe.activation(net, x, layer);
                    c = min(c, size(a, 3));
                    img = obj.withInput(obj.channelGrid(a, c), x);
                    [~, best] = max(squeeze(max(a, [], [1 2])));
                    txt = {sprintf('%s for the current x: %d channels of %dx%d (grey = 0, bright = +, dark = -).', ...
                        layer, size(a, 3), size(a, 1), size(a, 2)), ...
                        ['LEFT (large) = the current x itself. HOW IT WAS MADE: ONE forward pass of the current x (top bar) up to this layer; each tile = one channel''s ' ...
                         'output map for that single image; all tiles share one scale (the largest |value| of any channel).'], ...
                        sprintf('Strongest channel for this x: %d.  Channel %d (white frame): max %.2f, min %.2f, mean %.3f.', ...
                            best, c, max(a(:, :, c), [], 'all'), min(a(:, :, c), [], 'all'), mean(a(:, :, c), 'all')), ...
                        'Source: activation maps (Szeliski Sec. 5.4 (pp. 309-311, Figs. 5.47-5.50)).'};
                case {"Top-9 patches (dataset)", "Top-9 digits (dataset)"}
                    [T, rf, X] = obj.scan(layer);
                    c = min(c, size(T.Value, 1));
                    [P, cols, nTie] = obj.distinctTop(T, rf, X, c, section == "latent", 9);
                    what = "patches"; if section == "latent", what = "digits"; end
                    txt = {sprintf('Channel %d of %s: the %d strongest DISTINCT dataset %s (identical crops shown once).', c, layer, numel(cols), what), ...
                        sprintf(['HOW THEY WERE PICKED: for every one of the %d test digits, score = the HIGHEST value of channel %d''s ' ...
                            'map over all positions (its strongest response anywhere in that digit; raw value, no normalisation). ' ...
                            'Digits ranked by score, highest first; the first %d with different crops are shown.'], size(X, 4), c, numel(cols)), ...
                        sprintf('Scores of the shown %s: %s', what, mat2str(round(T.Value(c, cols), 2))), ...
                        sprintf('Effective receptive field %dx%d px (Luo et al. 2016 -- tool not in the texts).', rf.Size), ...
                        'Source: Szeliski Fig. 5.47 right (Zeiler & Fergus 2014).'};
                    if nTie > 1
                        txt{end+1} = sprintf(['%d dataset positions tie for the maximum: on 0/1 (binarised) images a linear filter ' ...
                            'is maximised by ONE exact pattern -- ink where the weight is +, background where it is -.'], nTie);
                    end
                    K = obj.firstLayerKernel(layer, c, rf);
                    if ~isempty(K)
                        P = cat(4, P, K);
                        txt{end+1} = 'LAST TILE = the filter weights of this channel (bright = positive, dark = negative, grey = 0).';
                    end
                    img = OceanViz.App.Common.ImageView.tile(P, min(5, size(P, 4)));
                case {"Top-9 + projections: guided backprop", "Top-9 + projections: deconvnet", "Top-9 + projections: saliency"}
                    rule = extractAfter(view, "projections: ");
                    rule = replace(rule, ["guided backprop", "saliency"], ["guided", "standard"]);
                    [T, rf, X] = obj.scan(layer);
                    c = min(c, size(T.Value, 1));
                    [P, cols] = obj.distinctTop(T, rf, X, c, false, 9);
                    G = zeros(size(P));
                    for q = 1:numel(cols)
                        j = cols(q);
                        g = OceanDL.Interpret.Probe.unitGradient(net, X(:, :, :, T.Image(c, j)), layer, c, T.Row(c, j), T.Col(c, j), rule);
                        G(:, :, :, q) = OceanDL.Interpret.ReceptiveField.crop(double(g), rf, T.Row(c, j), T.Col(c, j));
                    end
                    G = 0.5 + G / (2 * max(abs(G(:))) + eps);
                    img = OceanViz.App.Common.ImageView.tile(cat(4, P, G), size(P, 4));
                    txt = {sprintf('Channel %d of %s. TOP ROW: the %d winning patches, cut from the original (binarised) digits.', c, layer, numel(cols)), ...
                        sprintf(['HOW THEY WERE PICKED: score of each test digit = the HIGHEST value of channel %d''s map over all positions; ' ...
                            'digits ranked by score, first %d distinct crops (scores %s).'], c, numel(cols), mat2str(round(T.Value(c, cols), 2))), ...
                        sprintf(['BOTTOM ROW: for EACH winner separately, the single winning unit (channel %d at its winning position) ' ...
                            'projected back to the pixels with the %s rule: how much each pixel of that patch contributes to that one unit ' ...
                            '(grey = 0, bright = supports it, dark = works against it; one shared scale). Not a reconstruction.'], c, rule), ...
                        'Source: Szeliski Fig. 5.47 left (Zeiler & Fergus deconvnet; Springenberg guided backprop).'};
                    switch rule
                        case "standard"
                            txt{end+1} = ['RULE = saliency (Simonyan et al. 2014): the TRUE gradient d(unit)/d(pixel) -- how much the unit would ' ...
                                'change if that pixel changed slightly (positive and negative signals both pass back).'];
                        case "deconvnet"
                            txt{end+1} = ['RULE = deconvnet (Zeiler & Fergus 2014, the rule of Fig. 5.47): transposed filters going back; at every ' ...
                                'activation only POSITIVE signals pass, ignoring the forward activation -- shows what supports the unit.'];
                        case "guided"
                            txt{end+1} = ['RULE = guided backprop (Springenberg et al. 2015): only POSITIVE signals, and only through units that ' ...
                                'were ACTIVE in the forward pass (deconvnet + gradient mask) -- the sharpest, most patch-specific picture.'];
                    end
                    if rule ~= "standard"
                        txt{end+1} = ['NOTE: the texts define these rules for ReLU; this VAE uses SiLU -- the SiLU form ' ...
                            'is our adaptation (not in the texts).'];
                    end
                case "Activation maximisation"
                    [xs, info] = OceanDL.Interpret.FeatureVis.maximize(net, layer, c, 'InputSize', obj.Ctx.Model.ImageSize);
                    [xc, infoC] = OceanDL.Interpret.FeatureVis.maximize(net, layer, c, 'InputSize', obj.Ctx.Model.ImageSize, 'Objective', "channel");
                    [T, ~, ~] = obj.scan(layer);
                    bestReal = T.Value(min(c, size(T.Value, 1)), 1);
                    img = OceanViz.App.Common.ImageView.tile(cat(4, xs, xc), 2);
                    txt = {sprintf('Channel %d of %s. Left: input that maximises the CENTRE unit (activation %.2f);', c, layer, info.Activation), ...
                        sprintf('right: input that maximises the whole channel (mean activation %.3f).', infoC.Activation), ...
                        ['HOW IT WAS MADE: no dataset -- start from a plain grey image; 200 times: shift it randomly by <=1 px (jitter), ' ...
                         'forward pass, read the objective (centre unit / channel mean), backward pass to the image, small Adam step ' ...
                         'that INCREASES the objective. The image is built from smooth DCT waves (1/frequency scaled), so noise is slow to appear.'], ...
                        'Source: Szeliski Fig. 5.49 (Olah et al. 2017). Regularisers as in the Lucid/Lucent libraries:', ...
                        'DCT decorrelated parameterisation, jitter, L2 -- tools, not in the texts.'};
                    if info.Activation < bestReal
                        txt = [{sprintf(['RELIABILITY: WEAK -- the synthetic image reaches %.2f, BELOW the best real digit (%.2f). ' ...
                            'The optimisation did not find the preferred pattern; treat these images as inconclusive. ' ...
                            'Try "Activation maximisation: accentuate best real digit", or read the Top-9 views.'], info.Activation, bestReal)}, txt];
                    else
                        txt = [{sprintf('RELIABILITY: OK -- the synthetic image reaches %.2f, at least the best real digit (%.2f).', ...
                            info.Activation, bestReal)}, txt];
                    end
                case "Activation maximisation: accentuate best real digit"
                    [T, ~, X] = obj.scan(layer);
                    c = min(c, size(T.Value, 1));
                    x0 = X(:, :, :, T.Image(c, 1));
                    [xa, info] = OceanDL.Interpret.FeatureVis.maximize(net, layer, c, 'InputSize', obj.Ctx.Model.ImageSize, ...
                        'Init', x0, 'Row', T.Row(c, 1), 'Col', T.Col(c, 1), 'Decay', 0, 'Jitter', 0, 'LR', 0.2);
                    d = double(xa) - double(info.Start);
                    D = 0.5 + d / (2 * max(abs(d(:))) + eps);
                    img = OceanViz.App.Common.ImageView.tile(cat(4, double(info.Start), double(xa), D), 3);
                    txt = {sprintf(['WARNING -- NOT IN THE TEXTS: feature accentuation (Hamblin et al. 2024). Channel %d of %s, the unit at ' ...
                            'the winning position (%d, %d) of its best real digit (test digit #%d).'], c, layer, T.Row(c, 1), T.Col(c, 1), T.Image(c, 1)), ...
                        sprintf(['LEFT: that real digit (start). MIDDLE: after 200 steps that INCREASE this one unit. RIGHT: the change ' ...
                            '(bright = ink added, dark = ink removed, grey = unchanged).  Unit value: %.2f -> %.2f.'], info.StartActivation, info.Activation), ...
                        ['HOW IT WAS MADE: the real digit is written as smooth DCT waves (so added ink stays stroke-like); 200 Adam steps ' ...
                         '(learning rate 0.2) change those waves to INCREASE its winning unit -- no jitter (the unit is tied to a real ' ...
                         'position) and no L2 penalty (it would pull the digit back to grey). Reads as: "what would this unit like to see ' ...
                         'MORE of in a real example?" -- the added ink is the pattern it is tuned to.'], ...
                        'Why: on small, binary-trained models, optimising from grey often gets stuck in weak blurs; starting from real data does not.'};
                case "Hinton diagram: weights into channel #"
                    if ~ismethod(net, 'weightsFor'), error('This model does not expose its weights.'); end
                    W = double(net.weightsFor(layer));
                    if isempty(W), error('"%s" is not produced by a convolution (no weights to show).', obj.friendly(layer)); end
                    c = min(c, size(W, 4));
                    Wc = W(:, :, :, c);                                          % kh x kw x Cin
                    [kh, kw, Cin] = size(Wc, 1:3);
                    if Cin == 1, M = Wc; rowsTxt = 'rows/cols = the kernel''s spatial positions';
                    else, M = reshape(permute(Wc, [3 2 1]), Cin, kw * kh).'; rowsTxt = sprintf(['columns = the %d input channels (1 = left), ' ...
                        'rows = the %d kernel positions (top-left, top-middle, ..., bottom-right)'], Cin, kw * kh); end
                    img = obj.hintonImage(M);
                    [~, strongest] = sort(squeeze(sum(abs(Wc), [1 2])), 'descend');
                    txt = {sprintf('WARNING -- NOT IN THE TEXTS: Hinton diagram (G. Hinton''s classic weight plot) of the %dx%d convolution that produces channel %d of %s.', kh, kw, c, obj.friendly(layer)), ...
                        sprintf('Each square = one weight: AREA proportional to |w|, WHITE = positive, BLACK = negative. %s.', rowsTxt), ...
                        sprintf('Largest |w| = %.3f.', max(abs(Wc(:)))), ...
                        ['HOW IT WAS MADE: the trained weights of that convolution are read directly -- no data, no forward pass. ' ...
                         'Each square''s side is sqrt(|w| / max|w|) of the cell, so its area is proportional to |w|.']};
                    if Cin > 1
                        txt{end+1} = sprintf('Input channels with the largest total |w| into channel %d: %s.', c, mat2str(strongest(1:min(8, end)).'));
                    end
                case "Occlusion map (current x)"
                    f = OceanDL.Interpret.Occlusion.unitTarget(layer, c, [], []);
                    [M, base] = OceanDL.Interpret.Occlusion.map(net, x(:, :, :, 1), f, layer, 'Size', 3, 'Stride', 1, 'Fill', 0);
                    Ms = 0.5 + M / (2 * max(abs(M(:))) + eps);
                    img = OceanViz.App.Common.ImageView.tile(cat(4, x(:, :, :, 1), Ms), 2);
                    txt = {sprintf('Channel %d of %s (mean response %.3f). Left: x; right: drop of the response when a', c, layer, base), ...
                        '3x3 square is erased (fill 0) -- bright = that region drives the channel, dark = it suppresses it.', ...
                        sprintf(['HOW IT WAS MADE: response = channel %d''s MEAN over its map for the current x (%.3f). %d copies of x, ' ...
                            'each with a different 3x3 square set to 0, go through the network in one batch; drop = %.3f - response(copy); ' ...
                            'each pixel shows the AVERAGE drop of all squares that covered it.'], c, base, (size(x, 1) - 2) * (size(x, 2) - 2), base), ...
                        'Source: Szeliski Fig. 5.48 (Zeiler & Fergus occlusion).'};
                case "KL per latent unit (dataset)"
                    kl = obj.stat("kl");
                    i = obj.latentIndex(layer);
                    k = kl{i};
                    img = obj.channelGrid(k / max(k(:)) * 2 - 1, min(c, size(k, 3)));
                    active = nnz(k > 0.01);
                    txt = {sprintf('KL( q_phi(z|x) || p(z) ) of every unit of %s, averaged over the test set (%d units, total %.1f nats).', ...
                        layer, numel(k), sum(k(:))), ...
                        sprintf('Bright = carries information; dark = ~0 KL = inactive (Kingma & Welling Sec. 2.8). Units with KL > 0.01 nats: %d.', active), ...
                        sprintf('Per channel (nats): %s', mat2str(round(squeeze(sum(k, [1 2])).', 1))), ...
                        ['HOW IT WAS MADE: every test digit is encoded; for every latent unit the closed-form KL between its posterior ' ...
                         'q = N(mu, sigma^2) and its prior is computed and averaged over all test digits. Tiles: one per channel, one cell per position.'], ...
                        'Source: the ELBO KL term (Sec. 2.2), split per unit. The 0.01-nat activity threshold is our choice.'};
                case "Layer contributions"
                    [imgs, labels] = OceanDL.Interpret.DecoderAtoms.contributions(net, x(:, :, :, 1));
                    img = OceanViz.App.Common.ImageView.tile(imgs, size(imgs, 4));
                    txt = {sprintf('Left to right: %s.', strjoin(labels, ' | ')), ...
                        'Each panel decodes the posterior means of the top layers only; the remaining layers take their', ...
                        'prior mean p(z_i | z_<i). Shows what each latent layer adds (Kingma & Welling Ch. 4: several latent layers).', ...
                        ['HOW IT WAS MADE: the current x is encoded (posterior means mu of every layer); then decoded several times, ' ...
                         'each time keeping the codes of one more layer from the top and setting the others to their prior mean.']};
                case "Decoder atoms"
                    z0 = obj.stat("mu");
                    i = obj.latentIndex(layer);
                    c = min(c, size(z0{i}, 3));
                    [A, info] = OceanDL.Interpret.DecoderAtoms.atoms(net, z0, i, c, obj.Ctx.Model.ImageSize(1:2));
                    A = 0.5 + A / (2 * max(abs(A(:))) + eps);
                    img = OceanViz.App.Common.ImageView.tile(A, size(z0{i}, 2));
                    txt = {sprintf('WARNING -- NOT IN THE TEXTS: decoder atoms of latent layer z%d, channel %d: what ONE latent unit draws', i, c), ...
                        sprintf('(d E[x|z] / d z_j at the data-average code; one panel per position, %d x %d grid).', max(info.Rows), max(info.Cols)), ...
                        'Bright = the unit adds ink there, dark = removes ink. Standard analysis of generative decoders.', ...
                        ['HOW IT WAS MADE: reference code z0 = the average posterior mean over the test set. For each position of this ' ...
                         'channel: decode z0 with that ONE unit raised by +0.1 and lowered by -0.1; atom = difference of the two decoded ' ...
                         'images / 0.2 (all other units fixed).']};
                case "2-D map of all codes (t-SNE)"
                    i = obj.latentIndex(layer);
                    [xy, y, agree] = obj.codeMap(i, x);
                    img = obj.scatterImage(xy(1:end-1, :), y, xy(end, :));
                    names = obj.Ctx.Dataset.ClassNames;
                    pal = obj.paletteNames();
                    legendText = strjoin(compose("%s=%s", names(:), pal(1:numel(names)).'), ', ');
                    txt = {sprintf('WARNING -- NOT IN THE TEXTS: t-SNE map (van der Maaten & Hinton 2008) of the posterior means mu_phi(x) of %s', layer), ...
                        sprintf('for all %d test digits, coloured by the TRUE digit (not seen by the VAE); white ring = the current x.', size(y, 1)), ...
                        char("Colours: " + legendText), ...
                        sprintf('In the latent space itself (not the map): the 10 nearest codes share the digit %.0f%% of the time (chance ~10%%).', 100 * agree), ...
                        ['HOW IT WAS MADE: every test digit (and the current x) is encoded; its code mu (all positions and channels of this ' ...
                         'layer, as one long vector) is placed on the 2-D map by t-SNE, which keeps codes that are close in the latent space ' ...
                         'close on the map (distances between far-apart clusters are NOT meaningful).']};
                case {"Sparse features: overview", "Sparse feature # (= Channel field)"}
                    [D, TS, rf, X, order, S] = obj.dictionary(layer);
                    if view == "Sparse features: overview"
                        n = min(64, numel(order));
                        P = zeros(rf.Size(1), rf.Size(2), size(X, 3), n, 'like', X);
                        for j = 1:n
                            f = order(j);
                            P(:, :, :, j) = OceanDL.Interpret.ReceptiveField.crop(X(:, :, :, TS.Image(f, 1)), rf, TS.Row(f, 1), TS.Col(f, 1));
                        end
                        img = OceanViz.App.Common.ImageView.tile(P, 8);
                        txt = {sprintf('WARNING -- NOT IN THE TEXTS: sparse-autoencoder dictionary of %s (Bricken et al. 2023): %d features learned', layer, size(D.Wd, 1)), ...
                            sprintf('from %d channels, each a MIX of channels. Shown: the strongest patch of the %d most frequent features (ranked).', size(D.Wd, 2), n), ...
                            sprintf('Dictionary quality: explained variance %.1f%%, %.1f active features per position, %d/%d features alive, L1 %.2f.', ...
                                100 * S.ExplainedVariance, S.L0, S.Alive, size(D.Wd, 1), D.L1), ...
                            sprintf(['HOW IT WAS MADE: every POSITION of every test digit gives one example = its %d channel values at this layer ' ...
                                '(random subset, ~6000 per 256 digits). A sparse autoencoder learns %d features (fixed channel mixes, unit length) ' ...
                                'that rebuild each example from as FEW features as possible (1500 Adam steps). Features are ranked by how often ' ...
                                'they fire; each tile = the patch where that feature is strongest (highest code value over all test digits).'], ...
                                size(D.Wd, 2), size(D.Wd, 1)), ...
                            'Pick one with "Sparse feature #" (Channel field = rank) to see its 9 patches and which channels it combines.'};
                    else
                        r = min(c, numel(order)); f = order(r);
                        [P, cols] = obj.distinctTop(TS, rf, X, f, false, 9);
                        img = OceanViz.App.Common.ImageView.tile(P, 3);
                        [w, ch] = sort(abs(D.Wd(f, :)), 'descend');
                        top = ch(1:min(5, end));
                        mix = strjoin(compose("ch%d %+.2f", top(:), D.Wd(f, top).'), ', ');
                        txt = {sprintf('WARNING -- NOT IN THE TEXTS: sparse feature ranked %d (id %d) of %s -- fires at %.2f%% of positions.', r, f, layer, 100 * S.Frequency(f)), ...
                            sprintf(['Its %d strongest DISTINCT dataset patches. HOW THEY WERE PICKED: score of each test digit = the HIGHEST ' ...
                                'value of this feature''s code f over all positions; ranked, first distinct crops (scores %s).'], ...
                                numel(cols), mat2str(round(TS.Value(f, cols), 2))), ...
                            char(sprintf("It combines channels (largest atom weights): %s (%.0f%% of the atom norm in these 5).", mix, 100 * sum(w(1:numel(top)) .^ 2))), ...
                            'Source: sparse dictionary learning on activations (Bricken et al. 2023); patches as in Szeliski Fig. 5.47.'};
                    end
                otherwise
                    error('Unknown view "%s".', view);
            end
        end

        function x = currentX(obj)
            m = obj.Ctx.Model;
            like = OceanDL.Interpret.ITraceable.likeOf(m.Model);
            x = cast(reshape(double(obj.Ctx.Image), m.ImageSize), 'like', like);
        end

        function X = dataset(obj)
            if isKey(obj.Cache, 'X'), X = obj.Cache('X'); return; end
            D = obj.Ctx.Dataset;
            if isempty(D), error('This model card names no dataset; dataset views need one.'); end
            like = OceanDL.Interpret.ITraceable.likeOf(obj.Ctx.Model.Model);
            X = double(D.Xte);
            meta = obj.Ctx.Model.Meta;
            if isfield(meta, 'Binarized') && meta.Binarized
                X = double(rand(RandStream('mt19937ar', 'Seed', 12), size(X)) < X);   % as in training
            end
            X = cast(X, 'like', like);
            obj.Cache('X') = X;
        end

        function [T, rf, X] = scan(obj, layer)
            X = obj.dataset();
            key = char("scan:" + layer);
            if ~isKey(obj.Cache, key)
                net = obj.Ctx.Model.Model;
                T = OceanDL.Interpret.TopActivations.scan(net, X, layer, 'K', size(X, 4));   % best position of EVERY image; shown: 9 distinct
                if startsWith(layer, "z")
                    rf = struct('Size', size(X, 1:2), 'Stride', [1 1], 'LayerSize', [1 1]);
                else
                    rf = OceanDL.Interpret.ReceptiveField.effective(net, X, layer);
                end
                obj.Cache(key) = {T, rf};
            end
            v = obj.Cache(key); T = v{1}; rf = v{2};
        end

        function [P, cols, nTie] = distinctTop(~, T, rf, X, c, wholeImage, n)
            % The n strongest candidates of channel c whose crops differ (ties / repeats shown once).
            P = zeros(rf.Size(1), rf.Size(2), size(X, 3), 0, 'like', X);
            if wholeImage, P = zeros(size(X, 1), size(X, 2), size(X, 3), 0, 'like', X); end
            cols = [];
            for j = 1:size(T.Value, 2)
                if wholeImage, q = X(:, :, :, T.Image(c, j));
                else, q = OceanDL.Interpret.ReceptiveField.crop(X(:, :, :, T.Image(c, j)), rf, T.Row(c, j), T.Col(c, j)); end
                if any(arrayfun(@(k) isequal(P(:, :, :, k), q), 1:size(P, 4))), continue; end
                P = cat(4, P, q); cols(end+1) = j; %#ok<AGROW>
                if numel(cols) == n, break; end
            end
            nTie = nnz(abs(T.Value(c, :) - T.Value(c, 1)) < 1e-6 * max(1, abs(T.Value(c, 1))));
        end

        function K = firstLayerKernel(obj, layer, c, rf)
            % The weights of channel c of the first convolution, as a tile of the patch size (else []).
            K = [];
            net = obj.Ctx.Model.Model;
            if layer ~= "enc.L1.in" || ~isprop(net, 'ConvIn'), return; end
            W = double(OceanDL.Autograd.Tensor.raw(net.ConvIn.Weight.Data(:, :, 1, c)));
            W = 0.5 + W / (2 * max(abs(W(:))) + eps);
            K = 0.5 * ones(rf.Size(1), rf.Size(2));
            [kh, kw] = size(W);
            r0 = floor((rf.Size(1) - kh) / 2); c0 = floor((rf.Size(2) - kw) / 2);
            if r0 >= 0 && c0 >= 0, K(r0 + (1:kh), c0 + (1:kw)) = W; else, K = imresize(W, rf.Size, 'nearest'); end
        end

        function s = stat(obj, what)
            key = char("stat:" + what);
            if ~isKey(obj.Cache, key)
                X = obj.dataset(); net = obj.Ctx.Model.Model;
                if what == "kl", obj.Cache(key) = OceanDL.Interpret.LatentStats.klPerUnit(net, X);
                else, obj.Cache(key) = OceanDL.Interpret.LatentStats.meanCodes(net, X); end
            end
            s = obj.Cache(key);
        end

        function [xy, y, agree] = codeMap(obj, i, x)
            % t-SNE of the dataset codes of latent layer i (+ the current x as the last row); cached.
            key = sprintf('tsne:%d', i);
            net = obj.Ctx.Model.Model;
            X = obj.dataset();
            if ~isKey(obj.Cache, key)
                Z = OceanDL.Interpret.LatentStats.codes(net, X, i);
                y = double(obj.Ctx.Dataset.yte(:));
                d = pdist2(Z, Z); d(1:size(d, 1) + 1:end) = Inf;
                [~, nn] = mink(d, 10, 2);
                agree = mean(y(nn) == y, 'all');
                obj.Cache(key) = {Z, y, agree};
            end
            v = obj.Cache(key); Z = v{1}; y = v{2}; agree = v{3};
            z = OceanDL.Interpret.LatentStats.codes(net, x, i);
            prev = rng; rng(0);                                      % reproducible t-SNE
            m = OceanViz.Adapters.EmbeddingMapper2D([Z; z], 'Method', 'tsne');
            rng(prev);
            xy = m.EmbeddedPositions_;
        end

        function [D, TS, rf, X, order, S] = dictionary(obj, layer)
            % Sparse autoencoder on the layer's position vectors over the dataset; cached per layer.
            key = char("sae:" + layer);
            [~, rf, X] = obj.scan(layer);
            if ~isKey(obj.Cache, key)
                net = obj.Ctx.Model.Model;
                A = [];
                for b = 1:256:size(X, 4)
                    k = b:min(size(X, 4), b + 255);
                    a = OceanDL.Interpret.Probe.activation(net, X(:, :, :, k), layer);
                    A = [A; OceanDL.Interpret.SparseDictionary.samples(a, 6000, b)]; %#ok<AGROW>
                end
                D = OceanDL.Interpret.SparseDictionary(size(A, 2), 'Features', 4 * size(A, 2), 'L1', 0.05, 'Seed', 0);
                D.train(A, 'Steps', 1500, 'Batch', 1024, 'LR', 3e-3);
                S = D.metrics(A);
                TS = OceanDL.Interpret.TopActivations.scan(net, X, layer, 'K', size(X, 4), 'Transform', @(a) D.featureMaps(a));
                [~, order] = sort(S.Frequency, 'descend');
                order = order(S.Frequency(order) > 0);
                obj.Cache(key) = {D, TS, order, S};
            end
            v = obj.Cache(key); [D, TS, order, S] = v{:};
        end

        function img = scatterImage(obj, xy, y, here)
            % Rasterised scatter plot (so it shows in the image view): 3x3 dots in the class colour.
            n = 400; img = 0.08 * ones(n, n, 3);
            lo = min([xy; here], [], 1); hi = max([xy; here], [], 1);
            P = round(1 + (n - 1) * ([xy; here] - lo) ./ max(hi - lo, eps));
            pal = obj.palette();
            for k = 1:size(xy, 1)
                r = min(n, max(1, P(k, 2) + (-1:1))); cc = min(n, max(1, P(k, 1) + (-1:1)));
                img(r, cc, :) = repmat(reshape(pal(y(k), :), 1, 1, 3), numel(r), numel(cc));
            end
            [dy, dx] = ndgrid(-6:6, -6:6); ring = abs(sqrt(dy .^ 2 + dx .^ 2) - 5) < 1;
            r = P(end, 2) + dy(ring); cc = P(end, 1) + dx(ring); ok = r >= 1 & r <= n & cc >= 1 & cc <= n;
            for k = find(ok).', img(r(k), cc(k), :) = 1; end
        end

        function G = withInput(~, G, x)
            % The input image, scaled to the grid's height, left of the grid (grey separator).
            xs = imresize(double(x(:, :, 1, 1)), [size(G, 1) NaN], 'nearest');
            G = [min(1, max(0, xs)), 0.2 * ones(size(G, 1), 4), G];
        end

        function img = hintonImage(~, M)
            % Hinton diagram of matrix M: one cell per entry, a centred square with area ~ |w|.
            cell = 12; [r, c] = size(M);
            img = 0.5 * ones(r * cell, c * cell);
            mx = max(abs(M(:))) + eps;
            for i = 1:r
                for j = 1:c
                    side = round(cell * sqrt(abs(M(i, j)) / mx));
                    if side == 0, continue; end
                    o = floor((cell - side) / 2);
                    img((i - 1) * cell + o + (1:side), (j - 1) * cell + o + (1:side)) = double(M(i, j) > 0);
                end
            end
        end

        function pal = palette(~)
            pal = [1 0.2 0.2; 1 0.55 0; 1 1 0.2; 0.55 1 0.2; 0.1 0.8 0.3; 0.2 1 1; 0.2 0.55 1; 0.35 0.35 1; 0.7 0.3 1; 1 0.3 0.9];
        end

        function names = paletteNames(~)
            names = ["red", "orange", "yellow", "lime", "green", "cyan", "sky blue", "blue", "purple", "pink"];
        end

        function i = latentIndex(~, layer)
            i = sscanf(char(layer), 'z%d');
        end

        function G = channelGrid(~, a, highlight)
            % (h x w x C x 1) -> tiled image, one panel per channel, shared signed scale;
            % channel `highlight` gets a white frame (the 1-px separators around its panel).
            a = double(a(:, :, :, 1));
            s = max(abs(a(:))) + eps;
            [h, w, C] = size(a, 1:3);
            A = reshape(0.5 + a / (2 * s), h, w, 1, C);
            cols = min(8, C);
            G = OceanViz.App.Common.ImageView.tile(A, cols);
            if nargin > 2 && ~isempty(highlight) && highlight >= 1 && highlight <= C
                r = floor((highlight - 1) / cols); c = mod(highlight - 1, cols);
                rows = r * (h + 1) + (1:h + 2); colsIdx = c * (w + 1) + (1:w + 2);
                G(rows([1 end]), colsIdx) = 1; G(rows, colsIdx([1 end])) = 1;
            end
        end
    end
end
