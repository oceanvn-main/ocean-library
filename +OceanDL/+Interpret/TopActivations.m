classdef TopActivations
    % TOPACTIVATIONS  For every channel of a layer, the K dataset images (and
    %   positions) with the strongest response -- Szeliski 2nd ed. Sec. 5.4 (pp. 309-311, Figs. 5.47-5.50),
    %   Fig. 5.47 right (Zeiler & Fergus 2014: "top nine responses in one
    %   feature map"). One response per image: the channel's maximum over
    %   positions, so one image appears at most once per channel.
    %
    %   T = TopActivations.scan(model, X, layer, 'K', 9, 'Batch', 256, 'Transform', f)
    %       'Transform' f(a): maps the layer's (h x w x C x n) activations to other
    %       feature maps first (e.g. SparseDictionary.featureMaps) -- "channels" are then its outputs
    %       T.Value  (C x K) activations, descending
    %       T.Image  (C x K) index into X;  T.Row, T.Col (C x K) unit position
    %   P = TopActivations.patches(X, T, rf, channel)   (h x w x Cin x K) input patches
    %                                                  (ReceptiveField.crop)

    methods (Static)
        function T = scan(model, X, layer, opts)
            arguments
                model
                X
                layer
                opts.K (1,1) double = 9
                opts.Batch (1,1) double = 256
                opts.Transform = []
            end
            K = opts.K; N = size(X, 4);
            T = struct('Value', [], 'Image', [], 'Row', [], 'Col', []);
            for b = 1:opts.Batch:N
                k = b:min(N, b + opts.Batch - 1);
                a = double(OceanDL.Interpret.Probe.activation(model, X(:, :, :, k), layer));
                if ~isempty(opts.Transform), a = opts.Transform(a); end
                [h, w, C, n] = size(a);
                [v, pos] = max(reshape(a, h * w, C, n), [], 1);          % best position per image and channel
                v = reshape(v, C, n); pos = reshape(pos, C, n);
                [r, c] = ind2sub([h w], pos);
                V = [T.Value, v]; I = [T.Image, repmat(k, C, 1)];
                R = [T.Row, r]; Cc = [T.Col, c];
                [V, order] = sort(V, 2, 'descend');
                keep = order(:, 1:min(K, size(order, 2)));
                row = (1:C).';
                pick = @(M) M(sub2ind(size(M), repmat(row, 1, size(keep, 2)), keep));
                T.Value = V(:, 1:size(keep, 2)); T.Image = pick(I); T.Row = pick(R); T.Col = pick(Cc);
            end
        end

        function P = patches(X, T, rf, channel)
            K = size(T.Value, 2);
            P = zeros(rf.Size(1), rf.Size(2), size(X, 3), K, 'like', X);
            for j = 1:K
                P(:, :, :, j) = OceanDL.Interpret.ReceptiveField.crop(X(:, :, :, T.Image(channel, j)), rf, ...
                    T.Row(channel, j), T.Col(channel, j));
            end
        end
    end
end
