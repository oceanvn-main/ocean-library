classdef ConvEncoder < OceanDL.nn.Module
    % CONVENCODER  Convolutional encoder of modern autoencoders (Stable
    %   Diffusion AutoencoderKL layout, sized by options):
    %
    %     Conv3x3(InChannels -> Widths(1))
    %     for level l = 1..L:  ResPerLevel x ConvResBlock(-> Widths(l));
    %                          stride-2 Conv3x3 downsample if l < L
    %     mid:  ConvResBlock, [SpatialSelfAttention], ConvResBlock
    %     out:  GN -> SiLU -> Conv3x3(-> OutChannels)
    %
    %   (H x W x InChannels x N) -> (H/f x W/f x OutChannels x N), f = 2^(L-1).
    %   A VAE uses OutChannels = 2 * latent channels (mean and log-variance);
    %   a VQ-VAE uses the code dimension. H and W must be divisible by f.

    properties (SetAccess = protected)
        Body                % Sequential
        Factor (1,1) double
    end

    methods
        function obj = ConvEncoder(opts)
            arguments
                opts.InChannels (1,1) double = 3
                opts.Widths (1,:) double = [32 64]
                opts.ResPerLevel (1,1) double = 1
                opts.OutChannels (1,1) double = 8
                opts.MidAttention (1,1) logical = true
                opts.Groups (1,1) double = 32
                opts.Dropout (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            k = 0;
            seed = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);   % k passed explicitly (anonymous functions capture by value)
            common = {'Groups', opts.Groups, 'DataType', opts.DataType};
            W = opts.Widths;
            layers = {OceanDL.nn.Conv2d(opts.InChannels, W(1), 3, 'Padding', 1, 'Seed', seed(0), 'DataType', opts.DataType)};
            c = W(1);
            for l = 1:numel(W)
                for r = 1:opts.ResPerLevel
                    k = k + 10;
                    layers{end+1} = OceanDL.Vision.ConvResBlock(c, W(l), common{:}, 'Dropout', opts.Dropout, 'Seed', seed(k)); %#ok<AGROW>
                    c = W(l);
                end
                if l < numel(W)
                    k = k + 10;
                    layers{end+1} = OceanDL.nn.Conv2d(c, c, 3, 'Stride', 2, 'Padding', 1, 'Seed', seed(k), 'DataType', opts.DataType); %#ok<AGROW>
                end
            end
            layers = [layers, OceanDL.Vision.ConvEncoder.mid(c, opts, seed(k + 50))];
            k = k + 100;
            layers = [layers, {OceanDL.nn.GroupNorm(OceanDL.nn.GroupNorm.groupsFor(c, opts.Groups), c, 'DataType', opts.DataType), ...
                OceanDL.nn.Activation("silu"), ...
                OceanDL.nn.Conv2d(c, opts.OutChannels, 3, 'Padding', 1, 'Seed', seed(k), 'DataType', opts.DataType)}];
            obj.Body = OceanDL.nn.Sequential(layers{:});
            obj.Factor = 2 ^ (numel(W) - 1);
        end

        function z = forward(obj, x)
            [H, W] = size(x, 1:2);
            if mod(H, obj.Factor) || mod(W, obj.Factor)
                error('OceanDL:Vision:ConvEncoder:Size', 'Image %dx%d is not divisible by the downsampling factor %d.', H, W, obj.Factor);
            end
            z = obj.Body.forward(x);
        end
    end

    methods (Static)
        function layers = mid(c, opts, s0)
            % Shared by encoder and decoder: ResBlock, optional attention, ResBlock.
            common = {'Groups', opts.Groups, 'DataType', opts.DataType};
            s = @(j) OceanDL.Vision.ConvEncoder.offset(s0, 1000 + j);
            layers = {OceanDL.Vision.ConvResBlock(c, c, common{:}, 'Dropout', opts.Dropout, 'Seed', s(1))};
            if opts.MidAttention
                layers{end+1} = OceanDL.Vision.SpatialSelfAttention(c, common{:}, 'Seed', s(2));
            end
            layers{end+1} = OceanDL.Vision.ConvResBlock(c, c, common{:}, 'Dropout', opts.Dropout, 'Seed', s(3));
        end

        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
