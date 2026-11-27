classdef ConvDecoder < OceanDL.nn.Module
    % CONVDECODER  Mirror of ConvEncoder (AutoencoderKL layout):
    %
    %     Conv3x3(LatentChannels -> Widths(end))
    %     mid:  ConvResBlock, [SpatialSelfAttention], ConvResBlock
    %     for level l = L..1:  ResPerLevel x ConvResBlock(-> Widths(l));
    %                          nearest x2 upsample + Conv3x3 if l > 1
    %     out:  GN -> SiLU -> Conv3x3(-> OutChannels)
    %
    %   (h x w x LatentChannels x N) -> (h*f x w*f x OutChannels x N), f = 2^(L-1).
    %   Outputs are unconstrained: the likelihood (VAE 'Likelihood') decides
    %   whether they are Gaussian means or Bernoulli logits.

    properties (SetAccess = protected)
        Body
        Factor (1,1) double
    end

    methods
        function obj = ConvDecoder(opts)
            arguments
                opts.LatentChannels (1,1) double = 4
                opts.Widths (1,:) double = [32 64]         % same order as the encoder
                opts.ResPerLevel (1,1) double = 1
                opts.OutChannels (1,1) double = 3
                opts.MidAttention (1,1) logical = true
                opts.Groups (1,1) double = 32
                opts.Dropout (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            seed = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            common = {'Groups', opts.Groups, 'DataType', opts.DataType};
            W = opts.Widths;
            c = W(end);
            layers = {OceanDL.nn.Conv2d(opts.LatentChannels, c, 3, 'Padding', 1, 'Seed', seed(0), 'DataType', opts.DataType)};
            layers = [layers, OceanDL.Vision.ConvEncoder.mid(c, opts, seed(50))];
            k = 100;
            for l = numel(W):-1:1
                for r = 1:opts.ResPerLevel
                    k = k + 10;
                    layers{end+1} = OceanDL.Vision.ConvResBlock(c, W(l), common{:}, 'Dropout', opts.Dropout, 'Seed', seed(k)); %#ok<AGROW>
                    c = W(l);
                end
                if l > 1
                    k = k + 10;
                    layers = [layers, {OceanDL.nn.Upsample(2), ...
                        OceanDL.nn.Conv2d(c, c, 3, 'Padding', 1, 'Seed', seed(k), 'DataType', opts.DataType)}]; %#ok<AGROW>
                end
            end
            k = k + 10;
            layers = [layers, {OceanDL.nn.GroupNorm(OceanDL.nn.GroupNorm.groupsFor(c, opts.Groups), c, 'DataType', opts.DataType), ...
                OceanDL.nn.Activation("silu"), ...
                OceanDL.nn.Conv2d(c, opts.OutChannels, 3, 'Padding', 1, 'Seed', seed(k), 'DataType', opts.DataType)}];
            obj.Body = OceanDL.nn.Sequential(layers{:});
            obj.Factor = 2 ^ (numel(W) - 1);
        end

        function x = forward(obj, z)
            x = obj.Body.forward(z);
        end
    end
end
