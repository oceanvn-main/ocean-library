classdef ConvResBlock < OceanDL.nn.Module
    % CONVRESBLOCK  Pre-activation residual block of modern conv autoencoders
    %   (Stable Diffusion AutoencoderKL / DDPM ResnetBlock) on (H x W x C x N):
    %
    %     h = Conv3x3( SiLU( GN(x) ) )
    %     h = Conv3x3( Dropout( SiLU( GN(h) ) ) )
    %     y = skip(x) + h,     skip = identity  |  Conv1x1 when Cin ~= Cout
    %
    %   GroupNorm (not BatchNorm): independent of the batch, so encoder and
    %   decoder behave identically in training and generation.
    %
    %   CONDITIONING ('EmbedDim' > 0; diffusion U-Nets): forward(x, emb) with
    %   emb (EmbedDim x N), p = Linear(SiLU(emb)) per sample, applied after GN2:
    %     'ScaleShift' true  (ADM, Dhariwal & Nichol 2021, FiLM):
    %                        h = GN(h) .* (1 + scale) + shift,  [scale; shift] = p
    %     'ScaleShift' false (DDPM, Ho 2020):  h = GN(h + p)

    properties (SetAccess = protected)
        Norm1
        Conv1
        Norm2
        Drop
        Conv2
        Skip            % [] (identity) or Conv2d 1x1
        EmbProj         % [] or Linear(EmbedDim -> 2 Cout | Cout)
        ScaleShift (1,1) logical = true
    end

    methods
        function obj = ConvResBlock(inChannels, outChannels, opts)
            arguments
                inChannels (1,1) double {mustBePositive, mustBeInteger}
                outChannels (1,1) double {mustBePositive, mustBeInteger}
                opts.Groups (1,1) double {mustBePositive, mustBeInteger} = 32
                opts.Dropout (1,1) double = 0
                opts.EmbedDim (1,1) double = 0
                opts.ScaleShift (1,1) logical = true
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.ConvResBlock.offset(opts.Seed, k);
            gn = @(c) OceanDL.nn.GroupNorm(OceanDL.nn.GroupNorm.groupsFor(c, opts.Groups), c, 'DataType', opts.DataType);
            obj.Norm1 = gn(inChannels);
            obj.Conv1 = OceanDL.nn.Conv2d(inChannels, outChannels, 3, 'Padding', 1, 'Seed', s(1), 'DataType', opts.DataType);
            obj.Norm2 = gn(outChannels);
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
            obj.Conv2 = OceanDL.nn.Conv2d(outChannels, outChannels, 3, 'Padding', 1, 'Seed', s(2), 'DataType', opts.DataType);
            if opts.EmbedDim > 0
                obj.ScaleShift = opts.ScaleShift;
                obj.EmbProj = OceanDL.nn.Linear(opts.EmbedDim, outChannels * (1 + opts.ScaleShift), ...
                    'Seed', s(4), 'DataType', opts.DataType);
            end
            if inChannels ~= outChannels
                obj.Skip = OceanDL.nn.Conv2d(inChannels, outChannels, 1, 'Seed', s(3), 'DataType', opts.DataType);
            end
        end

        function y = forward(obj, x, emb)
            F = OceanDL.Autograd.F;
            h = obj.Conv1.forward(F.silu(obj.Norm1.forward(x)));
            if nargin < 3 || isempty(obj.EmbProj)
                h = obj.Norm2.forward(h);
            else
                C = size(h, 3);
                p = obj.EmbProj.forward(F.silu(emb));                      % (C or 2C) x N
                p = reshape(p, 1, 1, size(p, 1), size(p, 2));
                if obj.ScaleShift
                    h = obj.Norm2.forward(h) .* (1 + p(:, :, 1:C, :)) + p(:, :, C+1:end, :);
                else
                    h = obj.Norm2.forward(h + p);
                end
            end
            h = obj.Conv2.forward(obj.Drop.forward(F.silu(h)));
            if isempty(obj.Skip), y = h + x; else, y = h + obj.Skip.forward(x); end
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
