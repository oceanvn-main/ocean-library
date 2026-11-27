classdef BottleneckBlock < OceanDL.nn.Module
    % BOTTLENECKBLOCK  VDVAE residual block (Child 2021): the 3x3 convs run on
    %   a narrow Ratio * Cout channel bottleneck, so a block costs ~10x less
    %   than a full-width ConvResBlock -- which is what makes very deep
    %   hierarchies (dozens of latent layers) affordable.
    %
    %     h = Conv1x1(GELU(x))  -> Cmid         Cmid = round(Ratio * Cout)
    %     h = Conv3x3(GELU(h))  -> Cmid
    %     h = Conv3x3(GELU(h))  -> Cmid
    %     h = Conv1x1(GELU(h))  -> Cout
    %     y = skip(x) + h,      skip = identity | Conv1x1 when Cin ~= Cout
    %
    %   No normalisation (as VDVAE). 'OutScale' multiplies the initial weights
    %   of the last conv (VDVAE uses 1/sqrt(number of blocks) for stability).

    properties (SetAccess = protected)
        Conv1
        Conv2
        Conv3
        Conv4
        Skip
    end

    methods
        function obj = BottleneckBlock(inChannels, outChannels, opts)
            arguments
                inChannels (1,1) double {mustBePositive, mustBeInteger}
                outChannels (1,1) double {mustBePositive, mustBeInteger}
                opts.Ratio (1,1) double {mustBePositive} = 0.25
                opts.OutScale (1,1) double = 1
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            mid = max(1, round(opts.Ratio * outChannels));
            obj.Conv1 = OceanDL.nn.Conv2d(inChannels, mid, 1, 'Seed', s(1), dt{:});
            obj.Conv2 = OceanDL.nn.Conv2d(mid, mid, 3, 'Padding', 1, 'Seed', s(2), dt{:});
            obj.Conv3 = OceanDL.nn.Conv2d(mid, mid, 3, 'Padding', 1, 'Seed', s(3), dt{:});
            obj.Conv4 = OceanDL.nn.Conv2d(mid, outChannels, 1, 'Seed', s(4), dt{:});
            obj.Conv4.Weight.Data = opts.OutScale * obj.Conv4.Weight.Data;
            if inChannels ~= outChannels
                obj.Skip = OceanDL.nn.Conv2d(inChannels, outChannels, 1, 'Seed', s(5), dt{:});
            end
        end

        function y = forward(obj, x)
            F = OceanDL.Autograd.F;
            h = obj.Conv1.forward(F.gelu(x));
            h = obj.Conv2.forward(F.gelu(h));
            h = obj.Conv3.forward(F.gelu(h));
            h = obj.Conv4.forward(F.gelu(h));
            if isempty(obj.Skip), y = h + x; else, y = h + obj.Skip.forward(x); end
        end
    end
end
