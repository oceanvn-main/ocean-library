classdef Conv1d < OceanDL.nn.Module
    % CONV1D  1-D convolution over time (PyTorch nn.Conv1d) on CBT sequences.
    %   x (Cin x N x T) Tensor/array -> y (Cout x N x To) Tensor.
    %
    %   Weight (Cout x Cin/Groups x K) -- PyTorch order, so torch weights load
    %   without permutation; Bias (Cout x 1).
    %   Init (PyTorch default, kaiming_uniform with a = sqrt(5)):
    %     W, b ~ U(-1/sqrt(fanIn), 1/sqrt(fanIn)),   fanIn = (Cin/Groups) * K.
    %   Groups: 1 (dense) or Cin (depthwise); any divisor of Cin and Cout works.
    %
    %   OUTPUT LENGTH (contract §3, PyTorch formula), per example:
    %     To = floor((T + 2p - d(K-1) - 1)/s) + 1     (clamped at 0; T_n = 0 -> 0,
    %     a deliberate exception: an empty sequence has no valid output frames)
    %   Use outputLength(lengths) to propagate the valid lengths of a padded
    %   batch. Outputs t > To_n of example n read padding / steps past T_n:
    %   they are NOT valid and must be masked (OceanDL.Layout.SequenceMask).
    %
    % EXAMPLE:
    %   conv = OceanDL.nn.Conv1d(80, 128, 5, 'Stride', 2, 'Padding', 2, 'Seed', 1);
    %   y = conv.forward(X);                      % X: 80 x N x T
    %   lens = conv.outputLength(lens);
    %
    % See also: OceanDL.Autograd.Functions.Conv1d, OceanDL.Layout.SequenceMask

    properties (SetAccess = protected)
        Weight
        Bias
        InChannels (1,1) double
        OutChannels (1,1) double
        KernelSize (1,1) double
        Stride (1,1) double
        Padding (1,1) double
        Dilation (1,1) double
        Groups (1,1) double
    end

    methods
        function obj = Conv1d(inChannels, outChannels, kernelSize, opts)
            arguments
                inChannels (1,1) double {mustBePositive, mustBeInteger}
                outChannels (1,1) double {mustBePositive, mustBeInteger}
                kernelSize (1,1) double {mustBePositive, mustBeInteger}
                opts.Stride (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Padding (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.Dilation (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Groups (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Bias (1,1) logical = true
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            if mod(inChannels, opts.Groups) ~= 0 || mod(outChannels, opts.Groups) ~= 0
                error('OceanDL:nn:Conv1d:BadGroups', ...
                    'Groups = %d must divide both inChannels (%d) and outChannels (%d).', ...
                    opts.Groups, inChannels, outChannels);
            end
            obj.InChannels = inChannels;
            obj.OutChannels = outChannels;
            obj.KernelSize = kernelSize;
            obj.Stride = opts.Stride;
            obj.Padding = opts.Padding;
            obj.Dilation = opts.Dilation;
            obj.Groups = opts.Groups;
            cig = inChannels / opts.Groups;
            u = OceanDL.nn.Linear.uniform(opts.Seed);
            bound = 1 / sqrt(cig * kernelSize);
            W = bound * (2 * reshape(u(outChannels * cig * kernelSize, 1), outChannels, cig, kernelSize) - 1);
            obj.Weight = OceanDL.nn.Parameter(cast(W, opts.DataType));
            if opts.Bias
                obj.Bias = OceanDL.nn.Parameter(cast(bound * (2 * u(outChannels, 1) - 1), opts.DataType));
            end
        end

        function y = forward(obj, x)
            % FORWARD  x (Cin x N x T) -> y (Cout x N x To).
            if size(x, 1) ~= obj.InChannels
                error('OceanDL:nn:Conv1d:ChannelMismatch', ...
                    'Expected %d input channels (CBT layout), got %d.', obj.InChannels, size(x, 1));
            end
            op = OceanDL.Autograd.Functions.Conv1d('Stride', obj.Stride, 'Padding', obj.Padding, ...
                'Dilation', obj.Dilation, 'Groups', obj.Groups);
            if isempty(obj.Bias)
                y = op.apply(x, obj.Weight);
            else
                y = op.apply(x, obj.Weight, obj.Bias);
            end
        end

        function lens = outputLength(obj, lens)
            % OUTPUTLENGTH  Valid output length for each valid input length.
            arguments
                obj
                lens double {mustBeNonnegative, mustBeInteger}
            end
            empty = lens == 0;
            lens = OceanDL.Autograd.Functions.Conv1d.outLength(lens, obj.KernelSize, ...
                obj.Stride, obj.Padding, obj.Dilation);
            lens(empty) = 0;                           % an empty sequence stays empty (its padding-only frames are not valid)
        end
    end
end
