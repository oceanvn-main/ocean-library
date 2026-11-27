classdef GroupNorm < OceanDL.nn.Module
    % GROUPNORM  Group normalisation module (PyTorch nn.GroupNorm) on
    %   (H x W x C x N). Weight (gamma) init 1, Bias (beta) init 0, both
    %   (1 x 1 x C). See Autograd.Functions.GroupNorm for the formula.

    properties (SetAccess = protected)
        Weight
        Bias
        Groups (1,1) double
        Epsilon (1,1) double
    end

    methods
        function obj = GroupNorm(numGroups, numChannels, opts)
            arguments
                numGroups (1,1) double {mustBePositive, mustBeInteger}
                numChannels (1,1) double {mustBePositive, mustBeInteger}
                opts.Epsilon (1,1) double {mustBePositive} = 1e-5
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            if mod(numChannels, numGroups) ~= 0
                error('OceanDL:nn:GroupNorm:Groups', '%d channels are not divisible into %d groups.', numChannels, numGroups);
            end
            obj.Groups = numGroups;
            obj.Epsilon = opts.Epsilon;
            obj.Weight = OceanDL.nn.Parameter(ones(1, 1, numChannels, opts.DataType));
            obj.Bias = OceanDL.nn.Parameter(zeros(1, 1, numChannels, opts.DataType));
        end

        function y = forward(obj, x)
            y = OceanDL.Autograd.F.groupNorm(x, obj.Groups, obj.Weight, obj.Bias, obj.Epsilon);
        end
    end

    methods (Static)
        function g = groupsFor(channels, preferred)
            % Largest group count <= preferred that divides the channel count
            % (32 in Stable Diffusion; small models need fewer).
            g = max(divisors(channels) .* (divisors(channels) <= preferred));
        end
    end
end
