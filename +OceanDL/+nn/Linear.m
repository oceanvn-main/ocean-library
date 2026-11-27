classdef Linear < OceanDL.nn.Module
    % LINEAR  y = W x + b   (PyTorch nn.Linear), x: (InFeatures x N [x ...]).
    %
    %   W (Out x In), b (Out x 1), initialised U(-1/sqrt(In), 1/sqrt(In))
    %   like PyTorch's default, which keeps the output variance of order 1
    %   for unit-variance inputs. Extra trailing dims (e.g. time) are folded
    %   into the batch for the product and restored afterwards.

    properties (SetAccess = protected)
        Weight
        Bias
        InFeatures (1,1) double
        OutFeatures (1,1) double
    end

    methods
        function obj = Linear(inFeatures, outFeatures, opts)
            arguments
                inFeatures (1,1) double {mustBePositive, mustBeInteger}
                outFeatures (1,1) double {mustBePositive, mustBeInteger}
                opts.Bias (1,1) logical = true
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.InFeatures = inFeatures;
            obj.OutFeatures = outFeatures;
            u = OceanDL.nn.Linear.uniform(opts.Seed);
            bound = 1 / sqrt(inFeatures);
            obj.Weight = OceanDL.nn.Parameter(cast(bound * (2*u(outFeatures, inFeatures) - 1), opts.DataType));
            if opts.Bias
                obj.Bias = OceanDL.nn.Parameter(cast(bound * (2*u(outFeatures, 1) - 1), opts.DataType));
            end
        end

        function y = forward(obj, x)
            sz = size(x);
            if numel(sz) > 2
                x = reshape(x, sz(1), []);
            end
            y = OceanDL.Autograd.F.linear(x, obj.Weight, obj.Bias);
            if numel(sz) > 2
                y = reshape(y, [obj.OutFeatures, sz(2:end)]);
            end
        end
    end

    methods (Static)
        function u = uniform(seed)
            % U(r, c) draws uniform [0,1) numbers from a seeded stream (or the global one).
            if isempty(seed)
                u = @(r, c) rand(r, c);
            else
                s = RandStream('mt19937ar', 'Seed', seed);
                u = @(r, c) rand(s, r, c);
            end
        end
    end
end
