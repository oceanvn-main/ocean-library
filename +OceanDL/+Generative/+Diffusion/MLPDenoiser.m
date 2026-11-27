classdef MLPDenoiser < OceanDL.nn.Module
    % MLPDENOISER  Denoiser for vector data ("CB", D x N): the minimal network
    %   for low-dimensional diffusion (toy distributions, tabular data, latent
    %   vectors) and for testing objectives against analytic optima.
    %
    %     h = [x; TimestepEmbedding(t)] -> (Linear -> SiLU) x Depth -> Linear -> D
    %   Optional class conditioning adds a label embedding (row K + 1 = NULL).

    properties (SetAccess = protected)
        TimeEmbed
        ClassEmbed
        Layers cell = {}
        NumClasses (1,1) double
    end

    methods
        function obj = MLPDenoiser(dim, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.Hidden (1,1) double = 64
                opts.Depth (1,1) double = 2
                opts.TimeDim (1,1) double = 32
                opts.NumClasses (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            seed = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            obj.TimeEmbed = OceanDL.nn.TimestepEmbedding(opts.TimeDim, 'Seed', seed(1), dt{:});
            obj.NumClasses = opts.NumClasses;
            if opts.NumClasses > 0
                obj.ClassEmbed = OceanDL.nn.Embedding(opts.NumClasses + 1, opts.TimeDim, 'Seed', seed(2), dt{:});
            end
            in = dim + opts.TimeDim;
            for l = 1:opts.Depth
                obj.Layers{end+1} = OceanDL.nn.Linear(in, opts.Hidden, 'Seed', seed(10 + l), dt{:});
                in = opts.Hidden;
            end
            obj.Layers{end+1} = OceanDL.nn.Linear(in, dim, 'Seed', seed(50), dt{:});
        end

        function y = forward(obj, x, t, labels)
            if nargin < 4, labels = []; end
            
            e = obj.TimeEmbed.forward(t);

            if ~isempty(obj.ClassEmbed)
                if isempty(labels), labels = repmat(obj.NumClasses + 1, size(x, 2), 1); end
                e = e + obj.ClassEmbed.forward(reshape(labels, [], 1));        % ids N x 1 -> E x N
            end
            h = cat(1, OceanDL.Autograd.Tensor.wrap(x), e);
            for l = 1:numel(obj.Layers) - 1
                h = OceanDL.Autograd.F.silu(obj.Layers{l}.forward(h));
            end
            y = obj.Layers{end}.forward(h);
        end
    end
end
