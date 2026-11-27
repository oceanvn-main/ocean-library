classdef TimestepEmbedding < OceanDL.nn.Module
    % TIMESTEPEMBEDDING  Diffusion-time conditioning vector (DDPM / ADM / DiT):
    %
    %   sinusoid:  f_k = exp(-log(MaxPeriod) k / H),  k = 0..H-1,  H = FreqDim / 2
    %              phi(t) = [cos(Scale t f); sin(Scale t f)]          (FreqDim x N)
    %   MLP:       e = Linear( SiLU( Linear(phi) ) )                  (Dim x N)
    %
    %   t is continuous (a 1 x N vector); Scale = 1000 maps t in [0, 1] onto
    %   the DDPM step range the sinusoid was designed for. The same features
    %   as diffusers' get_timestep_embedding(flip_sin_to_cos = True).

    properties (SetAccess = protected)
        Fc1
        Fc2
        FreqDim (1,1) double
        MaxPeriod (1,1) double
        Scale (1,1) double
    end

    methods
        function obj = TimestepEmbedding(dim, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.FreqDim (1,1) double = 0            % 0 = dim
                opts.MaxPeriod (1,1) double = 10000
                opts.Scale (1,1) double = 1000
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            fd = opts.FreqDim; if fd == 0, fd = dim; end
            s = @(k) OceanDL.nn.TimestepEmbedding.offset(opts.Seed, k);
            obj.FreqDim = fd; obj.MaxPeriod = opts.MaxPeriod; obj.Scale = opts.Scale;
            obj.Fc1 = OceanDL.nn.Linear(fd, dim, 'Seed', s(1), 'DataType', opts.DataType);
            obj.Fc2 = OceanDL.nn.Linear(dim, dim, 'Seed', s(2), 'DataType', opts.DataType);
        end

        function e = forward(obj, t)
            like = obj.Fc1.Weight.Data;
            phi = OceanDL.nn.TimestepEmbedding.sinusoid(cast(reshape(t, 1, []), 'like', like), obj.FreqDim, obj.MaxPeriod, obj.Scale);
            e = obj.Fc2.forward(OceanDL.Autograd.F.silu(obj.Fc1.forward(phi)));
        end
    end

    methods (Static)
        function phi = sinusoid(t, dim, maxPeriod, scale)
            half = floor(dim / 2);
            f = exp(-log(maxPeriod) * (0:half - 1).' / half);
            arg = (scale * t) .* f;                                     % half x N
            phi = [cos(arg); sin(arg)];
            if mod(dim, 2), phi = [phi; zeros(1, size(t, 2), 'like', phi)]; end
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
