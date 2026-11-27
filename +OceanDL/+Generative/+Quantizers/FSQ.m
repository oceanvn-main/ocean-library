classdef FSQ < OceanDL.Generative.Quantizers.IQuantizer
    % FSQ  Finite Scalar Quantization (Mentzer et al. 2023, "VQ-VAE made
    %   simple"): no codebook, no auxiliary loss, no collapse.
    %
    %   The latent has D = numel(Levels) channels. Channel i is bounded and
    %   rounded to one of Levels(i) values:
    %     h_i  = floor(L_i / 2),   half = (L_i - 1)(1 + 1e-3) / 2
    %     off  = 0.5 if L_i even else 0,   shift = atanh(off / half)
    %     b    = tanh(z + shift) * half - off                (bounded)
    %     q    = round(b)  with straight-through gradient  b + stopgrad(round(b) - b)
    %     zq   = q / h_i                                    (in [-1, 1])
    %   The implicit codebook is the grid of prod(Levels) points; the index of
    %   a code is its mixed-radix number: sum_i (q_i + h_i) * prod_{j<i} L_j (+ 1).
    %   Every grid point is reachable, so usage is typically ~100%.

    properties (SetAccess = protected)
        Levels (1,:) double
    end

    methods
        function obj = FSQ(levels, opts)
            arguments
                levels (1,:) double {mustBeInteger, mustBeGreaterThan(levels, 1)}
                opts.Layout (1,1) string {mustBeMember(opts.Layout, ["SSCB", "CB"])} = "SSCB"
            end
            obj.Levels = levels;
            obj.Layout = opts.Layout;
        end

        function K = numCodes(obj)
            K = prod(obj.Levels);
        end

        function [zq, info] = forward(obj, z)
            L = OceanDL.Generative.Latent;
            [Zc, shape] = L.toColumns(OceanDL.Autograd.Tensor.wrap(z), obj.Layout);   % D x M
            if size(Zc, 1) ~= numel(obj.Levels)
                error('OceanDL:Generative:FSQ:Channels', 'Latent has %d channels; FSQ expects %d (numel(Levels)).', ...
                    size(Zc, 1), numel(obj.Levels));
            end
            [half, off, shift, h] = obj.constants(Zc.Data);
            b = tanh(Zc + shift) .* half - off;
            q = b + (round(b.Data) - b.Data);                       % straight-through round
            out = q ./ h;
            zq = L.fromColumns(out, shape, obj.Layout);
            idx = obj.codesToIndices(round(b.Data), h);
            [ppl, usage] = OceanDL.Generative.Quantizers.IQuantizer.usageStats(idx, obj.numCodes());
            info = struct('Indices', reshape(idx, L.indexShape(shape, obj.Layout)), 'Loss', 0, ...
                'Perplexity', ppl, 'Usage', usage, 'Restarted', 0);
        end

        function Z = lookup(obj, indices)
            [~, ~, ~, h] = obj.constants(0);
            radix = cumprod([1 obj.Levels(1:end-1)]).';
            r = double(indices(:).') - 1;
            digits = mod(floor(r ./ radix), obj.Levels.');          % D x M, each in 0..L-1
            cols = (digits - h) ./ h;
            shape = OceanDL.Generative.Latent.shapeFromIndices(indices, numel(obj.Levels), obj.Layout);
            Z = OceanDL.Generative.Latent.fromColumns(cols, shape, obj.Layout);
        end
    end

    methods (Access = private)
        function [half, off, shift, h] = constants(obj, like)
            Lv = obj.Levels(:);
            half = (Lv - 1) * (1 + 1e-3) / 2;
            off = 0.5 * (mod(Lv, 2) == 0);
            shift = atanh(off ./ half);
            h = floor(Lv / 2);
            half = cast(half, 'like', like); off = cast(off, 'like', like);
            shift = cast(shift, 'like', like); h = cast(h, 'like', like);
        end

        function idx = codesToIndices(obj, q, h)
            radix = cumprod([1 obj.Levels(1:end-1)]);
            idx = radix * double(gather(q + h)) + 1;                % 1 x M, 1..prod(Levels)
        end
    end
end
