classdef (Abstract) IQuantizer < OceanDL.nn.Module
    % IQUANTIZER  Contract for discrete bottlenecks (Strategy pattern): VQ-VAE
    %   swaps quantizers without touching encoder / decoder / prior.
    %
    %   [zq, info] = q.forward(z)   z: continuous latent in q.Layout;
    %                               zq: quantised latent (same shape), with a
    %                               gradient path to z (straight-through or
    %                               equivalent);
    %                               info.Indices     integer code per position (h x w x N | N x 1)
    %                               info.Loss        auxiliary loss Tensor (0 if none)
    %                               info.Perplexity  exp(entropy of batch code usage)
    %                               info.Usage       fraction of codes used in the batch
    %   Z = q.lookup(indices)       quantised latent (array) for given codes
    %   K = q.numCodes()            codebook size

    properties (SetAccess = protected)
        Layout (1,1) string = "SSCB"
    end

    methods (Abstract)
        [zq, info] = forward(obj, z)
        Z = lookup(obj, indices)
        K = numCodes(obj)
    end

    methods (Static)
        function [ppl, usage, counts] = usageStats(idx, K)
            % Perplexity = exp(-sum p log p) of the batch code histogram:
            % K when all codes are used equally, 1 when one code is used.
            counts = histcounts(double(gather(idx(:))), 0.5:1:K + 0.5);
            p = counts / max(1, sum(counts));
            nz = p(p > 0);
            ppl = exp(-sum(nz .* log(nz)));
            usage = nnz(counts) / K;
        end
    end
end
