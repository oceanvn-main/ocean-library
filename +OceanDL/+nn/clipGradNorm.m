function totalNorm = clipGradNorm(params, maxNorm)
% CLIPGRADNORM  Rescale gradients so their GLOBAL L2 norm is at most maxNorm
%   (Goodfellow et al. 2016, Sec. 10.11.1; PyTorch clip_grad_norm_).
%
%   ||g|| = sqrt( sum over all parameters of sum(g.^2) )
%   if ||g|| > maxNorm:   g <- g * maxNorm / ||g||   for every parameter
%
%   Scaling ALL gradients by the same factor keeps the update DIRECTION and
%   only shortens the step -- the standard fix for exploding gradients in
%   recurrent networks. Returns the norm BEFORE clipping (for monitoring).
arguments
    params cell
    maxNorm (1,1) double {mustBePositive}
end
sq = 0;
for k = 1:numel(params)
    g = params{k}.Grad;
    if ~isempty(g)
        sq = sq + sum(double(g(:)).^2);
    end
end
totalNorm = sqrt(sq);
if totalNorm > maxNorm
    scale = maxNorm / (totalNorm + 1e-6);
    for k = 1:numel(params)
        if ~isempty(params{k}.Grad)
            params{k}.Grad = params{k}.Grad * scale;
        end
    end
end
end
