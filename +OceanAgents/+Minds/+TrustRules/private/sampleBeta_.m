function p = sampleBeta_(a, b, stream)
% SAMPLEBETA_  One draw p ~ Beta(a, b) using ONLY the given RandStream (A2 contract section 4.1, section 10).
%
%   p = x / (x + y),   x ~ Gamma(a, 1),  y ~ Gamma(b, 1)      (x drawn first, then y)
%
% MATLAB's randg/betarnd cannot take an explicit stream and the global stream is forbidden, so the gamma variates
% come from the Marsaglia-Tsang (2000) squeeze/rejection method below, which needs only randn(stream) and
% rand(stream).
%   Marsaglia, G. & Tsang, W. W. (2000). A simple method for generating gamma variables.
%   ACM Transactions on Mathematical Software 26(3), 363-372.
x = gammaDraw_(a, stream);
y = gammaDraw_(b, stream);
if x + y > 0
    p = x / (x + y);
else
    % Both gammas underflowed to 0 (possible only for tiny shapes, via U^(1/a)); fall back to the Beta mean
    % rather than returning 0/0. The draws were still consumed, so the stream position is unaffected.
    p = a / (a + b);
end
end

function g = gammaDraw_(a, stream)
% GAMMADRAW_  x ~ Gamma(a, 1) by Marsaglia-Tsang.
if a < 1
    % Shape boost (Marsaglia & Tsang 2000, sec. 6): if X ~ Gamma(a+1) and U ~ U(0,1) then X * U^(1/a) ~ Gamma(a).
    g = gammaDraw_(a + 1, stream) * rand(stream) ^ (1 / a);
    return
end
d = a - 1/3;
c = 1 / sqrt(9 * d);
while true
    % Propose v = (1 + c x)^3 with x ~ N(0,1); only v > 0 is admissible.
    v = -1;
    while v <= 0
        x = randn(stream);
        v = (1 + c * x) ^ 3;
    end
    u = rand(stream);
    % Squeeze test (cheap, accepts ~98% of proposals), then the exact log acceptance test.
    if u < 1 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1 - v + log(v))
        g = d * v;
        return
    end
end
end
