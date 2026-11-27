function l_opt = optimalCodeLength(probs, options)
% OPTIMALCODELENGTH  Compute theoretical optimal (Shannon) code lengths.
%
% l_opt(k) = -log_Base(p_k)
%
% These are the ideal (fractional) lengths. Any integer-length prefix-free
% code must use ceil(l_opt(k)) or longer, introducing redundancy.
% Huffman achieves L_avg < H(X) + 1 bit by packing these optimally.
%
% Usage:
%   l_opt = OceanMath.Information.Coding.optimalCodeLength([0.5, 0.3, 0.2])
%   l_opt = OceanMath.Information.Coding.optimalCodeLength(probs, 'Base', 3)
%
% INPUTS:
%   probs - (1 x K) or (K x 1) non-negative doubles summing to 1
%
% Name-Value Options:
%   Base (double): logarithm base. Default: 2 (bits).
%
% OUTPUTS:
%   l_opt - (1 x K) double, optimal fractional codeword lengths.
%           Entries are +Inf for zero-probability symbols.
%
% See also: OceanMath.Information.Coding.huffmanEncode
%           OceanMath.Information.Coding.CodeTable.getOptimalLengths

    arguments
        probs   (1,:) double {mustBeNonnegative}
        options.Base (1,1) double {mustBePositive} = 2
    end

    if abs(sum(probs) - 1.0) > 1e-9
        error('OceanMath:Information:Coding:optimalCodeLength:InvalidPMF', ...
            'probs must sum to 1. Got %.10f.', sum(probs));
    end

    % l_opt = -log_Base(p) = -log(p) / log(Base)
    % For p=0: l_opt = +Inf (convention: zero-prob symbols need infinite bits)
    l_opt = zeros(1, numel(probs));
    for k = 1:numel(probs)
        if probs(k) > 0
            l_opt(k) = -log(probs(k)) / log(options.Base);
        else
            l_opt(k) = Inf;
        end
    end
end
