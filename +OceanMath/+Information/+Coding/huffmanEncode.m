function table = huffmanEncode(symbols, probs, options)
% HUFFMANENCODE  One-liner shortcut for Huffman source coding.
%
% Creates a HuffmanCoder, encodes the distribution, and returns a CodeTable.
% Equivalent to: HuffmanCoder('Base',B).encode(symbols, probs).
%
% Usage:
%   table = OceanMath.Information.Coding.huffmanEncode({'a','b','c'}, [0.5,0.3,0.2])
%   table = OceanMath.Information.Coding.huffmanEncode(symbols, probs, 'Base', 3)
%   table.printSummary()
%
% INPUTS:
%   symbols - (K x 1) or (1 x K) cell of char/string labels
%   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1
%
% Name-Value Options:
%   Base (double): code alphabet size. Default: 2 (binary).
%
% OUTPUTS:
%   table - OceanMath.Information.Coding.CodeTable
%
% See also: OceanMath.Information.Coding.HuffmanCoder
%           OceanMath.Information.Coding.CodeTable

    arguments
        symbols (:,1) cell
        probs   (1,:) double {mustBeNonnegative}
        options.Base (1,1) double {mustBePositive, mustBeInteger} = 2
    end

    coder = OceanMath.Information.Coding.HuffmanCoder('Base', options.Base);
    table = coder.encode(symbols, probs);
end
