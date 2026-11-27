function S = compareCoding(symbols, probs, options)
% COMPARECODING  Compare Huffman vs theoretical (optimal) coding metrics.
%
% Encodes a discrete source with Huffman and reports key metrics alongside
% theoretical bounds from Shannon's Source Coding Theorem.
%
% Usage:
%   S = OceanMath.Information.Coding.compareCoding({'a','b','c'}, [0.5,0.3,0.2])
%   S = OceanMath.Information.Coding.compareCoding(symbols, probs, 'Verbose', true)
%
%   % Access results:
%   S.huffman_table         % CodeTable from HuffmanCoder
%   S.entropy_bits          % H(X) in bits
%   S.huffman_L             % Huffman expected length
%   S.huffman_redundancy    % L_avg - H(X)
%   S.huffman_efficiency    % H(X) / L_avg
%   S.kraft_sum             % Kraft inequality check
%   S.optimal_lengths       % Theoretical -log2(p_k) per symbol
%   S.shannon_lower_bound   % H(X) (lower bound on L_avg)
%   S.shannon_upper_bound   % H(X) + 1 (upper bound for Huffman)
%
% INPUTS:
%   symbols - (K x 1) or (1 x K) cell of char/string labels
%   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1
%
% Name-Value Options:
%   Base    (double):  code base. Default: 2.
%   Verbose (logical): print summary table. Default: false.
%
% OUTPUTS:
%   S - struct with all comparison fields
%
% See also: OceanMath.Information.Coding.huffmanEncode
%           OceanMath.Information.Coding.optimalCodeLength
%           OceanMath.Information.Coding.CodeTable

    arguments
        symbols (:,1) cell
        probs   (1,:) double {mustBeNonnegative}
        options.Base    (1,1) double  {mustBePositive, mustBeInteger} = 2
        options.Verbose (1,1) logical = false
    end

    % --- Huffman coding ---
    coder = OceanMath.Information.Coding.HuffmanCoder('Base', options.Base);
    table = coder.encode(symbols, probs);

    % --- Theoretical bounds ---
    H    = table.entropyBits();
    L    = table.expectedLength();
    R    = table.redundancy();
    eta  = table.efficiency();
    K    = table.kraftSum();
    l_opt = OceanMath.Information.Coding.optimalCodeLength( ...
                probs, 'Base', options.Base);

    % --- Shannon bounds ---
    H_base = H / log2(options.Base);   % entropy in configured base units
    lb = H_base;          % lower bound: H(X)
    ub = H_base + 1;      % upper bound: H(X) + 1  (Huffman guarantee)

    % --- Assemble output struct ---
    S = struct();
    S.symbols              = symbols;
    S.probs                = probs;
    S.base                 = options.Base;
    S.huffman_table        = table;
    S.entropy_bits         = H;
    S.entropy_base         = H_base;
    S.huffman_L            = L;
    S.huffman_redundancy   = R;
    S.huffman_efficiency   = eta;
    S.kraft_sum            = K;
    S.kraft_complete       = table.isComplete();
    S.optimal_lengths      = l_opt;
    S.huffman_lengths      = table.getLengths();
    S.shannon_lower_bound  = lb;
    S.shannon_upper_bound  = ub;
    S.theorem_satisfied    = (L >= lb - 1e-9) && (L < ub + 1e-9);

    if options.Verbose
        fprintf('\n=== compareCoding  [Base-%d, K=%d symbols] ===\n', ...
            options.Base, numel(symbols));
        fprintf('  H(X)                = %.4f bits\n', H);
        fprintf('  H(X)/log2(base)     = %.4f base units\n', H_base);
        fprintf('  Huffman L_avg       = %.4f code symbols/symbol\n', L);
        fprintf('  Redundancy R        = %.4f\n', R);
        fprintf('  Efficiency eta      = %.4f (%.1f%%)\n', eta, eta*100);
        fprintf('  Kraft sum K         = %.6f (complete: %s)\n', K, mat2str(table.isComplete()));
        fprintf('  Shannon bounds      = [%.4f, %.4f)\n', lb, ub);
        fprintf('  Theorem satisfied   = %s\n', mat2str(S.theorem_satisfied));
        fprintf('\n');
    end
end
