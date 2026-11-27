function result = isValidHuffmanCode(codewords, options)
% ISVALIDHUFFMANCODE  Check if a set of codewords can be a Huffman code.
%
% A valid binary Huffman code satisfies THREE necessary conditions:
%
%   1. PREFIX-FREE: No codeword is a prefix of another.
%      (Equivalently: uniquely decodable without end-of-word markers)
%
%   2. COMPLETE (tight Kraft): sum(Base^(-l_k)) == 1.
%      An incomplete code (Kraft < 1) could always be shortened,
%      contradicting Huffman optimality.
%
%   3. SIBLING PROPERTY: Every codeword at the maximum depth has a
%      sibling — another codeword sharing the same (l-1)-length prefix,
%      differing only in the last symbol. For binary codes this means
%      the deepest codewords come in pairs.
%
% These conditions are NECESSARY but not sufficient for all probability
% assignments. A code failing ANY one cannot be a Huffman code for ANY
% probability assignment.
%
% Usage:
%   result = OceanMath.Information.Coding.isValidHuffmanCode({'0','10','11'})
%   result = OceanMath.Information.Coding.isValidHuffmanCode({'00','01','10','110'})
%   result = OceanMath.Information.Coding.isValidHuffmanCode(codewords, 'Base', 3)
%   result = OceanMath.Information.Coding.isValidHuffmanCode(codewords, 'Verbose', true)
%
% INPUTS:
%   codewords - (K x 1) or (1 x K) cell of char strings (e.g. '0','10','11')
%               or a single string like '0 10 11' (space-separated, auto-parsed)
%
% Name-Value Options:
%   Base    (double):  code alphabet size. Default: 2.
%   Verbose (logical): print detailed diagnostics. Default: false.
%
% OUTPUTS:
%   result - struct with fields:
%     .isValid          logical — passes all three checks
%     .isPrefixFree     logical — check 1
%     .isComplete       logical — check 2 (Kraft == 1)
%     .hasSiblingProp   logical — check 3
%     .kraftSum         double  — sum(Base^(-l_k))
%     .lengths          double  — (1 x K) codeword lengths
%     .maxDepth         double  — longest codeword length
%     .violations       cell    — description of each failing check
%     .codewords        cell    — the input codewords (normalised)
%
% Examples:
%   % Problem 5.6(a): {0, 10, 11} — valid Huffman code
%   r = OceanMath.Information.Coding.isValidHuffmanCode({'0','10','11'}, 'Verbose', true)
%
%   % Problem 5.6(b): {00, 01, 10, 110} — Kraft < 1, no sibling for 110
%   r = OceanMath.Information.Coding.isValidHuffmanCode({'00','01','10','110'}, 'Verbose', true)
%
%   % Problem 5.6(c): {01, 10} — Kraft < 1, no siblings share root prefix
%   r = OceanMath.Information.Coding.isValidHuffmanCode({'01','10'}, 'Verbose', true)
%
% See also: OceanMath.Information.Coding.CodeTable
%           OceanMath.Information.Coding.HuffmanCoder

    arguments
        codewords
        options.Base    (1,1) double  {mustBePositive, mustBeInteger} = 2
        options.Verbose (1,1) logical = false
    end

    Base = options.Base;

    % ---- Normalise input -------------------------------------------- %
    if ischar(codewords) || isstring(codewords)
        % Space-separated string: '0 10 11' → {'0','10','11'}
        parts     = strsplit(strtrim(char(codewords)));
        codewords = parts(:);
    end
    if ~iscell(codewords)
        error('OceanMath:Information:Coding:isValidHuffmanCode:BadInput', ...
            'codewords must be a cell array or space-separated string.');
    end
    codewords = codewords(:);   % ensure column

    K       = numel(codewords);
    lengths = cellfun(@numel, codewords);   % (K x 1)
    maxLen  = max(lengths);

    % Pre-validate: all characters must be in {'0'..'Base-1'}
    validChars = arrayfun(@(d) char('0'+d), 0:Base-1, 'UniformOutput', false);
    validSet   = [validChars{:}];
    for ki = 1:K
        cw = codewords{ki};
        if ~all(ismember(cw, validSet))
            error('OceanMath:Information:Coding:isValidHuffmanCode:BadCodeword', ...
                'Codeword "%s" contains characters outside {%s} for Base-%d.', ...
                cw, validSet, Base);
        end
    end

    violations = {};

    % ================================================================== %
    % CHECK 1: Prefix-free
    % ================================================================== %
    prefixViolations = {};
    for i = 1:K
        for j = 1:K
            if i == j, continue; end
            ci = codewords{i};
            cj = codewords{j};
            li = lengths(i);
            lj = lengths(j);
            if li <= lj && strcmp(cj(1:li), ci)
                prefixViolations{end+1} = sprintf('"%s" is a prefix of "%s"', ci, cj); %#ok<AGROW>
            end
        end
    end
    % Remove duplicates
    prefixViolations = unique(prefixViolations);
    isPrefixFree = isempty(prefixViolations);

    if ~isPrefixFree
        violations{end+1} = sprintf('NOT prefix-free: %s', strjoin(prefixViolations, '; '));
    end

    % ================================================================== %
    % CHECK 2: Tight Kraft inequality (completeness)
    % ================================================================== %
    K_kraft  = sum(Base .^ (-double(lengths)));
    isComplete_ = abs(K_kraft - 1.0) < 1e-9;

    if ~isComplete_
        if K_kraft < 1.0
            violations{end+1} = sprintf( ...
                'Kraft sum = %.6f < 1 (INCOMPLETE). A shorter code exists; Huffman would not produce this.', ...
                K_kraft);
        else
            violations{end+1} = sprintf( ...
                'Kraft sum = %.6f > 1 (OVER-COMPLETE). Not uniquely decodable.', ...
                K_kraft);
        end
    end

    % ================================================================== %
    % CHECK 3: Sibling property
    % ================================================================== %
    % For binary: every codeword at max depth must have its sibling
    % (same prefix, last bit flipped) also in the set.
    % For k-ary: all k siblings (same prefix, all k last symbols) present.
    %
    % Implementation: group max-depth codewords by their (length-1) prefix.
    % Each group must have exactly Base members with labels 0..Base-1.
    siblingViolations = {};

    % Collect all codewords at max depth
    deepCodes  = codewords(lengths == maxLen);
    nDeep      = numel(deepCodes);

    if nDeep > 0 && maxLen > 0
        % Group by parent prefix (first maxLen-1 characters)
        prefixMap = containers.Map('KeyType','char','ValueType','any');
        for di = 1:nDeep
            cw     = deepCodes{di};
            parent = cw(1:end-1);   % prefix = all but last symbol
            lastSym = cw(end);
            if prefixMap.isKey(parent)
                prefixMap(parent) = [prefixMap(parent), {lastSym}];
            else
                prefixMap(parent) = {lastSym};
            end
        end

        % Each parent must have exactly Base children labelled '0'..'Base-1'
        parents = keys(prefixMap);
        for pi = 1:numel(parents)
            par      = parents{pi};
            children = prefixMap(par);
            % Check: we have all symbols '0','1',...,'Base-1'
            for d = 0:Base-1
                sym = char('0' + d);
                if ~any(strcmp(children, sym))
                    siblingViolations{end+1} = sprintf( ...
                        'Codeword "%s%s" (sibling of depth-%d group) is missing', ...
                        par, sym, maxLen); %#ok<AGROW>
                end
            end
            % Check: no duplicates or extra children
            if numel(children) ~= Base
                siblingViolations{end+1} = sprintf( ...
                    'Parent prefix "%s" has %d children at max depth; expected %d', ...
                    par, numel(children), Base); %#ok<AGROW>
            end
        end
    end

    siblingViolations = unique(siblingViolations);
    hasSiblingProp    = isempty(siblingViolations);

    if ~hasSiblingProp
        violations{end+1} = sprintf('Sibling property VIOLATED: %s', ...
            strjoin(siblingViolations, '; '));
    end

    % ================================================================== %
    % Assemble result
    % ================================================================== %
    result.isValid        = isPrefixFree && isComplete_ && hasSiblingProp;
    result.isPrefixFree   = isPrefixFree;
    result.isComplete     = isComplete_;
    result.hasSiblingProp = hasSiblingProp;
    result.kraftSum       = K_kraft;
    result.lengths        = lengths(:)';
    result.maxDepth       = maxLen;
    result.violations     = violations;
    result.codewords      = codewords;

    % ================================================================== %
    % Verbose output
    % ================================================================== %
    if options.Verbose
        cwStr = strjoin(cellfun(@(c) sprintf('"%s"',c), codewords(:)', ...
                                'UniformOutput', false), ', ');
        fprintf('\n--- isValidHuffmanCode  [Base-%d] ---\n', Base);
        fprintf('  Codewords   : {%s}\n', cwStr);
        fprintf('  Lengths     : [%s]\n', num2str(lengths(:)', '%d '));
        fprintf('  Max depth   : %d\n',   maxLen);
        fprintf('  Kraft sum   : %.6f\n', K_kraft);
        fprintf('\n  Check 1 — Prefix-free  : %s\n', verdict_(isPrefixFree));
        if ~isPrefixFree
            for vi = 1:numel(prefixViolations)
                fprintf('    ! %s\n', prefixViolations{vi});
            end
        end
        fprintf('  Check 2 — Complete (Kraft=1): %s  (K=%.6f)\n', ...
            verdict_(isComplete_), K_kraft);
        fprintf('  Check 3 — Sibling property   : %s\n', verdict_(hasSiblingProp));
        if ~hasSiblingProp
            for vi = 1:numel(siblingViolations)
                fprintf('    ! %s\n', siblingViolations{vi});
            end
        end
        fprintf('\n  ==> VERDICT: %s be a Huffman code for any probability assignment.\n\n', ...
            ternary_(result.isValid, 'CAN', 'CANNOT'));
    end
end

% ---- Local helpers ------------------------------------------------------ %
function s = verdict_(tf)
    if tf
        s = 'PASS';
    else
        s = 'FAIL';
    end
end

function v = ternary_(cond, a, b)
    if cond, v = a; else, v = b; end
end
