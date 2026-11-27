classdef (Abstract) ISourceCoder < handle
    % ISOURCECODER  Abstract interface for all source-coding algorithms.
    %
    % Every coding algorithm (Huffman, Shannon-Fano, Arithmetic) implements
    % this interface, enabling transparent algorithm substitution via the
    % Strategy Pattern.
    %
    % Contract:
    %   table = coder.encode(symbols, probs)
    %     - symbols : (K x 1) or (1 x K) cell of symbol labels (char/string)
    %     - probs   : (K x 1) or (1 x K) non-negative values summing to 1
    %     - table   : CodeTable (immutable result)
    %
    %   name = coder.getName()
    %     - Returns a human-readable algorithm name (char/string)
    %
    %   tf = coder.isOptimal()
    %     - Returns true if the algorithm achieves the Huffman optimality
    %       bound (i.e., no other prefix-free code has shorter L_avg).
    %     - Huffman: true.  Shannon-Fano: false.  Arithmetic: true (asymptotically).
    %
    % Protected helper:
    %   validateInputs_(symbols, probs) — throws on malformed inputs.
    %   Concrete subclasses should call this at the top of encode().
    %
    % Extension:
    %   To add a new coder, create in +OceanMath/+Information/+Coding/:
    %   1. Implement ISourceCoder
    %   2. Override encode(), getName(), isOptimal()
    %   3. Add string alias in InformationEngine (optional)
    %
    % See also: OceanMath.Information.Coding.HuffmanCoder
    %           OceanMath.Information.Coding.CodeTable

    % ------------------------------------------------------------------ %
    methods (Abstract)
        % ENCODE  Run the coding algorithm and return a CodeTable.
        %
        % INPUTS:
        %   symbols - (K x 1) cell of char/string symbol labels
        %   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1
        %
        % OUTPUTS:
        %   table   - OceanMath.Information.Coding.CodeTable (immutable)
        table = encode(obj, symbols, probs)

        % GETNAME  Return a human-readable algorithm name.
        %
        % OUTPUTS:
        %   name - char or string (e.g., 'Huffman (binary)')
        name = getName(obj)

        % ISOPTIMAL  Return true if this coder achieves prefix-free optimality.
        %
        % OUTPUTS:
        %   tf - logical scalar
        tf = isOptimal(obj)
    end

    % ------------------------------------------------------------------ %
    methods (Access = protected)
        function validateInputs_(~, symbols, probs)
            % VALIDATEINPUTS_  Shared pre-condition checker for all subclasses.
            %
            % Throws a descriptive error on:
            %   - Non-cell symbols
            %   - Size mismatch
            %   - Negative probabilities
            %   - PMF not summing to 1
            %   - Empty inputs
            if ~iscell(symbols)
                error('OceanMath:Information:Coding:ISourceCoder:BadSymbols', ...
                    'symbols must be a cell array of labels. Got: %s.', class(symbols));
            end
            K = numel(symbols);
            if K == 0
                error('OceanMath:Information:Coding:ISourceCoder:EmptyInput', ...
                    'symbols must contain at least one element.');
            end
            if numel(probs) ~= K
                error('OceanMath:Information:Coding:ISourceCoder:SizeMismatch', ...
                    'symbols (%d) and probs (%d) must have the same length.', ...
                    K, numel(probs));
            end
            if any(probs < 0)
                error('OceanMath:Information:Coding:ISourceCoder:NegativeProb', ...
                    'All probabilities must be non-negative.');
            end
            if abs(sum(probs) - 1.0) > 1e-9
                error('OceanMath:Information:Coding:ISourceCoder:InvalidPMF', ...
                    'Probabilities must sum to 1. Got %.10f.', sum(probs));
            end
        end
    end
end
