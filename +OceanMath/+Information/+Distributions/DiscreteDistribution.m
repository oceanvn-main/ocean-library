classdef DiscreteDistribution < OceanMath.Information.Distributions.IDistribution
    % DISCRETEDISTRIBUTION  A validated probability mass function (PMF).
    %
    % Wraps a discrete probability distribution defined over a finite set of
    % states. This is the primary data object used by metric strategies such
    % as ShannonEntropyStrategy and KLDivergenceStrategy.
    %
    % Construction — two modes:
    %   1) From raw labels/data  -> DiscreteDistribution.fromData(samples)
    %      Internally counts frequencies and normalizes to probabilities.
    %
    %   2) From pre-computed PMF -> DiscreteDistribution.fromPMF(states, probs)
    %      Validates that probs sum to 1 and are non-negative.
    %
    % Usage:
    %   d = OceanMath.Information.Distributions.DiscreteDistribution.fromData([1 2 2 3 3 3]);
    %   p = d.evaluate(2);          % P(X=2) = 2/6 = 0.3333
    %   h = d.entropy();            % bits
    %
    %   d2 = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF([1 2 3], [0.5 0.3 0.2]);
    %
    % See also: OceanMath.Information.Distributions.IDistribution
    %           OceanMath.Information.Strategies.ShannonEntropyStrategy

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % States_  : (1 x K) array/cell of unique state labels
        % Probs_   : (1 x K) double, the PMF values. sum(Probs_) == 1.
        % StateMap_: containers.Map for O(1) state lookup
        States_   (:,1)
        Probs_    (1,:) double
        StateMap_ containers.Map
    end

    % ------------------------------------------------------------------ %
    methods (Static)
        function obj = fromData(samples)
            % FROMDATA  Build a PMF by counting frequencies in raw samples.
            %
            % Works with numeric vectors, string arrays, and categoricals.
            %
            % INPUTS:
            %   samples - (N x 1) vector of observed values
            %
            % OUTPUTS:
            %   obj - DiscreteDistribution
            arguments
                samples (:,1)
            end
            [states, ~, idx] = unique(samples);
            counts = accumarray(idx, 1);
            probs  = counts' / sum(counts);
            obj = OceanMath.Information.Distributions.DiscreteDistribution(states, probs);
        end

        function obj = fromPMF(states, probs)
            % FROMPMF  Build a PMF from pre-computed probability values.
            %
            % INPUTS:
            %   states - (K x 1) or (1 x K) array of unique state labels
            %   probs  - (K x 1) or (1 x K) non-negative values summing to 1
            %
            % OUTPUTS:
            %   obj - DiscreteDistribution
            arguments
                states (:,1)
                probs  (:,1) double {mustBeNonnegative}
            end
            if numel(states) ~= numel(probs)
                error('OceanMath:Information:DiscreteDistribution:SizeMismatch', ...
                    'states and probs must have the same number of elements. Got %d vs %d.', ...
                    numel(states), numel(probs));
            end
            if abs(sum(probs) - 1.0) > 1e-9
                error('OceanMath:Information:DiscreteDistribution:InvalidPMF', ...
                    'Probabilities must sum to 1. Got sum = %.10f.', sum(probs));
            end
            obj = OceanMath.Information.Distributions.DiscreteDistribution(states, probs(:)');
        end
    end

    % ------------------------------------------------------------------ %
    methods
        % ---- IDistribution Interface Implementation ---- %

        function p = evaluate(obj, x)
            % EVALUATE  Query P(X = x) for one or more query states.
            %
            % States not in the support return probability 0 (closed-world
            % assumption). This is correct for PMF semantics.
            %
            % INPUTS:
            %   x - scalar, vector, or array of query states (same type as States_)
            %
            % OUTPUTS:
            %   p - double array, same size as x. Each element is P(X = xi).
            x   = x(:);
            p   = zeros(size(x));
            for k = 1:numel(x)
                key = obj.toKey_(x(k));
                if obj.StateMap_.isKey(key)
                    p(k) = obj.StateMap_(key);
                end
            end
        end

        function tf = isValid(obj)
            % ISVALID  Return true if the PMF is well-formed.
            tf = all(obj.Probs_ >= 0) && abs(sum(obj.Probs_) - 1.0) < 1e-9;
        end

        function n = getSupportSize(obj)
            % GETSUPPORTSIZE  Number of distinct states in the support.
            n = numel(obj.States_);
        end

        function supp = getSupport(obj)
            % GETSUPPORT  Return the full vector of state labels.
            supp = obj.States_;
        end

        % ---- Efficient Override for entropy() ---- %

        function h = entropy(obj)
            % ENTROPY  Compute Shannon entropy H(X) in bits.
            %
            % Uses the internal probs directly, avoiding a full evaluate()
            % round-trip for maximum speed.
            %
            % H(X) = -sum(p(x) * log2(p(x))) over all x where p(x) > 0
            p = obj.Probs_(obj.Probs_ > 0);
            h = -sum(p .* log2(p));
        end

        % ---- Useful Utilities ---- %

        function tbl = toTable(obj)
            % TOTABLE  Return a table with States and Probabilities columns.
            tbl = table(obj.States_, obj.Probs_', ...
                'VariableNames', {'State', 'Probability'});
        end

        function disp(obj)
            % DISP  Pretty-print a summary of the distribution.
            fprintf('DiscreteDistribution  [%d states]\n', obj.getSupportSize());
            fprintf('  Entropy : %.4f bits\n', obj.entropy());
            fprintf('  Support : ');
            if isnumeric(obj.States_)
                fprintf('[%s]\n', num2str(obj.States_', '%-6g'));
            else
                disp(obj.States_');
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods
        % ---- Source Coding Bridge ---- %

        function table = huffmanEncode(obj, options)
            % HUFFMANENCODE  Encode this distribution using Huffman coding.
            %
            % Bridges DiscreteDistribution directly to HuffmanCoder.
            %
            % Name-Value Options:
            %   Base (double): code alphabet size. Default: 2.
            %
            % OUTPUTS:
            %   table - OceanMath.Information.Coding.CodeTable
            %
            % Usage:
            %   dist  = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF(...
            %               {'a','b','c'}, [0.5, 0.3, 0.2]);
            %   table = dist.huffmanEncode();
            %   table.printSummary()
            arguments
                obj
                options.Base (1,1) double {mustBePositive, mustBeInteger} = 2
            end
            % Convert states to cell of char for coder
            states = obj.getSupport();
            if isnumeric(states)
                symbols = arrayfun(@(x) num2str(x), states, 'UniformOutput', false);
            elseif ischar(states) || isstring(states)
                symbols = cellstr(states);
            else
                symbols = cellfun(@(x) char(string(x)), num2cell(states), 'UniformOutput', false);
            end
            symbols = symbols(:);   % ensure column cell
            coder  = OceanMath.Information.Coding.HuffmanCoder('Base', options.Base);
            table  = coder.encode(symbols, obj.Probs_);
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function obj = DiscreteDistribution(states, probs)
            % Private constructor — use static factory methods.
            obj.States_ = states(:);
            obj.Probs_  = probs(:)';   % Always row vector internally

            % Build O(1) lookup map
            keys   = cellfun(@obj.toKey_, num2cell(states), 'UniformOutput', false);
            values = num2cell(double(probs(:)'));
            obj.StateMap_ = containers.Map(keys, values);
        end

        function key = toKey_(~, x)
            % TOKEY_ Convert a state value to a consistent string key.
            if isnumeric(x) || islogical(x)
                key = num2str(double(x));
            elseif ischar(x) || isstring(x)
                key = char(x);
            elseif iscategorical(x)
                key = char(x);
            else
                key = char(string(x));
            end
        end
    end
end
