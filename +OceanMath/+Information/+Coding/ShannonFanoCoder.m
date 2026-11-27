classdef ShannonFanoCoder < OceanMath.Information.Coding.ISourceCoder
    % SHANNONFANOCODER Shannon-Fano binary prefix source coder.
    %
    % Implements the top-down binary prefix-free source coding algorithm
    % developed by Shannon and Fano. It sorts the symbols by probability
    % in descending order, recursively partitions them into two contiguous
    % groups of minimal probability difference, and prepends '0' or '1'
    % to build the prefix-free codes.
    %
    % Shannon-Fano is a top-down algorithm and is generally sub-optimal
    % compared to bottom-up Huffman coding.
    %
    % Properties:
    %   None (Stateless strategy class).
    %
    % Examples:
    %   coder = OceanMath.Information.Coding.ShannonFanoCoder();
    %   table = coder.encode({'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
    %   table.printSummary()
    %
    % See also: OceanMath.Information.Coding.ISourceCoder,
    %           OceanMath.Information.Coding.CodeTable,
    %           OceanMath.Information.Coding.HuffmanCoder

    methods
        function obj = ShannonFanoCoder()
            % SHANNONFANOCODER Construct a ShannonFanoCoder.
        end

        % ---- ISourceCoder interface implementation ---- %

        function table = encode(obj, symbols, probs)
            % ENCODE Run the Shannon-Fano algorithm and return a CodeTable.
            %
            % INPUTS:
            %   symbols - (K x 1) or (1 x K) cell of char/string labels.
            %   probs   - (K x 1) or (1 x K) non-negative doubles summing to 1.
            %
            % OUTPUTS:
            %   table - OceanMath.Information.Coding.CodeTable
            arguments
                obj
                symbols (:,1) cell
                probs   (1,:) double {mustBeNonnegative}
            end

            % Validate inputs using the protected base class helper
            obj.validateInputs_(symbols, probs);

            % Prepare shapes
            symbols_col = symbols(:);
            probs_row = probs(:)';
            K = numel(symbols_col);

            % Handle single symbol edge case
            if K == 1
                table = OceanMath.Information.Coding.CodeTable.build( ...
                    symbols_col, probs_row, {'0'}, ...
                    'Base', 2, ...
                    'CoderName', obj.getName());
                return;
            end

            % Sort probabilities in descending order
            [probs_sorted, sort_idx] = sort(probs_row, 'descend');

            % Pre-allocate codewords array for sorted symbols
            codewords_sorted = cell(K, 1);
            codewords_sorted(:) = {''};

            % Perform top-down recursive splitting
            recurseSplit(1:K);

            % Nested recursive function for partition search
            function recurseSplit(sub_indices)
                nSub = length(sub_indices);
                if nSub <= 1
                    return;
                end

                p_sub = probs_sorted(sub_indices);
                total_sub = sum(p_sub);
                
                % Find split index to minimize the difference in cumulative sum
                best_diff = Inf;
                split_idx = 1;
                running_sum = 0;
                for i = 1:(nSub - 1)
                    running_sum = running_sum + p_sub(i);
                    diff = abs(2 * running_sum - total_sub);
                    if diff < best_diff
                        best_diff = diff;
                        split_idx = i;
                    end
                end

                % Assign '0' to the first group, '1' to the second group
                for idx = 1:split_idx
                    g_idx = sub_indices(idx);
                    codewords_sorted{g_idx} = [codewords_sorted{g_idx}, '0'];
                end
                for idx = (split_idx + 1):nSub
                    g_idx = sub_indices(idx);
                    codewords_sorted{g_idx} = [codewords_sorted{g_idx}, '1'];
                end

                % Recurse on subgroups
                recurseSplit(sub_indices(1:split_idx));
                recurseSplit(sub_indices(split_idx + 1:end));
            end

            % Map codewords back to original input order
            codewords_orig = cell(K, 1);
            codewords_orig(sort_idx) = codewords_sorted;

            % Build and return immutable CodeTable
            table = OceanMath.Information.Coding.CodeTable.build( ...
                symbols_col, probs_row, codewords_orig, ...
                'Base', 2, ...
                'CoderName', obj.getName());
        end

        function name = getName(~)
            % GETNAME Return the human-readable algorithm name.
            name = 'Shannon-Fano (binary)';
        end

        function tf = isOptimal(~)
            % ISOPTIMAL Shannon-Fano is not optimal in general.
            tf = false;
        end
    end
end
