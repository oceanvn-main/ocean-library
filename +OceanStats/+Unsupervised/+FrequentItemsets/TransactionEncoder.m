classdef TransactionEncoder < handle
    % TRANSACTIONENCODER  Converts transactional baskets to sparse Boolean matrix.
    %
    %   Follows the fit/apply pattern (consistent with IDataTransform):
    %     state = encoder.fit(baskets)         % Learn column mapping from data
    %     X     = encoder.apply(baskets, state) % Convert baskets to sparse matrix
    %     [X, state] = encoder.fitApply(baskets) % Convenience: fit + apply
    %
    %   The encoder handles non-contiguous item IDs by building a contiguous
    %   column mapping (colMap), so the output matrix always has columns 1..p.
    %
    %   OUTPUT:
    %     X      — sparse(N x p) logical matrix (1 = item present in basket)
    %     state  — struct with fields:
    %       .colMap    — (1 x p) uint32 vector: column j represents item colMap(j)
    %       .lookup    — dense array: lookup(itemID) = column index
    %       .p         — number of unique items (columns)
    %       .maxItemId — largest item ID seen during fit
    %
    %   USAGE:
    %     encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();
    %     [X, state] = encoder.fitApply(result.baskets);
    %
    %     % Decode back to baskets:
    %     decoded = encoder.decode(X, state);
    %
    %     % Validate round-trip:
    %     ok = encoder.validate(X, result.baskets, state);
    %
    %   See also: OceanPipeline.Ingestion.TransactionalReader
    
    methods
        function state = fit(~, baskets)
            % FIT  Discover unique item IDs and build column mapping.
            %
            %   state = encoder.fit(baskets)
            %
            %   Scans all baskets to find unique item IDs, sorts them, and
            %   builds a dense lookup array for O(1) ID → column conversion.
            
            fprintf('[TransactionEncoder] fit: scanning %d baskets...\n', numel(baskets));
            tFit = tic;
            
            % Collect all unique item IDs across all baskets
            allItems = cellfun(@(b) b(:), baskets, 'UniformOutput', false);
            allItems = cell2mat(allItems);
            uniqueItems = unique(allItems);
            
            % Build column mapping: sorted unique IDs
            state.colMap    = uint32(sort(uniqueItems)');   % (1 x p) row vector
            state.p         = numel(state.colMap);
            state.maxItemId = max(state.colMap);
            
            % Build dense lookup array: lookup(itemID) = column index
            % This avoids slow containers.Map lookups during apply()
            state.lookup = zeros(state.maxItemId, 1, 'uint32');
            state.lookup(state.colMap) = uint32(1:state.p); %LEFT size: which slots to fill, RIGHT: what values to put
            
            fprintf('[TransactionEncoder] fit complete: p=%d unique items (%.2f s)\n', ...
                state.p, toc(tFit));
        end
        
        function X = apply(~, baskets, state)
            % APPLY  Convert baskets to sparse Boolean matrix using learned mapping.
            %
            %   X = encoder.apply(baskets, state)
            %
            %   Uses COO (Coordinate) construction: collects all (row, col)
            %   indices first, then calls sparse() once. This is orders of
            %   magnitude faster than inserting elements one at a time.
            
            N = numel(baskets);
            fprintf('[TransactionEncoder] apply: encoding %d baskets into sparse(%d x %d)...\n', ...
                N, N, state.p);
            tApply = tic;
            
            % --- Memory estimate & safety check ---
            totalNZ = sum(cellfun(@numel, baskets));
            sparseMemMB = totalNZ * 16 / 1e6;  % ~16 bytes per nonzero in MATLAB sparse
            fprintf('[TransactionEncoder]   Total nonzeros: %d (~%.0f MB sparse)\n', ...
                totalNZ, sparseMemMB);
            
            if sparseMemMB > 2048
                warning('TransactionEncoder:LargeMatrix', ...
                    'Estimated sparse matrix size %.0f MB exceeds 2 GB. Consider subsampling.', ...
                    sparseMemMB);
            end
            
            % --- COO construction( coordinate) ---
            % Pre-allocate row and column index arrays
            rowIdx = zeros(totalNZ, 1, 'double');
            colIdx = zeros(totalNZ, 1, 'double');
            
            ptr = 1;
            for i = 1:N
                items = baskets{i};
                k = numel(items);
                
                %Row = number of baskets, columns = number of unique items
                % Row indices: all equal i (this basket)
                rowIdx(ptr:ptr+k-1) = i;
                
                % Column indices: look up each item ID in the dense mapping
                colIdx(ptr:ptr+k-1) = double(state.lookup(items));
                
                ptr = ptr + k;
            end
            
            % Remove any unmapped items (colIdx == 0 means item not in fit vocabulary)
            valid = colIdx > 0;
            if any(~valid)
                nDropped = sum(~valid);
                fprintf('[TransactionEncoder]   WARNING: %d items not in vocabulary (dropped)\n', nDropped);
                rowIdx = rowIdx(valid);
                colIdx = colIdx(valid);
            end
            
            % Construct sparse logical matrix in one call
            X = sparse(rowIdx, colIdx, true, N, state.p);
            
            elapsedSec = toc(tApply);
            fprintf('[TransactionEncoder] apply complete: nnz=%d, density=%.6f%% (%.2f s)\n', ...
                nnz(X), 100 * nnz(X) / (N * state.p), elapsedSec);
        end
        
        function [X, state] = fitApply(obj, baskets)
            % FITAPPLY  Convenience: fit + apply in one call.
            %
            %   [X, state] = encoder.fitApply(baskets)
            
            state = obj.fit(baskets);
            X = obj.apply(baskets, state);
        end
        
        function baskets = decode(~, X, state)
            % DECODE  Reverse: sparse matrix → cell array of baskets.
            %
            %   baskets = encoder.decode(X, state)
            %
            %   For each row of X, find the nonzero columns and map them
            %   back to original item IDs using state.colMap.
            
            %How many rows
            N = size(X, 1);
            baskets = cell(N, 1); %Empty cell array for results
            
            for i = 1:N
                colIndices = find(X(i, :)); %Find the columns that are not zero
                baskets{i} = state.colMap(colIndices);  % map back to original IDs
            end
        end
        
        function ok = validate(obj, X, baskets, state)
            % VALIDATE  Round-trip integrity check: encode → decode → compare.
            %
            %   ok = encoder.validate(X, baskets, state)
            %
            %   Decodes the sparse matrix back to baskets and compares with
            %   the originals. Returns true if all baskets match.
            
            decoded = obj.decode(X, state);
            ok = true;
            N = numel(baskets);
            
            for i = 1:N
                original     = sort(baskets{i}(:))';
                reconstructed = sort(decoded{i}(:))';
                
                if ~isequal(original, reconstructed)
                    ok = false;
                    fprintf('[TransactionEncoder] MISMATCH at basket %d:\n', i);
                    fprintf('  Original:      [%s]\n', num2str(original));
                    fprintf('  Reconstructed: [%s]\n', num2str(reconstructed));
                    return;
                end
            end
            
            fprintf('[TransactionEncoder] Round-trip validation PASSED (%d baskets)\n', N);
        end
    end
    
    methods (Static)
        function printProfile(X, state, label)
            % PRINTPROFILE  Print matrix statistics.
            %
            %   TransactionEncoder.printProfile(X, state, 'aisle')
            
            if nargin < 3, label = ''; end
            
            [N, p] = size(X);
            nz = nnz(X);
            sparsity = 1 - nz / (N * p);
            memMB = nz * 16 / 1e6;
            
            fmt = @OceanPipeline.Ingestion.TransactionalReader.formatNumber;
            
            fprintf('  %-14s | %12s | %8s | %12s | %8.0f MB\n', ...
                label, fmt(N), fmt(p), fmt(nz), memMB);
        end
    end
end
