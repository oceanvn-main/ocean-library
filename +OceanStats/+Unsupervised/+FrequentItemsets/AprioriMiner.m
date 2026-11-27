classdef AprioriMiner < OceanStats.Unsupervised.FrequentItemsets.IMinerStrategy
    % APRIORIMINER  Level-wise frequent itemset mining with anti-monotone pruning.
    %
    %   Implements the classic Apriori algorithm:
    %     k=1: Count single items via column sums
    %     k=2: Count all pairs of frequent-1 items (bottleneck)
    %     k=3+: Generate candidates, prune by anti-monotone, count support
    %     Stop when no more candidates survive
    %
    %   USAGE:
    %     miner = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();
    %     result = miner.mine(X, 0.01);
    %
    %     result.itemsets   — cell array of frequent itemsets (column indices)
    %     result.support    — support value for each itemset
    %     result.stats      — timing, k-level breakdown, pruning counts
    %
    %   See also: IMinerStrategy, TransactionEncoder
    
    methods
        function result = mine(~, X, minSupport, opts)
            %MINE  Find all frequent itemsets using the Apriori algorithm.
            %
            %   result = miner.mine(X, minSupport)
            %   result = miner.mine(X, minSupport, opts)
            %
            %   opts fields (all optional):
            %     .maxK      — maximum itemset size (default: Inf)
            %     .verbose   — print progress (default: true)
            
            if nargin < 4, opts = struct(); end
            if ~isfield(opts, 'maxK'),    opts.maxK = Inf; end
            if ~isfield(opts, 'verbose'), opts.verbose = true; end
            
            tTotal = tic;
            [N, p] = size(X);
            minCount = ceil(minSupport * N);  % Absolute count threshold
            
            if opts.verbose
                fprintf('\n[AprioriMiner] ============================================\n');
                fprintf('[AprioriMiner] Starting Apriori\n');
                fprintf('[AprioriMiner]   N=%d, p=%d, minSupport=%.4f (minCount=%d)\n', ...
                    N, p, minSupport, minCount);
                fprintf('[AprioriMiner] ============================================\n\n');
            end
            
            % Collectors for all frequent itemsets across all k levels
            allItemsets = {};
            allSupport  = [];
            
            % Stats tracking per level
            statsK.candidates = [];
            statsK.pruned     = [];
            statsK.frequent   = [];
            statsK.elapsed    = [];
            
            %% ---- k=1: Single items ----
            tK = tic;
            %Count how often each individual item appears
            [freq1Items, freq1Support] = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.findFrequent1(X, N, minCount);
            nFreq1 = numel(freq1Items);
            
            % Store as cell array of single-element vectors
            for i = 1:nFreq1
                allItemsets{end+1} = uint32(freq1Items(i)); %#ok<AGROW>
            end
            allSupport = [allSupport; freq1Support(:)];
            
            statsK.candidates(1) = p; %Start with ALL p items as candidates
            statsK.pruned(1)     = p - nFreq1; 
            statsK.frequent(1)   = nFreq1;
            statsK.elapsed(1)    = toc(tK);
            
            if opts.verbose
                fprintf('[AprioriMiner] k=1: %d/%d items frequent (%.2f s)\n', ...
                    nFreq1, p, statsK.elapsed(1));
            end
            
            if nFreq1 < 2 || opts.maxK < 2
                result = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.assembleResult(allItemsets, allSupport, statsK, tTotal);
                return;
            end
            
            %% ---- k=2: Pairs (optimized — the bottleneck) ----
            tK = tic;
            %How many such pairs, N choose 2
            nCandidates2 = nFreq1 * (nFreq1 - 1) / 2;
            
            if opts.verbose
                fprintf('[AprioriMiner] k=2: %d candidates (%.0f pairs from %d frequent items)\n', ...
                    nCandidates2, nCandidates2, nFreq1);
            end
            
            [freq2Itemsets, freq2Support] = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.findFrequent2( ...
                X, N, minCount, freq1Items);
            nFreq2 = numel(freq2Support);
            
            for i = 1:nFreq2
                allItemsets{end+1} = freq2Itemsets(i, :); %#ok<AGROW>
            end
            allSupport = [allSupport; freq2Support(:)];
            
            statsK.candidates(2) = nCandidates2;
            statsK.pruned(2)     = nCandidates2 - nFreq2;
            statsK.frequent(2)   = nFreq2;
            statsK.elapsed(2)    = toc(tK);
            
            if opts.verbose
                fprintf('[AprioriMiner] k=2: %d/%d pairs frequent (%.2f s)\n', ...
                    nFreq2, nCandidates2, statsK.elapsed(2));
            end
            
            %% ---- k=3+: General level-wise search ----
            prevFreqItemsets = freq2Itemsets;  % (nFreq2 x 2) matrix
            k = 3;
            
            while size(prevFreqItemsets, 1) >= k && k <= opts.maxK
                tK = tic;
                
                % Step 1: Generate candidates from frequent-(k-1) itemsets
                candidates = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.generateCandidates(prevFreqItemsets, k);
                nCands = size(candidates, 1);
                
                if nCands == 0
                    if opts.verbose
                        fprintf('[AprioriMiner] k=%d: 0 candidates → STOP\n', k);
                    end
                    break;
                end
                
                % Step 2: Prune by anti-monotone property
                % Build set of frequent (k-1)-itemsets for fast lookup
                %For each candidate, check if anysubset of it is not frequent, drop it
                candidates = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.pruneByAntiMonotone(candidates, prevFreqItemsets, k);
                nAfterPrune = size(candidates, 1);
                
                if nAfterPrune == 0
                    if opts.verbose
                        fprintf('[AprioriMiner] k=%d: %d candidates → all pruned → STOP\n', k, nCands);
                    end
                    statsK.candidates(k) = nCands;
                    statsK.pruned(k)     = nCands;
                    statsK.frequent(k)   = 0;
                    statsK.elapsed(k)    = toc(tK);
                    break;
                end
                
                % Step 3: Count support for surviving candidates
                [freqItemsets, freqSupport] = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.countSupport( ...
                    X, N, minCount, candidates);
                nFreqK = numel(freqSupport);
                
                % Store results
                for i = 1:nFreqK
                    allItemsets{end+1} = freqItemsets(i, :); %#ok<AGROW>
                end
                allSupport = [allSupport; freqSupport(:)]; %#ok<AGROW>
                
                statsK.candidates(k) = nCands;
                statsK.pruned(k)     = nCands - nFreqK;
                statsK.frequent(k)   = nFreqK;
                statsK.elapsed(k)    = toc(tK);
                
                if opts.verbose
                    fprintf('[AprioriMiner] k=%d: %d candidates → %d pruned → %d frequent (%.2f s)\n', ...
                        k, nCands, nCands - nFreqK, nFreqK, statsK.elapsed(k));
                end
                
                % Advance to next level
                prevFreqItemsets = freqItemsets;
                k = k + 1;
            end
            
            %% ---- Assemble result ----
            result = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.assembleResult(allItemsets, allSupport, statsK, tTotal);
            
            if opts.verbose
                fprintf('\n[AprioriMiner] COMPLETE: %d frequent itemsets found (%.2f s total)\n', ...
                    numel(result.itemsets), result.stats.elapsed);
                OceanStats.Unsupervised.FrequentItemsets.AprioriMiner.printSummaryTable(result.stats);
            end
        end
    end
    
    %% ---- PRIVATE STATIC HELPERS ----
    methods (Static, Access = private)
        
        function [freqItems, freqSupport] = findFrequent1(X, N, minCount)
            % k=1: column sums — count each item's frequency
            counts = full(sum(X, 1));   % (1 x p) row vector. sum(X, 1): sums down each column. Count how many baskets contain each item
            mask = counts >= minCount;
            freqItems = uint32(find(mask));          % column indices
            freqSupport = counts(mask)' / N;         % support values
        end
        
        function [freqPairs, freqSupport] = findFrequent2(X, N, minCount, freq1Items)
            % k=2: count support for all pairs of frequent-1 items
            % This is the bottleneck — optimized with vectorized AND
            
            nF1 = numel(freq1Items);
            
            % Pre-extract columns for frequent items only
            Xf = X(:, freq1Items);   % N x nF1 submatrix — much smaller
            
            % Count all pairs via matrix multiplication: Xf' * Xf
            % Entry (i,j) = number of baskets containing both item i AND item j
            pairCounts = full(Xf' * Xf);   % nF1 x nF1 — symmetric
            
            % Extract upper triangle (avoid counting (a,b) and (b,a) twice)
            [rowIdx, colIdx] = find(triu(pairCounts, 1) >= minCount);
            
            nPairs = numel(rowIdx);
            freqPairs = zeros(nPairs, 2, 'uint32');
            freqSupport = zeros(nPairs, 1);
            
            for i = 1:nPairs
                % Map back from submatrix indices to original column indices
                freqPairs(i, :) = [freq1Items(rowIdx(i)), freq1Items(colIdx(i))];
                freqSupport(i) = pairCounts(rowIdx(i), colIdx(i)) / N;
            end
        end
        
        function candidates = generateCandidates(freqPrev, k)
            % Generate k-itemset candidates from frequent (k-1)-itemsets
            % Uses the "join step": merge two (k-1)-itemsets that share
            % the first (k-2) items
            
            nPrev = size(freqPrev, 1);
            candidates = zeros(0, k, 'uint32');
            
            % Sort each itemset (should already be sorted)
            freqPrev = sort(freqPrev, 2);
            
            for i = 1:nPrev
                for j = i+1:nPrev
                    % Check if first (k-2) items are identical
                    if isequal(freqPrev(i, 1:k-2), freqPrev(j, 1:k-2)) %Do the prefixes match?
                        % Join: combine the two itemsets
                        newCandidate = sort([freqPrev(i, :), freqPrev(j, k-1)]);
                        
                        % Only add if unique (k items)
                        if numel(unique(newCandidate)) == k
                            candidates(end+1, :) = newCandidate; %#ok<AGROW>
                        end
                    end
                end
            end
            
            % Remove duplicate candidates
            if ~isempty(candidates)
                candidates = unique(candidates, 'rows');
            end
        end
        
        function pruned = pruneByAntiMonotone(candidates, freqPrev, k)
            % Remove candidates whose (k-1)-subsets are NOT all frequent
            % Anti-monotone: if any subset is infrequent, the superset must be too
            
            nCands = size(candidates, 1);
            keep = true(nCands, 1);
            
            % Build hash set of frequent (k-1)-itemsets for fast lookup
            % Convert each row to a string key
            prevSet = containers.Map('KeyType', 'char', 'ValueType', 'logical');
            for i = 1:size(freqPrev, 1)
                key = mat2str(freqPrev(i, :)); %Conver to string
                prevSet(key) = true;
                % prevSet = {"[24 42]"  → true }
            end
            
            for i = 1:nCands
                candidate = candidates(i, :);
                
                % Check all (k-1)-subsets by removing one item at a time
                for j = 1:k
                    subset = candidate([1:j-1, j+1:k]);  % Remove item j
                    key = mat2str(sort(subset));
                    
                    if ~prevSet.isKey(key)
                        keep(i) = false;
                        break;  % One infrequent subset is enough to prune
                    end
                end
            end
            
            pruned = candidates(keep, :); %only rows where keep = true
        end
        
        function [freqItemsets, freqSupport] = countSupport(X, N, minCount, candidates)
            % Count support for candidate itemsets using vectorized AND
            
            nCands = size(candidates, 1); % how many candidates
            k = size(candidates, 2); %How many items per candidate
            counts = zeros(nCands, 1);
            
            for i = 1:nCands
                % AND all columns for this itemset
                mask = true(N, 1);
                for j = 1:k
                    mask = mask & X(:, candidates(i, j)); %AND all columns for this itemset
                end
                counts(i) = sum(mask);
            end
            
            % Filter by minCount
            keep = counts >= minCount;
            freqItemsets = candidates(keep, :);
            freqSupport  = counts(keep) / N;
        end
        
        function result = assembleResult(allItemsets, allSupport, statsK, tTotal)
            % Pack everything into the output struct
            result.itemsets = allItemsets(:);     % column cell array
            result.support  = allSupport(:);     % column vector
            
            result.stats.kBreakdown  = statsK.frequent;
            result.stats.kCandidates = statsK.candidates;
            result.stats.kPruned     = statsK.pruned;
            result.stats.kElapsed    = statsK.elapsed;
            result.stats.elapsed     = toc(tTotal);
            result.stats.algorithm   = 'Apriori';
            result.stats.nItemsets   = numel(allItemsets);
        end
        
        function printSummaryTable(stats)
            % Print k-level breakdown table
            fprintf('\n  %-4s | %12s | %12s | %12s | %8s\n', ...
                'k', 'Candidates', 'Pruned', 'Frequent', 'Time');
            fprintf('  %s\n', repmat('-', 1, 56));
            
            for k = 1:numel(stats.kBreakdown)
                fprintf('  %-4d | %12d | %12d | %12d | %6.2f s\n', ...
                    k, stats.kCandidates(k), stats.kPruned(k), ...
                    stats.kBreakdown(k), stats.kElapsed(k));
            end
            
            fprintf('  %s\n', repmat('-', 1, 56));
            fprintf('  Total: %d frequent itemsets\n', stats.nItemsets);
        end
    end
end
