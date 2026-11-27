classdef RuleGenerator < handle
    % RULEGENERATOR  Convert frequent itemsets into directed association rules.
    %
    %   Takes frequent itemsets from AprioriMiner and splits each into
    %   directed rules with support, confidence, and lift metrics.
    %
    %   USAGE:
    %     gen = OceanStats.Unsupervised.Rules.RuleGenerator();
    %     rules = gen.generate(mineResult, minConfidence);
    %     rules = gen.generate(mineResult, minConfidence, 'ColMap', state.colMap, 'Vocabulary', vocab);
    %
    %   INPUTS:
    %     mineResult     — struct from AprioriMiner.mine() with .itemsets and .support
    %     minConfidence  — minimum confidence threshold (0..1)
    %     'ColMap'       — (optional) column-to-itemID mapping from TransactionEncoder state
    %     'Vocabulary'   — (optional) containers.Map: itemID → human-readable name
    %
    %   OUTPUT:
    %     rules — struct with fields:
    %       .antecedent    — cell array of uint32 vectors (left side column indices)
    %       .consequent    — cell array of uint32 vectors (right side column indices)
    %       .antecedentNames — cell array of strings (human-readable, if vocabulary provided)
    %       .consequentNames — cell array of strings
    %       .support       — support(A ∪ B)
    %       .confidence    — support(A ∪ B) / support(A)
    %       .lift          — confidence / support(B)
    %       .stats         — timing, rule counts per k-level
    %
    %   See also: AprioriMiner, TransactionEncoder
    
    methods
        function rules = generate(~, mineResult, minConfidence, varargin)
            %GENERATE  Generate all association rules above minimum confidence.
            
            p = inputParser();
            p.addRequired('mineResult', @isstruct);
            p.addRequired('minConfidence', @(x) x > 0 && x <= 1);
            p.addParameter('ColMap', [], @isnumeric);
            p.addParameter('Vocabulary', [], @(x) isa(x, 'containers.Map') || isempty(x));
            p.parse(mineResult, minConfidence, varargin{:});
            
            colMap = p.Results.ColMap;
            vocab  = p.Results.Vocabulary;
            hasVocab = ~isempty(colMap) && ~isempty(vocab);
            
            tTotal = tic;
            
            fprintf('\n[RuleGenerator] ============================================\n');
            fprintf('[RuleGenerator] Generating association rules\n');
            fprintf('[RuleGenerator]   minConfidence=%.2f%%\n', minConfidence * 100);
            fprintf('[RuleGenerator]   %d frequent itemsets as input\n', numel(mineResult.itemsets));
            fprintf('[RuleGenerator] ============================================\n\n');
            
            % --- Build support lookup: itemset string → support value  to compute confidence and lift faster (instant lookup)
            supportMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for i = 1:numel(mineResult.itemsets)
                key = mat2str(sort(mineResult.itemsets{i})); %To string
                supportMap(key) = mineResult.support(i);
            end
            
            % --- Generate rules from each itemset ---
            allAntecedent = {};
            allConsequent = {};
            allSupport    = [];
            allConfidence = [];
            allLift       = [];
            ruleKSize     = [];  % Track which k-level each rule came from
            
            nSkippedConf = 0;
            
            for i = 1:numel(mineResult.itemsets)
                itemset = sort(mineResult.itemsets{i});
                k = numel(itemset);
                
                if k < 2, continue; end  % Singletons can't form rules
                
                supAB = mineResult.support(i);  % support(A ∪ B)
                
                % Generate all non-empty proper subsets as antecedents
                % For k items, there are 2^k - 2 possible splits
                nSubsets = 2^k - 2;
                
                for s = 1:nSubsets
                    % Use binary representation to select subset
                    bits = bitget(s, 1:k);
                    antecedent = itemset(bits == 1);
                    consequent = itemset(bits == 0);
                    
                    % Look up support(A) and support(B)
                    keyA = mat2str(sort(antecedent));
                    keyB = mat2str(sort(consequent));
                    
                    if ~supportMap.isKey(keyA) || ~supportMap.isKey(keyB)
                        continue;  % Subset not in frequent itemsets (shouldn't happen)
                    end
                    
                    supA = supportMap(keyA);
                    supB = supportMap(keyB);
                    
                    % Compute metrics
                    confidence = supAB / supA;
                    lift = confidence / supB;
                    
                    % Filter by minConfidence
                    if confidence < minConfidence
                        nSkippedConf = nSkippedConf + 1;
                        continue;
                    end
                    
                    % Store this rule
                    allAntecedent{end+1} = antecedent; %#ok<AGROW>
                    allConsequent{end+1} = consequent; %#ok<AGROW>
                    allSupport(end+1)    = supAB;      %#ok<AGROW>
                    allConfidence(end+1) = confidence;  %#ok<AGROW>
                    allLift(end+1)       = lift;        %#ok<AGROW>
                    ruleKSize(end+1)     = k;           %#ok<AGROW>
                end
            end
            
            % --- Assemble output ---
            rules.antecedent  = allAntecedent(:);
            rules.consequent  = allConsequent(:);
            rules.support     = allSupport(:);
            rules.confidence  = allConfidence(:);
            rules.lift        = allLift(:);
            rules.itemsetSize = ruleKSize(:);
            
            % --- Translate to human-readable names ---
            nRules = numel(allAntecedent);
            
            if hasVocab
                rules.antecedentNames = cell(nRules, 1);
                rules.consequentNames = cell(nRules, 1);
                
                for i = 1:nRules
                    %ColMap : matrixI column -> item ID
                    %Vocab: item ID -> human name
                    rules.antecedentNames{i} = OceanStats.Unsupervised.Rules.RuleGenerator.colsToNames( ...
                        rules.antecedent{i}, colMap, vocab);
                    rules.consequentNames{i} = OceanStats.Unsupervised.Rules.RuleGenerator.colsToNames( ...
                        rules.consequent{i}, colMap, vocab);
                end
            else
                rules.antecedentNames = {};
                rules.consequentNames = {};
            end
            
            % --- Stats ---
            rules.stats.nRules      = nRules;
            rules.stats.nSkipped    = nSkippedConf;
            rules.stats.elapsed     = toc(tTotal);
            rules.stats.minConfidence = minConfidence;
            
            % Count rules per itemset size
            if nRules > 0
                maxK = max(ruleKSize);
                rules.stats.rulesPerK = zeros(1, maxK);
                for kk = 2:maxK
                    rules.stats.rulesPerK(kk) = sum(ruleKSize == kk);
                end
            else
                rules.stats.rulesPerK = [];
            end
            
            fprintf('[RuleGenerator] COMPLETE: %d rules generated, %d skipped (conf < %.2f%%)\n', ...
                nRules, nSkippedConf, minConfidence * 100);
            fprintf('[RuleGenerator] Elapsed: %.2f s\n', rules.stats.elapsed);
            
            % Print rules-per-k breakdown
            if nRules > 0
                fprintf('\n  %-6s | %12s\n', 'k', 'Rules');
                fprintf('  %s\n', repmat('-', 1, 22));
                for kk = 2:maxK
                    if rules.stats.rulesPerK(kk) > 0
                        fprintf('  %-6d | %12d\n', kk, rules.stats.rulesPerK(kk));
                    end
                end
                fprintf('  %s\n', repmat('-', 1, 22));
                fprintf('  Total: %d rules\n', nRules);
            end
        end
    end
    
    %% ---- PRIVATE STATIC HELPERS ----
    methods (Static, Access = private)
        
        function nameStr = colsToNames(cols, colMap, vocab)
            % Convert column indices → item IDs → human-readable names string
            names = cell(1, numel(cols));
            for j = 1:numel(cols)
                itemId = colMap(cols(j));
                if vocab.isKey(itemId)
                    names{j} = vocab(itemId);
                else
                    names{j} = sprintf('ID:%d', itemId);
                end
            end
            nameStr = strjoin(names, ' + ');
        end
    end
end
