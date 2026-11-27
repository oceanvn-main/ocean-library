classdef RuleFilter < handle
    % RULEFILTER  Filter, rank, and prune association rules.
    %
    %   Takes raw rules from RuleGenerator and applies threshold-based
    %   filtering, redundancy removal, and ranking.
    %
    %   USAGE:
    %     filt = OceanStats.Unsupervised.Rules.RuleFilter();
    %     filtered = filt.filter(rules, 'MinLift', 1.0, 'MaxRules', 50);
    %
    %   See also: RuleGenerator, AprioriMiner
    
    methods
        function filtered = filter(~, rules, varargin)
            %FILTER  Apply filters and return cleaned rule set.
            %
            %   filtered = filt.filter(rules, Name, Value)
            %
            %   Name-Value pairs:
            %     'MinSupport'      — minimum support (default: 0)
            %     'MinConfidence'   — minimum confidence (default: 0)
            %     'MinLift'         — minimum lift, 1.0 = better than random (default: 1.0)
            %     'MaxRules'        — return top N rules (default: Inf)
            %     'RemoveRedundant' — remove weaker supersets (default: false)
            %     'SortBy'          — 'lift', 'confidence', or 'support' (default: 'lift')
            
            p = inputParser();
            p.addRequired('rules', @isstruct);
            p.addParameter('MinSupport', 0, @isnumeric);
            p.addParameter('MinConfidence', 0, @isnumeric);
            p.addParameter('MinLift', 1.0, @isnumeric);
            p.addParameter('MaxRules', Inf, @isnumeric);
            p.addParameter('RemoveRedundant', false, @islogical);
            p.addParameter('SortBy', 'lift', @(x) ismember(x, {'lift','confidence','support'}));
            p.parse(rules, varargin{:});
            
            opts = p.Results;
            nOriginal = numel(rules.support);
            
            fprintf('\n[RuleFilter] ============================================\n');
            fprintf('[RuleFilter] Filtering %d rules\n', nOriginal);
            fprintf('[RuleFilter]   MinSupport=%.4f, MinConfidence=%.4f, MinLift=%.4f\n', ...
                opts.MinSupport, opts.MinConfidence, opts.MinLift);
            fprintf('[RuleFilter]   MaxRules=%d, RemoveRedundant=%d, SortBy=%s\n', ...
                opts.MaxRules, opts.RemoveRedundant, opts.SortBy);
            fprintf('[RuleFilter] ============================================\n\n');
            
            % --- Build boolean mask ---
            keep = true(nOriginal, 1);
            
            % 1. Support filter
            keep = keep & (rules.support >= opts.MinSupport);
            nAfterSup = sum(keep);
            
            % 2. Confidence filter
            keep = keep & (rules.confidence >= opts.MinConfidence);
            nAfterConf = sum(keep);
            
            % 3. Lift filter
            keep = keep & (rules.lift >= opts.MinLift);
            nAfterLift = sum(keep);
            
            fprintf('[RuleFilter] After support filter:    %d\n', nAfterSup);
            fprintf('[RuleFilter] After confidence filter: %d\n', nAfterConf);
            fprintf('[RuleFilter] After lift filter:       %d\n', nAfterLift);
            
            % Apply mask
            idx = find(keep);
            
            % 4. Redundancy removal (optional)
            if opts.RemoveRedundant && numel(idx) > 1
                nBeforeRedundant = numel(idx);
                idx = OceanStats.Unsupervised.Rules.RuleFilter.removeRedundant(rules, idx);
                fprintf('[RuleFilter] After redundancy removal: %d (removed %d)\n', ...
                    numel(idx), nBeforeRedundant - numel(idx));
            end
            
            % 5. Sort
            switch opts.SortBy
                case 'lift'
                    [~, sortOrder] = sort(rules.lift(idx), 'descend');
                case 'confidence'
                    [~, sortOrder] = sort(rules.confidence(idx), 'descend');
                case 'support'
                    [~, sortOrder] = sort(rules.support(idx), 'descend');
            end
            idx = idx(sortOrder);
            
            % 6. Truncate to MaxRules
            if numel(idx) > opts.MaxRules
                idx = idx(1:opts.MaxRules);
            end
            
            % --- Build output struct ---
            filtered.antecedent  = rules.antecedent(idx);
            filtered.consequent  = rules.consequent(idx);
            filtered.support     = rules.support(idx);
            filtered.confidence  = rules.confidence(idx);
            filtered.lift        = rules.lift(idx);
            filtered.itemsetSize = rules.itemsetSize(idx);
            
            if ~isempty(rules.antecedentNames)
                filtered.antecedentNames = rules.antecedentNames(idx);
                filtered.consequentNames = rules.consequentNames(idx);
            else
                filtered.antecedentNames = {};
                filtered.consequentNames = {};
            end
            
            filtered.stats.nOriginal  = nOriginal;
            filtered.stats.nFiltered  = numel(idx);
            filtered.stats.sortedBy   = opts.SortBy;
            
            fprintf('[RuleFilter] Final: %d rules (sorted by %s)\n', numel(idx), opts.SortBy);
        end
    end
    
    methods (Static, Access = private)
        function keptIdx = removeRedundant(rules, idx)
            % Remove redundant rules: if A→B exists and A→B,C has
            % lower confidence, the larger rule is redundant.
            
            nRules = numel(idx);
            isRedundant = false(nRules, 1);
            
            % Build lookup: antecedent string → best confidence
            antMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
            
            %Best score for every antecedent
            for i = 1:nRules
                ri = idx(i);
                antKey = mat2str(sort(rules.antecedent{ri}));
                
                if ~antMap.isKey(antKey) || rules.confidence(ri) > antMap(antKey)
                    antMap(antKey) = rules.confidence(ri);
                end
            end
            
            % Mark rules whose antecedent is a superset of another
            % with equal or lower confidence
            %Detect if any subsets of those antecedent can beat parent
            for i = 1:nRules
                ri = idx(i);
                ant = sort(rules.antecedent{ri});
                
                if numel(ant) < 2, continue; end
                
                % Check all proper subsets of the antecedent
                for j = 1:numel(ant)
                    %Remove one item
                    subset = ant([1:j-1, j+1:end]);
                    subKey = mat2str(subset);
                    
                    if antMap.isKey(subKey) && antMap(subKey) >= rules.confidence(ri)
                        isRedundant(i) = true;
                        break;
                    end
                end
            end
            
            keptIdx = idx(~isRedundant);
        end
    end
end
