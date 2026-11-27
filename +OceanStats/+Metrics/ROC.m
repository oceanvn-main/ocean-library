classdef ROC
    % ROC Receiver Operating Characteristic curve calculation.
    %
    % This class provides methods to compute TPR, FPR, and AUC for binary 
    % classification problems using a custom internal implementation.
    % 
    % Author: Ocean Library System
    % Date: 2026-02-18

    methods (Static)
        function [fpr, tpr, thresholds, auc] = compute(targets, scores, posClass)
            % COMPUTE Calculate ROC curve and AUC.
            %
            % Usage:
            %   1. Binary / Single Class Interest:
            %      [fpr, tpr, thr, auc] = ROC.compute(Y, scores, 1);
            %
            %   2. Multi-Class (One-vs-Rest for ALL classes):
            %      results = ROC.compute(Y, scores);
            %      % results is a struct array with fields: Class, FPR, TPR, AUC
            %
            % Args:
            %   targets: True class labels (N x 1).
            %   scores:  Probability scores (N x K matrix) or (N x 1 vector).
            %            If N x 1, it matches the 'posClass' or 2nd class in binary.
            %   posClass: (Optional) Specific class to evaluate.
            %             If OMITTED and scores is N x K, returns Struct Array for all classes.
            
            arguments
                targets
                scores
                posClass = []
            end
            
            targets = targets(:);
            classes = unique(targets);
            K = length(classes);
            
            % --- CASE A: Multi-Class Logic (Return Struct Array) ---
            if isempty(posClass) && size(scores, 2) > 1
                % If scores is vector but K > 2, we can't do much unless we know which class the scores belong to.
                % Assume scores is N x K matrix corresponding to 'classes'.
                
                if size(scores, 2) ~= K
                     error('ROC:DimensionMismatch', 'Scores matrix columns (%d) must match number of classes (%d).', size(scores, 2), K);
                end
                
                % Initialize result struct
                results = struct('Class', {}, 'FPR', {}, 'TPR', {}, 'Thresholds', {}, 'AUC', {});
                
                for k = 1:K
                    cls = classes(k);
                    % Recursive call for single class
                    [f, t, thr, a] = OceanStats.Metrics.ROC.compute(targets, scores(:, k), cls);
                    
                    results(k).Class = cls;
                    results(k).FPR = f;
                    results(k).TPR = t;
                    results(k).Thresholds = thr;
                    results(k).AUC = a;
                end
                
                % Map outputs to first argument for convenience
                fpr = results;
                tpr = []; thresholds = []; auc = [];
                return;
            end
            
            % --- CASE B: Single Class / Binary Logic ---
            
            % 1. Determine Positive Class if missing (Binary default)
            if isempty(posClass)
                if islogical(targets)
                    % Logical implies True is positive
                    posClass = true; 
                    K = 2; % Treat as binary even if only all false or all true present
                else
                    % Default to last class for binary or multi-class with single score
                    posClass = classes(end);
                end
            end
            
            % 2. Extract relevant scores
            currentScores = scores;
            if size(scores, 2) > 1
                % Find column for posClass
                % Try index match
                if isscalar(posClass) && isnumeric(posClass) && posClass <= size(scores, 2) && ...
                   (isnumeric(classes) && any(classes == posClass))
                    % Ambiguity: Is '2' the label 2 or the 2nd index?
                    % If classes are [1, 2], label 2 is index 2.
                    colIdx = find(classes == posClass, 1);
                else
                    % Find by label (ISMEMBER for set support)
                    if iscell(classes) || isstring(classes) || ischar(classes)
                        [~, colIdx] = ismember(posClass, classes);
                        colIdx = colIdx(colIdx > 0); % Filter unmatched
                    else
                         [~, colIdx] = ismember(posClass, classes);
                         colIdx = colIdx(colIdx > 0);
                    end
                end
                
                if isempty(colIdx)
                    error('ROC:ClassNotFound', 'Positive class "%s" not found in target classes.', string(posClass));
                end
                
                % If posClass is a set (e.g. ["A", "C"]), we need to aggregate scores?
                % The standard usage for N x K scores is One-vs-Rest.
                % If the user wants "A or C" vs "Rest", they should probably sum the probabilities of A and C manually 
                % BEFORE calling this, OR we support summing here.
                %
                % Current design: 'scores' arg is N x 1 if specific focus. 
                % If N x K passed, and strict single posClass, we pick that column.
                % If multiple posClasses passed with N x K scores, we should probably SUM those columns 
                % to get P(Positive).
                
                if length(colIdx) > 1
                    currentScores = sum(scores(:, colIdx), 2);
                else
                    currentScores = scores(:, colIdx);
                end
            end
            
            currentScores = currentScores(:);
            
            if length(targets) ~= length(currentScores)
                error('ROC:InputSizeMismatch', 'Targets and scores must have same length.');
            end
            
            % 3. Binary ROC Calculation
            % Determine positive samples
            if islogical(targets)
                % If targets is logical, TRUE is always positive
                isPos = targets;
                if ~isempty(posClass)
                     % If user provided posClass for logical, warn or check consistency?
                     % For now, assume logical implies the mapping is done.
                end
            else
                 % Standard case: isPos if target is in set of posClass
                 if isempty(posClass)
                     error('ROC:NoPosClass', 'Should have strictly determined posClass by now.');
                 end
                 isPos = ismember(targets, posClass);
            end
            P = sum(isPos);
            N = sum(~isPos);
            
            if P == 0 || N == 0
                 warning('ROC:NoSamples', 'Class %s has no positive or no negative samples.', string(posClass));
                 fpr = nan; tpr = nan; thresholds = nan; auc = nan;
                 return;
            end
            
            [sortedScores, idx] = sort(currentScores, 'descend');
            sortedIsPos = isPos(idx);
            
            % Unique thresholds (standard efficient algorithm)
            diff_idx = diff(sortedScores) ~= 0;
            valid_idx = [true; diff_idx]; 
            
            thresholds = [Inf; sortedScores(valid_idx); -Inf];
            
            tp_cum = [0; cumsum(sortedIsPos)];
            fp_cum = [0; cumsum(~sortedIsPos)];
            
            % Pick points at change indices
            change_indices = [find(diff_idx); length(sortedScores)];
            
            tp_vals = [0; tp_cum(change_indices + 1)];
            fp_vals = [0; fp_cum(change_indices + 1)];
            
            tpr = tp_vals / P;
            fpr = fp_vals / N;
            
            auc = OceanStats.Metrics.ROC.calculateAUC(fpr, tpr);
        end
        
        function auc = calculateAUC(fpr, tpr)
            % CALCULATEAUC Compute Area Under Curve using trapezoidal rule.
            % fpr, tpr should be sorted by fpr ascending.
            
            % Ensure sorted by FPR
            [fpr, sIdx] = sort(fpr);
            tpr = tpr(sIdx);
            
            % Trapezoidal rule: sum (x_i+1 - x_i) * (y_i + y_i+1) / 2
            dx = diff(fpr);
            y_avg = (tpr(1:end-1) + tpr(2:end)) / 2;
            auc = sum(dx .* y_avg);
        end
    end
end
