classdef CVSelector < OceanStats.Supervised.PRIM.Selection.IBoxSelector
    % CVSELECTOR Evaluates the candidate boxes on hold-out folds using the ICrossValidator.
    % Generalizes better than the greedy HeuristicSelector by reducing overfitting to small boxes.
    
    properties
        Validator % OceanStats.Base.ICrossValidator implementation
    end
    
    methods
        function obj = CVSelector(validator, primInstance)
            arguments
                validator % E.g., OceanStats.Selection.CrossValidation.KFold
                primInstance = [] % Reference back to the parent PRIM object to grow boxes
            end
            obj.Validator = validator;
            % We will store the parent PRIM instance here so we can call growBoxSequence 
            % within each cross-validation fold.
        end
        
        function bestIdx = select(obj, boxesSeq, X, Y, objectiveFn, primInstance)
            % Modified interface to pass the primInstance down for isolated fold generation
            [N, ~] = size(X);
            numFolds = obj.Validator.NumFolds;
            
            % It's possible the primary sequence grew fewer boxes than MaxBoxes
            % due to the MinBoxSize stopping condition. We only cross-validate 
            % up to the depth of the master sequence.
            numBoxes = length(boxesSeq);
            cvScores = -Inf(numBoxes, numFolds);
            
            % Reset Validator to ensure clean splits if it's stateful
            if ismethod(obj.Validator, 'reset')
                obj.Validator.reset();
            end
            
            % STRICT CROSS-VALIDATION
            % To avoid data leakage, we CANNOT evaluate the `boxesSeq` directly on the hold-out 
            % because those boxes were grown using the entire dataset X! 
            % We must grow an entirely new sequence inside each fold.
            for k = 1:numFolds
                [trainIdx, testIdx] = obj.Validator.getIndices(N, k);
                if ~any(testIdx) || ~any(trainIdx)
                    continue; 
                end
                
                X_train = X(trainIdx, :);
                Y_train = Y(trainIdx);
                X_test  = X(testIdx, :);
                Y_test  = Y(testIdx);
                
                % Grow a fresh sequence strictly on the training fold
                foldBoxesSeq = primInstance.growBoxSequence(X_train, Y_train);
                foldNumBoxes = length(foldBoxesSeq);
                
                % Evaluate the k-th step of the fold-sequence on the holdout test data
                for nu = 1:min(numBoxes, foldNumBoxes)
                    foldBox = foldBoxesSeq{nu};
                    mask = foldBox.isInside(X_test);
                    
                    if sum(mask) > 0
                        cvScores(nu, k) = objectiveFn.evaluate(Y_test(mask), X_test(mask, :));
                    else
                        % If the box is empty in the test set, we score it 0 (or some penalty)
                        % rather than skipping it, so it naturally gets penalized in average
                        cvScores(nu, k) = 0; % Assuming TargetClass proportional objective
                    end
                end
                
                % If the fold-sequence terminated early, carry forward the final score
                % to match the depth of the main sequence.
                if foldNumBoxes < numBoxes
                    for nu = (foldNumBoxes+1):numBoxes
                        cvScores(nu, k) = cvScores(foldNumBoxes, k);
                    end
                end
            end
            
            % Find the box index using the 1-Standard-Error (1-SE) rule
            avgScores = mean(cvScores, 2);
            seScores = std(cvScores, 0, 2) / sqrt(numFolds);
            
            % Debug printing
            fprintf('CV Selection Debug:\n');
            for di = 1:numBoxes
                fprintf('  Step %d: CV Mean = %.4f (SE %.4f)\n', di, avgScores(di), seScores(di));
            end
            
            [maxScore, maxIdx] = max(avgScores);
            threshold = maxScore - seScores(maxIdx);
            candidates = find(avgScores >= threshold);
            bestIdx = candidates(1);
        end
    end
end
