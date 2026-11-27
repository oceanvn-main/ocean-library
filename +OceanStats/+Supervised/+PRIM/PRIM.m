classdef PRIM < handle
    % PRIM - Patient Rule Induction Method
    % Faithful implementation with strict alpha peeling
    
    properties
        ObjectiveFunction
        Alpha      (1,1) double {mustBePositive, mustBeLessThan(Alpha,1)} = 0.05
        MinBoxSize (1,1) double {mustBePositive} = 10
        MaxBoxes   (1,1) double {mustBePositive} = 5
        BoxSelector
        
        % Specifies how the algorithm handles point-masses (e.g., zero-inflated data).
        % 'Standard' (Default) - "Impatient" native. Accepts any boundary move that maximizes purity, 
        %                        even if it drops > Alpha portion of the data instantly (e.g., dropping all zeros).
        %                        Best for predictive accuracy and highly potent sparse sets.
        % 'Patient'            - Strict adherence to the `Alpha` parameter. If a boundary move 
        %                        implies dropping a point-mass that is larger than Alpha, the 
        %                        feature is ignored. Forces continuous feature exploration (ESL Textbook style).
        PeelingStrategy (1,:) char {mustBeMember(PeelingStrategy, {'Standard', 'Patient'})} = 'Standard'
        
        Boxes cell
        GlobalMean (1,1) double
        GlobalScore (1,1) double
    end
    
    methods
        function obj = PRIM(options)
            arguments
                options.Alpha = 0.05
                options.MinBoxSize = 10
                options.MaxBoxes = 5
                options.ObjectiveFunction = OceanStats.Supervised.PRIM.Criterion.MeanResponse()
                options.BoxSelector = OceanStats.Supervised.PRIM.Selection.HeuristicSelector()
                options.PeelingStrategy = 'Standard'
            end
            
            obj.Alpha = options.Alpha;
            obj.MinBoxSize = options.MinBoxSize;
            obj.MaxBoxes = options.MaxBoxes;
            obj.ObjectiveFunction = options.ObjectiveFunction;
            obj.BoxSelector = options.BoxSelector;
            obj.PeelingStrategy = options.PeelingStrategy;
        end
        
        function fit(obj, X, Y)
            
            assert(size(X,1) == numel(Y), 'X and Y size mismatch.')
            
            % Internal Shuffle removed to see if it brings us closer to textbook order
            % N = size(X,1);
            % shuffleIdx = randperm(N);
            % X = X(shuffleIdx,:);
            % Y = Y(shuffleIdx);
            
            obj.GlobalMean  = mean(Y);
            obj.GlobalScore = obj.ObjectiveFunction.evaluate(Y, X);
            obj.Boxes = {};
            
            remainX = X;
            remainY = Y;
            
            for k = 1:obj.MaxBoxes
                
                if numel(remainY) < obj.MinBoxSize
                    break;
                end
                
                boxesSeq = obj.growBoxSequence(remainX, remainY);
                
                if isempty(boxesSeq)
                    break;
                end
                
                bestInd = obj.BoxSelector.select( ...
                    boxesSeq, remainX, remainY, ...
                    obj.ObjectiveFunction, obj);
                
                bestBox = boxesSeq{bestInd};
                
                bestBox = obj.pasteBox(bestBox, remainX, remainY);
                bestBox = obj.pruneBox(bestBox, remainX, remainY);
                
                % Stopping rule: require improvement over remaining global score
                remainScore = obj.ObjectiveFunction.evaluate(remainY, remainX);
                
                if bestBox.Score <= remainScore
                    break;
                end
                
                obj.Boxes{end+1} = bestBox;
                
                mask = bestBox.isInside(remainX);
                remainX(mask,:) = [];
                remainY(mask) = [];
            end
        end
        
        function boxesSeq = growBoxSequence(obj, X, Y)
            
            import OceanStats.Supervised.PRIM.Box;
            
            [N, ~] = size(X);
            
            currentBox = Box(X, Y, obj.ObjectiveFunction);
            boxesSeq = {currentBox};
            
            indices = (1:N)';
            
            while numel(indices) > obj.MinBoxSize
                
                stepNum = numel(boxesSeq);
                [nextBox, keepMask] = ...
                    obj.peelOneStep(currentBox, X(indices,:), Y(indices));
                
                if isempty(nextBox)
                    break;
                end
                
                % Diagnostic: Log first 10 steps of Rule 1
                if numel(obj.Boxes) == 0 && stepNum <= 10
                    fprintf('Step %d: Support %.4f, Purity %.4f, BestFeat: %d\n', ...
                        stepNum, nextBox.NSamples / N, nextBox.Score, size(nextBox.Bounds,1));
                    % We need to find which feature changed.
                    diff = find(nextBox.Bounds(:,1) ~= currentBox.Bounds(:,1) | ...
                                nextBox.Bounds(:,2) ~= currentBox.Bounds(:,2));
                    if ~isempty(diff)
                        fprintf('       Moved Feat: %d\n', diff(1));
                    else
                        fprintf('       (No boundary move)\n');
                    end
                end
                
                currentBox = nextBox;
                boxesSeq{end+1} = currentBox;
                
                indices = indices(keepMask);
            end
        end
        
        function [bestNextBox, keepMask] = ...
                peelOneStep(obj, currentBox, X_box, Y_box)
            
            [N, D] = size(X_box);
            
            peelCount = max(1, floor(N * obj.Alpha));
            
            if N - peelCount < obj.MinBoxSize
                bestNextBox = [];
                keepMask = [];
                return;
            end
            
            bestScore = -Inf;
            bestSupport = -Inf;
            bestMovesBoundary = false;
            bestNextBox = [];
            keepMask = [];
            
            for j = 1:D
                
                feature = X_box(:,j);
                [sortedVal, sortIdx] = sort(feature);
                
                % ----- Bottom peel
                % We want to peel at least `peelCount` points, but we MUST
                % move the boundary. So we find the value at the peelCount index
                % and remove ALL points <= that value.
                valA_threshold = feature(sortIdx(peelCount));
                
                % Keep points that are strictly greater than the threshold
                % OR if that would empty the box, we just can't peel this feature
                maskA = feature > valA_threshold;
                supportA = sum(maskA);
                
                if supportA >= obj.MinBoxSize
                    scoreA = obj.ObjectiveFunction.evaluate(Y_box(maskA), X_box(maskA,:));
                    valA = min(feature(maskA));
                    
                    movesA = valA > currentBox.Bounds(j,1);
                    
                    isPatient = true;
                    if strcmp(obj.PeelingStrategy, 'Patient')
                        fractionRemoved = (N - supportA) / N;
                        if fractionRemoved > max(obj.Alpha * 2.5, 0.15)
                            isPatient = false;
                        end
                    end
                    
                    if movesA && isPatient && obj.isBetterPeel(scoreA, supportA, movesA, ...
                                      bestScore, bestSupport, bestMovesBoundary)
                        bestScore = scoreA;
                        bestSupport = supportA;
                        bestMovesBoundary = movesA;
                        keepMask = maskA;
                        bestNextBox = currentBox.copy();
                        bestNextBox.Bounds(j,1) = valA;
                        bestNextBox.Score = scoreA;
                        bestNextBox.NSamples = supportA;
                        bestNextBox.BoxMean = mean(Y_box(maskA));
                    end
                end
                
                % ----- Top peel
                valB_threshold = feature(sortIdx(end - peelCount + 1));
                
                % Keep points strictly less than threshold
                maskB = feature < valB_threshold;
                supportB = sum(maskB);
                
                if supportB >= obj.MinBoxSize
                    scoreB = obj.ObjectiveFunction.evaluate(Y_box(maskB), X_box(maskB,:));
                    valB = max(feature(maskB));
                    movesB = valB < currentBox.Bounds(j,2);
                    
                    isPatient = true;
                    if strcmp(obj.PeelingStrategy, 'Patient')
                        fractionRemoved = (N - supportB) / N;
                        if fractionRemoved > max(obj.Alpha * 2.5, 0.15)
                             isPatient = false;
                        end
                    end
                    
                    if movesB && isPatient && obj.isBetterPeel(scoreB, supportB, movesB, ...
                                      bestScore, bestSupport, bestMovesBoundary)
                        bestScore = scoreB;
                        bestSupport = supportB;
                        bestMovesBoundary = movesB;
                        keepMask = maskB;
                        bestNextBox = currentBox.copy();
                        bestNextBox.Bounds(j,2) = valB;
                        bestNextBox.Score = scoreB;
                        bestNextBox.NSamples = supportB;
                        bestNextBox.BoxMean = mean(Y_box(maskB));
                    end
                end
            end
        end
        
        function better = isBetterPeel(obj, s, sup, mov, bS, bSup, bMov)
            % s, sup, mov = current candidate score, support, boundary_moved
            % bS, bSup, bMov = previous best
            
            % 1. Score priority (modulo small tolerance)
            if s > bS + 1e-12
                better = true; return;
            elseif s < bS - 1e-12
                better = false; return;
            end
            
            % 2. Tied Score: Prefer boundary movement (Signal over patient stripping)
            if mov && ~bMov
                better = true; return;
            elseif ~mov && bMov
                better = false; return;
            end
            
            % 3. Tied Score and Boundary Status: Support priority (Patience)
            if sup > bSup + 1e-12
                better = true; return;
            elseif sup < bSup - 1e-12
                better = false; return;
            end
            
            % 4. All tied: Deterministic (favor higher feature index/last feature)
            better = true; 
        end
        
        function box = pasteBox(obj, box, X, Y)
            
            [~, D] = size(X);
            improved = true;
            
            while improved
                improved = false;
                bestScore = box.Score;
                bestBounds = box.Bounds;
                bestMask = box.isInside(X);
                
                for j = 1:D
                    
                    % ----- Expand lower bound
                    candidates = X(:,j) < box.Bounds(j,1);
                    
                    if any(candidates)
                        
                        nextVal = max(X(candidates,j));
                        
                        tempBox = box.copy();
                        tempBox.Bounds(j,1) = nextVal;
                        
                        mask = tempBox.isInside(X);
                        
                        if sum(mask) >= obj.MinBoxSize
                            
                            score = obj.ObjectiveFunction.evaluate( ...
                                Y(mask), X(mask,:));
                            
                            if score > bestScore
                                bestScore = score;
                                bestBounds = tempBox.Bounds;
                                bestMask = mask;
                                improved = true;
                            end
                        end
                    end
                    
                    % ----- Expand upper bound
                    candidates = X(:,j) > box.Bounds(j,2);
                    
                    if any(candidates)
                        
                        nextVal = min(X(candidates,j));
                        
                        tempBox = box.copy();
                        tempBox.Bounds(j,2) = nextVal;
                        
                        mask = tempBox.isInside(X);
                        
                        if sum(mask) >= obj.MinBoxSize
                            
                            score = obj.ObjectiveFunction.evaluate( ...
                                Y(mask), X(mask,:));
                            
                            if score > bestScore
                                bestScore = score;
                                bestBounds = tempBox.Bounds;
                                bestMask = mask;
                                improved = true;
                            end
                        end
                    end
                end
                
                if improved
                    box.Bounds = bestBounds;
                    box.Score = bestScore;
                    box.NSamples = sum(bestMask);
                    box.BoxMean = mean(Y(bestMask));
                end
            end
        end
        
        function box = pruneBox(obj, box, X, Y)
            % Greedy Backward Elimination.
            % Aligns with textbook simplicity by dropping non-critical bounds.
            
            [~, D] = size(X);
            
            % Tolerance for pruning. 
            % Instead of cascading downward tolerance at every step,
            % we establish a hard baseline from the initial pasted box.
            tolerance = 0.05;  
            baselineScore = box.Score - tolerance;
            improved = true;
            
            while improved
                improved = false;
                
                % Find the relaxation that gives us the best score
                % (as long as it stays >= baselineScore)
                bestDropScore = -Inf;
                bestNewBounds = [];
                bestNewMask = [];
                
                for j = 1:D
                    % Try relax lower (drop the bound entirely to -Inf)
                    if box.Bounds(j,1) > -Inf
                        tempBounds = box.Bounds;
                        tempBounds(j,1) = -Inf;
                        mask = obj.isInsideWithBounds(X, tempBounds);
                        
                        if sum(mask) >= obj.MinBoxSize
                            score = obj.ObjectiveFunction.evaluate(Y(mask), X(mask,:));
                            if score > bestDropScore
                                bestDropScore = score;
                                bestNewBounds = tempBounds;
                                bestNewMask = mask;
                            end
                        end
                    end
                    
                    % Try relax upper (drop the bound entirely to Inf)
                    if box.Bounds(j,2) < Inf
                        tempBounds = box.Bounds;
                        tempBounds(j,2) = Inf;
                        mask = obj.isInsideWithBounds(X, tempBounds);
                        
                        if sum(mask) >= obj.MinBoxSize
                            score = obj.ObjectiveFunction.evaluate(Y(mask), X(mask,:));
                            if score > bestDropScore
                                bestDropScore = score;
                                bestNewBounds = tempBounds;
                                bestNewMask = mask;
                            end
                        end
                    end
                end
                
                % Accept the best relaxation if it respects the global baseline
                if ~isempty(bestNewBounds) && bestDropScore >= baselineScore
                    box.Bounds = bestNewBounds;
                    box.Score = bestDropScore;
                    box.NSamples = sum(bestNewMask);
                    box.BoxMean = mean(Y(bestNewMask));
                    improved = true;
                    
                    % DO NOT lower the baseline! This prevents a cascading
                    % death spiral of purity.
                end
            end
        end
        
        function mask = isInsideWithBounds(~, X, Bounds)
            % Helper to avoid creating Box objects in inner loop
            [N, D] = size(X);
            mask = true(N, 1);
            for j = 1:D
                mask = mask & (X(:,j) >= Bounds(j,1)) & (X(:,j) <= Bounds(j,2));
            end
        end
        
        function yt = predict(obj, X)
            
            N = size(X,1);
            yt = repmat(obj.GlobalMean, N,1);
            unassigned = true(N,1);
            
            for b = 1:numel(obj.Boxes)
                
                box = obj.Boxes{b};
                mask = box.isInside(X) & unassigned;
                
                yt(mask) = box.BoxMean;
                unassigned(mask) = false;
            end
        end
    end
end