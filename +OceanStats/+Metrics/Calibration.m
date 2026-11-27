classdef Calibration
    % CALIBRATION Utility for computing reliability diagrams and ECE.
    %
    % This class provides methods to assess the probabilistic calibration 
    % of classifiers, comparing predicted probabilities against actual frequencies.
    % 
    % KEY METRICS:
    %   - Expected Calibration Error (ECE): A weighted average of the absolute 
    %     difference between predicted confidence and actual frequency.
    
    methods (Static)
        function [bins, frequencies, ece] = compute(targets, scores, nBins)
            % COMPUTE Calculates calibration data (Reliability Diagram).
            %
            % Args:
            %   targets : Binary labels (N x 1, logical or {0,1}).
            %   scores  : Predicted probabilities (N x 1, in [0,1]).
            %   nBins   : Number of bins (default: 10).
            
            arguments
                targets
                scores
                nBins (1,1) double = 10
            end
            
            targets = double(targets(:));
            scores = scores(:);
            
            if length(targets) ~= length(scores)
                error('Calibration:InputSizeMismatch', 'Targets and scores must have same length.');
            end
            
            % 1. Create Bins (Uniform interval in [0, 1])
            binEdges = linspace(0, 1, nBins + 1);
            bins = (binEdges(1:end-1) + binEdges(2:end)) / 2;
            
            frequencies = zeros(1, nBins);
            binCounts = zeros(1, nBins);
            binConf = zeros(1, nBins);
            
            % 2. Bucket Data (Identify which prediction falls into which probability bin)
            for i = 1:nBins
                idx = (scores >= binEdges(i)) & (scores < binEdges(i+1));
                
                % Inclusion fix: Put exactly 1.0 into the last bin
                if i == nBins
                    idx = idx | (scores == 1.0);
                end
                
                if any(idx)
                    % Calculate Average Actual Outcome (Reliability)
                    frequencies(i) = mean(targets(idx));
                    
                    % Track bin volume and average confidence for ECE calculation
                    binCounts(i) = sum(idx);
                    binConf(i) = mean(scores(idx));
                else
                    frequencies(i) = NaN; % Mark empty bins for the plotting engine
                end
            end
            
            % 3. Calculate ECE (Expected Calibration Error)
            % Formula: sum ( (bin_count / totalN) * |actual_freq - predicted_conf| )
            totalN = length(targets);
            ece = 0;
            for i = 1:nBins
                if binCounts(i) > 0
                    % Weight the error by how many points are in the bin
                    ece = ece + (binCounts(i) / totalN) * abs(frequencies(i) - binConf(i));
                end
            end
        end
    end
end
