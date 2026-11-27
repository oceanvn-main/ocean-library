% =========================================================================
% Exercise 3.5: Difference Keying (Parts 1 and 2)
% =========================================================================

clc; clear; close all;
% Define the path to the input dataset based on the screenshot
inputDir = 'C:\Ocean Library\Matlab code practice\Data\videos\dataset\baseline\highway\input';

% Check if directory exists (Update this path if running on a different machine)
if ~isfolder(inputDir)
    error('Directory not found. Please ensure the path to the dataset is correct.');
end

% Get a list of all jpg files in the directory
imageFiles = dir(fullfile(inputDir, '*.jpg'));
numFiles = length(imageFiles);

if numFiles == 0
    error('No images found in the specified directory.');
end

% =========================================================================
% STEP 1: Compute the Background Model (Median and Robust Variance)
% =========================================================================
disp('Step 1: Computing background model...');

% Number of frames used to train the background model. 
% 100 frames is usually sufficient to capture a reliable median background.
numTrainFrames = min(100, numFiles); 

% Read the first frame to get image dimensions
firstFrame = imread(fullfile(inputDir, imageFiles(1).name));
[height, width, channels] = size(firstFrame);

% Pre-allocate a 4D array to hold the training sequence in memory
% (Height x Width x Channels x NumberOfFrames)
trainData = zeros(height, width, channels, numTrainFrames, 'double');

% Load the training frames
for i = 1:numTrainFrames
    imgPath = fullfile(inputDir, imageFiles(i).name);
    trainData(:, :, :, i) = double(imread(imgPath));
end

% Compute the background model using the library class (MedianMAD method)
bs = OceanMath.Image.Resampling.BackgroundSubtractor(trainData, 'Method', 'MedianMAD');

bgMedian = bs.CentralTendency;

% Convert the model's Variance back to standard deviation equivalent
bgStdDev = sqrt(bs.Variance);

% Apply Variance Bounding (Ceiling) to prevent contaminated training frames
% from blinding the detector in high-variance regions (like the top road entry).
maxStdDev = 20.0;
bgStdDev = min(bgStdDev, maxStdDev);

% Add a small constant (epsilon) to the standard deviation to prevent 
% dividing by zero or being overly sensitive in completely static dark areas
epsilon = 2.0;
bgStdDev = bgStdDev + epsilon;

disp('Background model calculated successfully.');

% Plot the background model after step 1
figBg = figure('Name', 'Step 1: Background Model', 'Position', [100, 100, 950, 450]);

% Display temporal median
axBg1 = subplot(1, 2, 1);
imshow(uint8(bgMedian));
title(axBg1, 'Background Central Tendency (Temporal Median)', 'FontSize', 12);
xlabel(axBg1, 'Outliers (cars) successfully removed', 'FontSize', 9);

% Display standard deviation map
axBg2 = subplot(1, 2, 2);
% Mean across color channels to show general variance map
stdMap = mean(bgStdDev, 3);
imshow(stdMap, []);
colormap(axBg2, 'parula');
colorbar(axBg2);
title(axBg2, 'Robust Standard Deviation Map (MAD + Epsilon)', 'FontSize', 12);
xlabel(axBg2, 'Higher values represent dynamic areas', 'FontSize', 9);

drawnow; % Force display before starting step 2

% Load new background image for Part c
bgPath = 'C:\Ocean Library\Matlab code practice\Data\images\composited_highway_background.jpg';
if ~isfile(bgPath)
    error('Composited highway background image not found at: %s', bgPath);
end
newBg = double(imread(bgPath));
if size(newBg, 1) ~= height || size(newBg, 2) ~= width
    newBg = imresize(newBg, [height, width]);
end

% Multi-object centroid tracker setup for Part d
next_track_id = 1;
tracks = struct('id', {}, 'centroid', {}, 'prev_y', {}, 'crossed', {}, 'lost', {}, 'path', {});
all_paths = {};
car_count = 0;
midline_y = height * 0.55;
max_dist = 100; % pixels (increased to handle high velocity/displacement near camera)
max_lost = 15;  % frames (increased to tolerate detection dropouts in low contrast frames)

% Number of standard deviations a pixel must deviate to be considered foreground
k = 3;

% Check if display is available (Desktop vs Headless/Batch mode)
hasDisplay = feature('ShowFigureWindows');

% Create a figure for visualization (invisible if headless to allow batch processing to complete)
if hasDisplay
    fig = figure('Name', 'Difference Keying Results (Parts a, b, c, d)', 'Position', [80, 80, 1100, 850]);
else
    fig = figure('Name', 'Difference Keying Results (Parts a, b, c, d)', 'Position', [80, 80, 1100, 850], 'Visible', 'off');
end

% Process the frames following the training sequence
for i = (numTrainFrames + 1) : numFiles
    % Check if the user closed the figure window (clicked the X)
    % Only abort if running in an interactive session with figure display
    if hasDisplay && ~ishandle(fig)
        disp('Figure window closed. Terminating frame classification.');
        break;
    end
    
    % Read the new frame
    imgPath = fullfile(inputDir, imageFiles(i).name);
    currentFrame = double(imread(imgPath));
    
    % Calculate absolute difference between current frame and background median
    pixelDiff = abs(currentFrame - bgMedian);
    
    % Classify pixels: 
    % A pixel is foreground (1) if its difference is > k * StandardDeviation
    isForegroundChannels = pixelDiff > (k * bgStdDev);
    
    % Combine RGB channels: If a pixel changes significantly in Red, Green, OR Blue, 
    % it is considered a foreground object.
    fgMask = isForegroundChannels(:,:,1) | ...
             isForegroundChannels(:,:,2) | ...
             isForegroundChannels(:,:,3);
         
    % Set background pixels to 0 (Black) as requested in Step 2
    % Initialize output image with zeros
    outputFrame = zeros(height, width, channels, 'uint8');
    
    % Extract original channels
    R = currentFrame(:,:,1);
    G = currentFrame(:,:,2);
    B = currentFrame(:,:,3);
    
    % Zero out the background pixels
    R(~fgMask) = 0;
    G(~fgMask) = 0;
    B(~fgMask) = 0;
    
    % Reconstruct the output image
    outputFrame(:,:,1) = uint8(R);
    outputFrame(:,:,2) = uint8(G);
    outputFrame(:,:,3) = uint8(B);
    
    % --- Part d: Morphology & Connected Components Tracking ---
    % Manually compute intermediate morphological steps to display them later
    se = strel('disk', 5);
    openedMask = imopen(fgMask, se);
    closedMask = imclose(openedMask, se);
    cleanMask  = closedMask;
    
    % Connected components and bounding boxes for visualization
    cc = bwconncomp(cleanMask);
    stats = regionprops(cc, 'BoundingBox', 'Centroid', 'Area');
    
    % Centroid Tracker Update
    valid_indices = find([stats.Area] >= 25);
    num_detections = length(valid_indices);
    detections = zeros(num_detections, 2);
    for d = 1:num_detections
        detections(d, :) = stats(valid_indices(d)).Centroid;
    end
    
    num_tracks = length(tracks);
    matched_detections = false(num_detections, 1);
    matched_tracks = false(num_tracks, 1);
    
    if num_tracks > 0 && num_detections > 0
        dist_matrix = zeros(num_tracks, num_detections);
        for t = 1:num_tracks
            for d = 1:num_detections
                dist_matrix(t, d) = norm(tracks(t).centroid - detections(d, :));
            end
        end
        
        while true
            [min_val, min_idx] = min(dist_matrix(:));
            if min_val > max_dist || isinf(min_val)
                break;
            end
            [t, d] = ind2sub(size(dist_matrix), min_idx);
            
            tracks(t).centroid = detections(d, :);
            tracks(t).path = [tracks(t).path; detections(d, :)];
            tracks(t).lost = 0;
            matched_tracks(t) = true;
            matched_detections(d) = true;
            
            prev_y = tracks(t).prev_y;
            curr_y = tracks(t).centroid(2);
            if ~tracks(t).crossed
                if (prev_y <= midline_y && curr_y > midline_y) || ...
                   (prev_y >= midline_y && curr_y < midline_y)
                    car_count = car_count + 1;
                    tracks(t).crossed = true;
                    fprintf('[CarCounter] Track %d crossed midline! Total: %d\n', tracks(t).id, car_count);
                end
            end
            tracks(t).prev_y = curr_y;
            
            dist_matrix(t, :) = Inf;
            dist_matrix(:, d) = Inf;
        end
    end
    
    for t = 1:num_tracks
        if ~matched_tracks(t)
            tracks(t).lost = tracks(t).lost + 1;
        end
    end
    
    for d = 1:num_detections
        if ~matched_detections(d)
            new_track = struct(...
                'id', next_track_id, ...
                'centroid', detections(d, :), ...
                'prev_y', detections(d, 2), ...
                'crossed', false, ...
                'lost', 0, ...
                'path', detections(d, :) ...
            );
            tracks = [tracks; new_track]; %#ok<AGROW>
            next_track_id = next_track_id + 1;
        end
    end
    
    % Save completed paths
    removed_tracks = tracks([tracks.lost] > max_lost);
    for r = 1:length(removed_tracks)
        all_paths{end+1} = removed_tracks(r).path; %#ok<AGROW>
    end
    tracks = tracks([tracks.lost] <= max_lost);
    
    % --- Part c: Composite over new background ---
    alpha = double(cleanMask);
    compositeFrame = zeros(height, width, channels);
    for c = 1:3
        compositeFrame(:,:,c) = alpha .* currentFrame(:,:,c) + (1.0 - alpha) .* newBg(:,:,c);
    end
    
    % Visualization (2x2 layout)
    if ishandle(fig)
        set(groot, 'CurrentFigure', fig); % Explicitly target only the results window
        
        % Subplot 1: Original Input Frame
        subplot(2, 2, 1);
        imshow(uint8(currentFrame));
        title(sprintf('Original Frame %d', i));
        
        % Subplot 2: Classification Mask
        subplot(2, 2, 2);
        imshow(fgMask);
        title('Foreground Binary Mask');
        
        % Subplot 3: Part c Alpha Compositing
        subplot(2, 2, 3);
        imshow(uint8(compositeFrame));
        title('Composite over New Background');
        
        % Subplot 4: Part d Multi-Object Tracking & Counter
        subplot(2, 2, 4);
        imshow(uint8(currentFrame));
        hold on;
        
        % Draw midline
        line([1, width], [midline_y, midline_y], 'Color', [1.0, 0.85, 0.0], 'LineWidth', 2, 'LineStyle', '--');
        
        % Draw active tracks
        for t = 1:length(tracks)
            plot(tracks(t).centroid(1), tracks(t).centroid(2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
            text(tracks(t).centroid(1)+5, tracks(t).centroid(2)-5, sprintf('#%d', tracks(t).id), 'Color', 'g', 'FontWeight', 'bold');
        end
        
        % Draw active bounding boxes
        for d = 1:length(valid_indices)
            box = stats(valid_indices(d)).BoundingBox;
            rectangle('Position', box, 'EdgeColor', 'c', 'LineWidth', 1.5);
        end
        
        % Display counter text
        text(10, 20, sprintf('Cars: %d', car_count), 'Color', 'y', 'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
        title('Centroid Multi-Object Tracker');
        hold off;
        
        drawnow; % Force MATLAB to update the figure window
        
        % Save the 2x2 dashboard figure at frame 900 to Results folder
        if i == 900
            results_dir = 'C:\Ocean Library\Matlab code practice\Experiments\Scripts\Computer Vision\Results';
            if ~exist(results_dir, 'dir'), mkdir(results_dir); end
            out_fig = fullfile(results_dir, 'ex35_parts_c_d_results.png');
            exportgraphics(fig, out_fig, 'Resolution', 150);
            fprintf('  Saved Parts c & d dashboard to: %s\n', out_fig);
            
            % Generate and save morphology steps comparison plot
            figMorph = figure('Name', 'Morphological Cleanup Steps (Frame 900)', 'Position', [150, 150, 1200, 380]);
            
            axM1 = subplot(1, 3, 1);
            imshow(fgMask);
            title(axM1, '1. Raw Foreground Mask', 'FontSize', 12);
            xlabel(axM1, 'Contains isolated noise (waving leaves)', 'FontSize', 9);
            
            axM2 = subplot(1, 3, 2);
            imshow(openedMask);
            title(axM2, '2. After Morphological Opening', 'FontSize', 12);
            xlabel(axM2, 'Isolated noise removed', 'FontSize', 9);
            
            axM3 = subplot(1, 3, 3);
            imshow(closedMask);
            title(axM3, '3. After Morphological Closing', 'FontSize', 12);
            xlabel(axM3, 'Hollow spots inside vehicles filled', 'FontSize', 9);
            
            drawnow;
            out_morph = fullfile(results_dir, 'ex35_morphology_steps.png');
            exportgraphics(figMorph, out_morph, 'Resolution', 150);
            fprintf('  Saved morphology steps comparison to: %s\n', out_morph);
            close(figMorph);
        end
    end
    
    % Pause briefly so the animation is visible
    % Remove or reduce this if you want it to process at maximum speed
    pause(0.01); 
end