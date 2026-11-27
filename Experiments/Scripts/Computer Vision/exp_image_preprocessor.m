% exp_image_preprocessor.m
% Launches the interactive Image Preprocessing Sandbox GUI.
%
% Author: Antigravity AI
% Date: 2026-06-11

fprintf('=== Launching Image Preprocessing Sandbox App ===\n');

% Ensure paths are set
rootDir = fullfile('c:', 'Ocean Library', 'Matlab code practice');
addpath(genpath(rootDir));
addpath(fullfile(rootDir, '+OceanViz', '+App', 'Computer Vision'));

app = ImagePreprocessor();
app.show();

fprintf('Image Preprocessing Sandbox loaded successfully.\n');
