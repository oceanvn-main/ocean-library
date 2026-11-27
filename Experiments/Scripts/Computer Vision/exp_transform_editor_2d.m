% exp_transform_editor_2d.m
% Launches the interactive 2D Transform Editor (Szeliski Exercise 2.2).
%
% Author: Antigravity AI
% Date: 2026-06-10

fprintf('=== Launching 2D Transform Editor App ===\n');

% Ensure paths are set
rootDir = fullfile('c:', 'Ocean Library', 'Matlab code practice');
addpath(genpath(rootDir));
addpath(fullfile(rootDir, '+OceanViz', '+App', 'Computer Vision'));

app = TransformEditor2D();
app.show();

fprintf('2D Transform Editor loaded successfully.\n');
