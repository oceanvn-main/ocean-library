% exp_viewer_3d.m
% Launches the interactive 3D Primitives Viewer (Szeliski Exercise 2.3).
%
% Author: Antigravity AI
% Date: 2026-06-10

fprintf('=== Launching 3D Primitives Viewer App ===\n');

% Ensure paths are set
rootDir = fullfile('c:', 'Ocean Library', 'Matlab code practice');
addpath(genpath(rootDir));
addpath(fullfile(rootDir, '+OceanViz', '+App', 'Computer Vision'));

app = Viewer3D();
app.show();

fprintf('3D Primitives Viewer loaded successfully.\n');
