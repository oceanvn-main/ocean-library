% exp_computer_vision_sandbox.m
% Launches the interactive Computer Vision Primitives Sandbox GUI.
%
% Author: Antigravity AI
% Date: 2026-06-19

fprintf('=== Launching Computer Vision Primitives Sandbox App ===\n');

% Ensure paths are set
rootDir = fullfile('c:', 'Ocean Library', 'Matlab code practice');
addpath(genpath(rootDir));
addpath(fullfile(rootDir, '+OceanViz', '+App', 'Computer Vision'));

app = ComputerVisionSandbox();
app.show();

fprintf('Computer Vision Sandbox loaded successfully.\n');
