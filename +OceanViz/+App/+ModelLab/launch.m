function lab = launch(model, opts)
% LAUNCH  Open the Model Lab: one app for CNN interpretability and for
%   generative models (DDPM, VAE, ...), whose tabs adapt to the loaded model.
%
%   lab = OceanViz.App.ModelLab.launch();                       % empty; pick a model in the top bar
%   lab = OceanViz.App.ModelLab.launch("Ideal DDPM - shapes 16x16 (no training)");
%   lab = OceanViz.App.ModelLab.launch('...\Results\ddpm_textbook\ddpm_shapes16.mat');   % a ModelCard
%   lab = OceanViz.App.ModelLab.launch("Dummy CNN (random weights)");
%
%   Book references (tab tooltips): Lai et al., "The Principles of Diffusion
%   Models" -- forward process Sec. 2.2.1, reverse process Sec. 2.2.2-2.2.6,
%   L_simple Eq. (2.2.10), classifier-free guidance Sec. 8.3; VAE Sec. 2.1.
arguments
    model = []
    opts.Visible (1,1) logical = true
end
lab = OceanViz.App.ModelLab.Shell('Visible', opts.Visible);
if ~isempty(model), lab.loadModel(model); end
end
