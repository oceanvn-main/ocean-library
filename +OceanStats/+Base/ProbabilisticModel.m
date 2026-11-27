classdef (Abstract) ProbabilisticModel < handle
    % PROBABILISTICMODEL Abstract base class for models in OceanStats.
    %
    % Defines the interface required for:
    % 1. Model fitting and prediction.
    % 2. Integration with OceanStats.Data.BootstrapSampler.
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    methods (Abstract)
        % FIT Fits the model to data.
        % Inputs:
        %   data - (N x D) Data matrix (usually [X, Y])
        % Outputs:
        %   params - Struct or object containing fitted parameters.
        params = fit(obj, data)

        % PREDICT Predicts response for new input.
        % Inputs:
        %   x      - Input data (N x D_x)
        %   params - (Optional) Parameters to use for prediction. 
        %            If omitted, use internally stored fitted parameters.
        y_hat = predict(obj, x, params)
    end

    methods (Abstract)       
        % BOOT_GEN Generates synthetic data from model parameters.
        % Inputs:
        %   params - Parameters to generate from.
        %   n      - Number of samples to generate.
        % Outputs:
        %   data_new - (n x D) Synthetic data matrix.
        data_new = boot_gen(obj, params, n)
    end
end

