classdef BatchNormLayer < OceanDL.Norm.INormalization
    % BATCHNORMLAYER  Batch Normalization (Ioffe & Szegedy, 2015).
    %
    %   Normalizes across the BATCH dimension N for each feature independently.
    %   During training: uses current batch mean/var; updates running stats.
    %   During inference: uses frozen running mean/var (no batch dependency).
    %
    %   FORMULA (training, per feature d_i):
    %     mu_i     = mean(Z(i, :))               scalar over N
    %     var_i    = var(Z(i, :))  + eps          scalar
    %     Z_hat_i  = (Z(i,:) - mu_i) / sqrt(var_i)   (1 x N)
    %     Z_out_i  = gamma_i * Z_hat_i + beta_i       (1 x N)
    %
    %   Running stats (exponential moving average, momentum = 0.1):
    %     RunningMean = (1-m)*RunningMean + m*batch_mean
    %     RunningVar  = (1-m)*RunningVar  + m*batch_var
    %
    %   Ref: Ioffe, S., & Szegedy, C. (2015). Batch normalization.
    %        ICML 2015. arXiv:1502.03167
    %
    %   Usage:
    %     norm = OceanDL.Norm.BatchNormLayer(64);
    %     Z_norm = norm.normalize(Z, 'train');   % updates running stats
    %     Z_norm = norm.normalize(Z, 'infer');   % uses frozen running stats

    properties
        Gamma       (:,1) double    % Scale   (d x 1), init 1
        Beta        (:,1) double    % Shift   (d x 1), init 0
        Epsilon     (1,1) double = 1e-5
        Momentum    (1,1) double = 0.1  % For running stat EMA
    end

    properties (SetAccess = private)
        RunningMean (:,1) double    % (d x 1), init 0
        RunningVar  (:,1) double    % (d x 1), init 1
    end

    methods
        function obj = BatchNormLayer(d, opts)
            % BATCHNORMLAYER  Construct for a layer of width d.
            %
            %   norm = BatchNormLayer(64)
            %   norm = BatchNormLayer(64, 'Momentum', 0.01)
            arguments
                d              (1,1) double {mustBePositive}
                opts.Epsilon   (1,1) double = 1e-5
                opts.Momentum  (1,1) double = 0.1
            end
            obj.Gamma       = ones(d, 1);
            obj.Beta        = zeros(d, 1);
            obj.Epsilon     = opts.Epsilon;
            obj.Momentum    = opts.Momentum;
            obj.RunningMean = zeros(d, 1);
            obj.RunningVar  = ones(d, 1);
        end

        function Z_out = normalize(obj, Z_in, mode)
            % NORMALIZE  Apply batch normalization.
            %   mode = 'train' — uses batch stats, updates running stats
            %   mode = 'infer' — uses frozen running stats (no batch bias)
            if nargin < 3, mode = 'train'; end

            if strcmp(mode, 'train')
                mu  = mean(Z_in, 2);                    % (d x 1)
                v   = var(Z_in, 0, 2) + obj.Epsilon;   % (d x 1)

                % Update running stats
                m = obj.Momentum;
                obj.RunningMean = (1 - m) * obj.RunningMean + m * mu;
                obj.RunningVar  = (1 - m) * obj.RunningVar  + m * v;
            else
                mu = obj.RunningMean;
                v  = obj.RunningVar + obj.Epsilon;
            end

            Z_hat = (Z_in - mu) ./ sqrt(v);             % (d x N) broadcast
            Z_out = obj.Gamma .* Z_hat + obj.Beta;      % (d x N)
        end

        function p = parameters(obj)
            p.gamma = obj.Gamma;
            p.beta  = obj.Beta;
        end

        function s = name(~)
            s = 'batchnorm';
        end
    end
end
