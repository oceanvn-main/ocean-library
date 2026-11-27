classdef ParameterGroup < handle
    % PARAMETERGROUP  Container for parameter subset indices and group hyperparameters.
    %
    %   PyTorch-style parameter grouping allowing custom learning rates,
    %   weight decay rates, or multiplier factors for specific layers
    %   (e.g., fine-tuning pretrained vision backbones vs new NLP heads).
    %
    %   USAGE:
    %     g1 = OceanStats.Optim.ParameterGroup('Backbone', 1:100, 'LR', 1e-5, 'WeightDecay', 0.01);
    %     g2 = OceanStats.Optim.ParameterGroup('Head', 101:150, 'LR', 1e-3, 'WeightDecay', 0.00);
    %
    %   See also: OceanStats.Optim.VectorizedOptimizerEngine
    %             OceanStats.Optim.IOptimizerStrategy

    properties
        Name         (1,:) char   = 'Group'  % Group identifier
        ParamIndices (:,1) double = []       % 1-based indices into 1D theta_flat
        LR           (1,1) double = 0.001    % Learning rate for this group
        WeightDecay  (1,1) double = 0.0      % Decoupled weight decay rate
        LRMultiplier (1,1) double = 1.0      % Multiplier for dynamic schedulers
    end

    methods
        function obj = ParameterGroup(name, paramIndices, opts)
            % PARAMETERGROUP  Constructor.
            %
            %   g = ParameterGroup('Layer1', 1:12, 'LR', 0.01, 'WeightDecay', 0.01)
            arguments
                name                 (1,:) char
                paramIndices         (:,1) double
                opts.LR              (1,1) double = 0.001
                opts.WeightDecay     (1,1) double = 0.00
                opts.LRMultiplier    (1,1) double = 1.0
            end
            obj.Name         = name;
            obj.ParamIndices = paramIndices;
            obj.LR           = opts.LR;
            obj.WeightDecay  = opts.WeightDecay;
            obj.LRMultiplier = opts.LRMultiplier;
        end

        function n = numParams(obj)
            % NUMPARAMS  Total number of parameters in this group.
            n = numel(obj.ParamIndices);
        end

        function effectiveLR = getEffectiveLR(obj)
            % GETEFFECTIVELR  Calculates effective learning rate considering multiplier.
            effectiveLR = obj.LR * obj.LRMultiplier;
        end
    end
end
