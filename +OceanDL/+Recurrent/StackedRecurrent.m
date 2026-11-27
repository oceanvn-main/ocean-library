classdef StackedRecurrent < OceanDL.nn.Module
    % STACKEDRECURRENT  Deep RNN: layer l reads the output sequence of layer
    %   l-1 (Goodfellow Sec. 10.5, Fig. 10.13a; Graves et al. 2013).
    %
    %   Between layers (training only):
    %     Dropout      p in [0,1)
    %     Variational  true: ONE dropout mask per sequence, reused at every
    %                  time step (Gal & Ghahramani 2016) -- dropping a unit
    %                  at some steps but not others would inject noise the
    %                  recurrence then amplifies
    %     Residual     true: y_l = layer_l(y_{l-1}) + y_{l-1} when sizes match
    %                  (eases optimisation of deep stacks)
    %   Layers may be RecurrentLayer or Bidirectional modules.

    properties (SetAccess = protected)
        Layers cell = {}
        Dropout (1,1) double = 0
        Variational (1,1) logical = true
        Residual (1,1) logical = false
    end

    methods
        function obj = StackedRecurrent(layers, opts)
            arguments
                layers cell
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.Variational (1,1) logical = true
                opts.Residual (1,1) logical = false
            end
            obj.Layers = layers;
            obj.Dropout = opts.Dropout;
            obj.Variational = opts.Variational;
            obj.Residual = opts.Residual;
        end

        function [Y, states] = forward(obj, X, opts)
            arguments
                obj
                X
                opts.Mask = []
            end
            Y = X;
            states = cell(1, numel(obj.Layers));
            for l = 1:numel(obj.Layers)
                if l > 1 && obj.Training && obj.Dropout > 0
                    Y = obj.dropoutBetween(Y);
                end
                [Z, states{l}] = obj.Layers{l}.forward(Y, 'Mask', opts.Mask);
                if obj.Residual && isequal(size(Z), size(Y))
                    Y = Z + Y;
                else
                    Y = Z;
                end
            end
        end
    end

    methods (Access = private)
        function Y = dropoutBetween(obj, Y)
            if ~obj.Variational
                Y = OceanDL.Autograd.F.dropout(Y, obj.Dropout, true);
                return;
            end
            raw = OceanDL.Autograd.Tensor.raw(Y);
            keep = cast(rand(size(raw, 1), size(raw, 2)) >= obj.Dropout, 'like', raw) / (1 - obj.Dropout);
            Y = Y .* keep;                          % (D x N) mask broadcast over all T steps
        end
    end
end
