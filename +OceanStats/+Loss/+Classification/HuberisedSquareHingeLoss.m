classdef HuberisedSquareHingeLoss < OceanStats.Loss.ILoss
    % HUBERISEDSQUAREHINGELOSS  Huberised Square Hinge Loss for binary classification.
    %
    %   CONVENTION:
    %     Y_true ∈ {-1, +1}  — signed class labels
    %     Y_pred             — raw decision score f(x)
    %
    %   FORMULA (per sample):
    %     margin m = y * f(x)
    %
    %     L = -4*m                 if m < -1
    %     L = max(0, 1 - m)^2      otherwise (m >= -1)
    %
    %     dL/df:
    %       -4y                    if m < -1
    %       -2y(1 - m)             if -1 <= m < 1
    %       0                      if m >= 1
    %
    %     d²L/df²:
    %       2                      if -1 <= m < 1
    %       0                      otherwise
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Classification.HuberisedSquareHingeLoss();
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %     h    = loss.hessian(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Classification.HingeLoss
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            
            margin = Y_true .* Y_pred;
            loss_vals = zeros(size(margin));
            
            idx_linear = margin < -1;
            idx_quad = margin >= -1;
            
            loss_vals(idx_linear) = -4 * margin(idx_linear);
            loss_vals(idx_quad) = max(0, 1 - margin(idx_quad)).^2;
            
            L = mean(loss_vals);
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            
            margin = Y_true .* Y_pred;
            g = zeros(size(margin));
            
            idx_linear = margin < -1;
            idx_quad = (margin >= -1) & (margin < 1);
            
            g(idx_linear) = -4 * Y_true(idx_linear);
            g(idx_quad) = -2 * Y_true(idx_quad) .* (1 - margin(idx_quad));
            % Note: g is already 0 where margin >= 1
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            
            margin = Y_true .* Y_pred;
            h = zeros(size(margin));
            
            idx_quad = (margin >= -1) & (margin < 1);
            h(idx_quad) = 2;
        end

        function n = name(~)
            n = "Huberised Square Hinge";
        end
    end

    methods (Static, Access = private)
        function checkLabels_(Y_true)
            if ~all(Y_true == 1 | Y_true == -1)
                error('OceanStats:Loss:HuberisedSquareHingeLoss:InvalidLabels', ...
                    'HuberisedSquareHingeLoss requires Y_true in {-1, +1}. Got other values.');
            end
        end
    end
end
