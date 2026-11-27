classdef ImageRegularizationEnergy < OceanMath.Variational.VariationalEnergy
    % IMAGEREGULARIZATIONENERGY  2D Image Regularization Energy.
    %
    %   Subclass of OceanMath.Variational.VariationalEnergy specialized for
    %   2D spatial image grids using DifferenceOperator2D.
    %
    %   Input Representations:
    %     - f              : 2D double matrix (rows x cols) representing estimated image state
    %     - g              : 2D double matrix (rows x cols) representing observed noisy image
    %     - penalty        : IRobustPenalty strategy (e.g. L1Penalty, HuberPenalty, GNCPenalty)
    %     - Lambda         : Scalar double specifying regularization strength (tradeoff factor)
    %     - SpatialWeights : (Optional) Struct with fields:
    %                           .X -> (rows x cols-1) horizontal spatial weights s_x
    %                           .Y -> (rows-1 x cols) vertical spatial weights s_y
    %
    %   Output Representations:
    %     - E_reg          : Scalar double -> total spatial regularization energy
    %     - E              : Scalar double -> total combined objective energy
    %                           E(f) = (1/2) * ||f - g||_F^2 + Lambda * E_reg(f)
    %     - D              : Sparse matrix (K x N) -> 2D spatial difference operator matrix,
    %                           where N = rows*cols and K = rows*(cols-1) + (rows-1)*cols
    %     - W_diag         : Sparse matrix (K x K) -> diagonal IRLS weight matrix
    %
    %   Package: OceanMath.Image.Variational
    %   Author: Ocean Library  |  2026-08-05

    properties
        SpatialWeights = []  % Spatially-varying weights s_x and s_y (optional struct)
    end

    methods
        function obj = ImageRegularizationEnergy(penalty, options)
            % IMAGEREGULARIZATIONENERGY Constructor.
            %
            %   Inputs:
            %     penalty — IRobustPenalty instance (default L1Penalty)
            %     Name-Value Options:
            %       'Lambda'         — (1,1) double {mustBeNonnegative} (default 0.1)
            %       'SpatialWeights' — Struct with .X and .Y matrices, or []
            arguments
                penalty  = OceanMath.Regularization.L1Penalty()
                options.Lambda         (1,1) double {mustBeNonnegative} = 0.1
                options.SpatialWeights                                  = []
            end

            % Initialize base VariationalEnergy with DifferenceOperator2D
            obj@OceanMath.Variational.VariationalEnergy( ...
                OceanMath.Operators.DifferenceOperator2D(), ...
                penalty, ...
                'Lambda', options.Lambda);

            obj.SpatialWeights = options.SpatialWeights;
        end

        function E_reg = regularizationEnergy(obj, f)
            % REGULARIZATIONENERGY Compute 2D spatial regularization energy.
            %
            %   Input:
            %     f — (rows x cols) 2D double image matrix
            %   Output:
            %     E_reg — Scalar double energy value
            if isempty(obj.SpatialWeights)
                E_reg = regularizationEnergy@OceanMath.Variational.VariationalEnergy(obj, f);
            else
                dx = diff(f, 1, 2);
                dy = diff(f, 1, 1);
                sx = obj.SpatialWeights.X;
                sy = obj.SpatialWeights.Y;
                E_reg = sum(sx(:) .* obj.Penalty.evaluate(dx(:))) + ...
                        sum(sy(:) .* obj.Penalty.evaluate(dy(:)));
            end
        end
    end
end
