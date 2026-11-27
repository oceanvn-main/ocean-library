classdef KernelMachine < OceanStats.Base.IModelStrategy
    % KERNELMACHINE  Unified Engine for Kernel Methods (SVM, KRR, etc.)
    %
    %   Evaluates the dual function f(x) = sum_i alpha_i K(x, x_i) + b.
    %   The optimization of alpha and b is dictated completely by the injected
    %   loss function and the injected solver strategy (Representer Theorem).
    %
    %   See ESL Section 12.3.7 for theoretical basis.
    %   Author: Ocean Library | 2026-03-13

    properties
        Name        string  = "Kernel Machine"
        SupportsAIC logical = false

        Kernel   % OceanStats.Supervised.SVM.Base.ISVMKernel
        Loss     % OceanStats.Loss.ILoss
        Solver   % OceanStats.Supervised.KernelMethods.IKernelSolver
        
        % Regularisation constant. Interpretation depends on solver/loss.
        C (1,1) double {mustBePositive} = 1.0
        
        % Eps: threshold to treat alpha as a support vector.
        Eps (1,1) double {mustBePositive} = 1e-8
    end

    methods
        function obj = KernelMachine(options)
            arguments
                options.Kernel OceanStats.Supervised.SVM.Base.ISVMKernel = ...
                               OceanStats.Supervised.SVM.Kernels.LinearKernel()
                options.Loss   OceanStats.Loss.ILoss = ...
                               OceanStats.Loss.Classification.HingeLoss()
                options.Solver OceanStats.Supervised.KernelMethods.IKernelSolver = ...
                               OceanStats.Supervised.KernelMethods.SMOSolver()
                options.C      (1,1) double {mustBePositive} = 1.0
                options.Eps    (1,1) double {mustBePositive} = 1e-8
            end
            obj.Kernel = options.Kernel;
            obj.Loss   = options.Loss;
            obj.Solver = options.Solver;
            obj.C      = options.C;
            obj.Eps    = options.Eps;
        end

        function model = train(obj, X, Y, W)
            % TRAIN  Fit the Kernel Machine.
            arguments
                obj
                X  (:,:) double
                Y  (:,1) double
                W  (:,1) double = ones(size(Y)) %#ok<INUSA> 
            end

            N = size(X, 1);
            
            % 1. Pre-compute Kernel matrix
            K = obj.Kernel.compute(X, X);

            % 2. Delegate optimization to the Solver
            options.C = obj.C;
            [alpha, b, status] = obj.Solver.solve(K, Y, obj.Loss, options);

            if isfield(status, 'solved') && ~status.solved
                warning('OceanStats:KernelMethods:KernelMachine:SolverConvergence', ...
                        'The solver reported that it did not fully converge.');
            end

            % 3. Extract Support Vectors (alphas > eps)
            % In KRR, all are support vectors. In SVM, only a subset.
            svMask = abs(alpha) > obj.Eps;
            
            model.alpha    = alpha;
            model.sv_X     = X(svMask, :);
            model.sv_y     = Y(svMask);     % Useful for SMO logic / prediction
            model.sv_alpha = alpha(svMask); 
            model.b        = b;
            model.kernel   = obj.Kernel;
            model.C        = obj.C;
            model.nSV      = sum(svMask);
            
            % Store loss class to determine output mode (classification vs regression)
            model.lossClass = class(obj.Loss);
        end

        function Y_hat = predict(~, model, X)
            % PREDICT  f(x) = sum_i alpha_i y_i K(x, x_i) + b
            % 
            % Note: SMO calculates the objective as sum_i alpha_i y_i K, 
            % so we include y_i in the multiplier here. For pure regression
            % solvers (not SMO), y_i is usually implicitly 1 in the formula.
            arguments
                ~
                model struct
                X     (:,:) double
            end
            
            K = model.kernel.compute(model.sv_X, X); % (nSV x M)
            
            % Some solvers (like standard SVM SMO) absorb Y into alpha 
            % formulation during prediction: sum (a_i * y_i * K)
            if isfield(model, 'sv_y') && ~isempty(model.sv_y)
                coeff = model.sv_alpha .* model.sv_y;
            else
                coeff = model.sv_alpha;
            end
            
            f = (coeff' * K)' + model.b; % (M x 1)
            
            % Dispatch output mapping based on Loss function
            if contains(model.lossClass, 'Classification')
                Y_hat = sign(f);
                Y_hat(Y_hat == 0) = 1; 
            else
                Y_hat = f;
            end
        end

        function err = evaluate(obj, Y_true, Y_pred)
            % EVALUATE  Delegate to the Loss.forward method (raw risk)
            %  or calculate accuracy for classification.
            arguments
                obj
                Y_true (:,1) double
                Y_pred (:,1) double
            end
            if contains(class(obj.Loss), 'Classification')
                err = mean(Y_true ~= Y_pred);
            else
                err = obj.Loss.forward(Y_true, Y_pred);
            end
        end

        function k = getComplexity(~, model)
            k = model.nSV;
        end
    end
end
