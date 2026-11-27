classdef SVMClassifier < OceanStats.Base.IModelStrategy
    % SVMCLASSIFIER  Binary Support Vector Machine (soft-margin C-SVM).
    %
    %   Solves the dual QP using Sequential Minimal Optimization (SMO).
    %   Labels Y must be in {-1, +1} (auto-remaps {0,1} with a warning).
    %
    %   [REFACTORED 2026-03-13]
    %   This class now acts as a backward-compatible wrapper around 
    %   OceanStats.Supervised.KernelMethods.KernelMachine.
    %   It instantiates the generic KernelMachine with:
    %     - Loss: HingeLoss
    %     - Solver: SMOSolver
    %
    %   USAGE
    %   -----
    %   import OceanStats.Supervised.SVM.*
    %   svm   = SVMClassifier('C', 1.0, 'Kernel', Kernels.RBFKernel('Gamma', 0.5));
    %   model = svm.train(X_tr, Y_tr);
    %   Y_hat = svm.predict(model, X_te);
    %   err   = svm.evaluate(Y_te, Y_hat);
    %
    %   Author: Ocean Library | 2026-03-13

    % ------------------------------------------------------------------ %
    properties
        Name        string  = "SVM Classifier"
        SupportsAIC logical = false
    end

    properties
        % C  Regularisation constant. Large C → small margin allowance.
        C       (1,1) double {mustBePositive} = 1.0

        % Kernel  ISVMKernel strategy. Default: LinearKernel.
        Kernel

        % Tol  KKT violation tolerance (convergence stopping criterion).
        Tol     (1,1) double {mustBePositive} = 1e-3

        % MaxPasses  Max SMO passes over all training examples.
        MaxPasses (1,1) double {mustBePositive} = 200

        % Eps  Minimum alpha change to accept as a meaningful step.
        Eps     (1,1) double {mustBePositive} = 1e-8
        
        % KernelMachine Instance
        Engine  OceanStats.Supervised.KernelMethods.KernelMachine
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = SVMClassifier(options)
            arguments
                options.C         (1,1) double {mustBePositive} = 1.0
                options.Kernel    OceanStats.Supervised.SVM.Base.ISVMKernel = ...
                                  OceanStats.Supervised.SVM.Kernels.LinearKernel()
                options.Tol       (1,1) double {mustBePositive} = 1e-3
                options.MaxPasses (1,1) double {mustBePositive} = 200
                options.Eps       (1,1) double {mustBePositive} = 1e-8
            end
            obj.C         = options.C;
            obj.Kernel    = options.Kernel;
            obj.Tol       = options.Tol;
            obj.MaxPasses = options.MaxPasses;
            obj.Eps       = options.Eps;
            
            % Setup underlying machine
            obj.Engine = OceanStats.Supervised.KernelMethods.KernelMachine( ...
                'Kernel', obj.Kernel, ...
                'Loss', OceanStats.Loss.Classification.HingeLoss(), ...
                'Solver', OceanStats.Supervised.KernelMethods.SMOSolver( ...
                    'Tol', obj.Tol, ...
                    'MaxPasses', obj.MaxPasses, ...
                    'Eps', obj.Eps), ...
                'C', obj.C, ...
                'Eps', obj.Eps);
        end

        % -------------------------------------------------------------- %
        function model = train(obj, X, Y, W)
            arguments
                obj
                X  (:,:) double
                Y  (:,1) double
                W  (:,1) double = ones(size(Y)) %#ok<INUSA>
            end

            Y = obj.validateLabels_(Y);
            
            % Update engine properties if they were changed after construction
            obj.Engine.C = obj.C;
            obj.Engine.Kernel = obj.Kernel;
            obj.Engine.Eps = obj.Eps;
            
            model = obj.Engine.train(X, Y, W);
        end

        % -------------------------------------------------------------- %
        function Y_hat = predict(obj, model, X)
            Y_hat = obj.Engine.predict(model, X);
        end

        % -------------------------------------------------------------- %
        function err = evaluate(obj, Y_true, Y_pred)
            err = obj.Engine.evaluate(Y_true, Y_pred);
        end

        % -------------------------------------------------------------- %
        function k = getComplexity(obj, model)
            k = obj.Engine.getComplexity(model);
        end
    end

    % ================================================================== %
    methods (Access = private)
        function Y = validateLabels_(~, Y)
            % VALIDATELABELS_  Ensure labels are in {-1,+1}.
            classes = unique(Y);
            if numel(classes) ~= 2
                error('OceanStats:Supervised:SVM:SVMClassifier:BadLabels', ...
                      'SVMClassifier requires exactly 2 classes.');
            end
            if isequal(sort(classes(:)), [0;1])
                Y = 2*Y - 1;
                warning('OceanStats:Supervised:SVM:SVMClassifier:LabelRemapped', ...
                        'Labels {0,1} auto-remapped to {-1,+1}.');
            elseif ~isequal(sort(classes(:)), [-1;1])
                error('OceanStats:Supervised:SVM:SVMClassifier:BadLabels', ...
                      'Labels must be in {-1,+1} (or {0,1} for auto-remap).');
            end
        end
    end
end
