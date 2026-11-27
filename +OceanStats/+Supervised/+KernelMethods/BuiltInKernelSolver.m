classdef BuiltInKernelSolver < OceanStats.Supervised.KernelMethods.IKernelSolver
    % BUILTINKERNELSOLVER Uses MATLAB's fminunc to solve unconstrained
    % kernel machines (e.g. Kernel Logistic Regression).
    %
    %   Author: Ocean Library | 2026-03-14

    methods
        function [alpha, b, status] = solve(~, K, Y, loss, options)
            N = length(Y);
            if ~isfield(options, 'C'), options.C = 1.0; end
            C = options.C;
            
            % Initial guess: a = 0, b = 0
            % Let theta = [a; b] where a is N x 1
            theta0 = zeros(N + 1, 1);
            
            % Objective function
            % f(a, b) = K*a + b
            % J(a, b) = Loss(Y, f) + 1/(2*C*N) * a^T K a
            objFun = @(theta) OceanStats.Supervised.KernelMethods.BuiltInKernelSolver.computeObjGradient(theta, K, Y, loss, C, N);
            
            % Use quasi-newton so we don't have to provide the exact Hessian,
            % which may be singular if K is rank-deficient.
            opt = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                               'SpecifyObjectiveGradient', true, ...
                               'Display', 'off', 'MaxIterations', 2000);
            
            [theta_opt, ~, exitflag, output] = fminunc(objFun, theta0, opt);
            
            a = theta_opt(1:N);
            b = theta_opt(end);
            
            % KernelMachine uses (alpha .* Y)' * K + b for prediction.
            % So we map 'a' to 'alpha' such that alpha .* Y = a.
            % Since Y in {-1, 1}, we have alpha = a .* Y.
            alpha = a .* Y;
            
            status.solved = (exitflag > 0);
            status.iterations = output.iterations;
        end
        
        function name = getName(~)
            name = "Built-in Kernel Solver (fminunc)";
        end
    end
    
    methods (Static, Hidden)
        function [J, grad] = computeObjGradient(theta, K, Y, loss, C, N)
            a = theta(1:N);
            b = theta(end);
            
            f = K * a + b;
            
            % Remap {-1, 1} to {0, 1} for pure cross-entropy losses
            if contains(class(loss), 'BinaryBCE') && all(ismember(Y, [-1, 1]))
                Y_loss = (Y + 1) / 2;
            else
                Y_loss = Y;
            end
            
            L = loss.forward(Y_loss, f);
            
            % Regularization: 1/(2*C*N) a^T K a
            reg = (0.5 / (C * N)) * (a' * K * a);
            
            % Add a tiny ridge to 'a' directly to strictly convexify the problem 
            % globally, making fminunc extremely stable.
            eps_ridge = 1e-8 * (a' * a);
            
            J = L + reg + eps_ridge;
            
            if nargout > 1
                % loss.gradient returns the d(loss_i)/df_i for each sample
                g = loss.gradient(Y_loss, f);
                
                % grad_a = (1/N) * K * g + (1/(C*N)) * K * a + 2e-8 * a
                grad_a = K * (g / N + a / (C * N)) + 2e-8 * a;
                
                % grad_b = (1/N) * sum(g)
                grad_b = sum(g) / N;
                
                grad = [grad_a; grad_b];
            end
        end
    end
end
