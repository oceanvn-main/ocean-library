classdef SMOSolver < OceanStats.Supervised.KernelMethods.IKernelSolver
    % SMOSOLVER  Sequential Minimal Optimization (SMO) algorithm for SVMs.
    %
    %   Solves the dual QP using SMO for L1 Hinge Loss constraints.
    %   This solver is specifically bound to the box constraint 0 <= a_i <= C.
    %   If the injected loss is not compatible (e.g., MSE or Smooth Hinge), 
    %   the SMO solver cannot be used.
    %
    %   Reference: Platt (1998), "Fast Training of SVMs using SMO"
    %   Author: Ocean Library | 2026-03-13

    properties
        Tol       (1,1) double {mustBePositive} = 1e-3
        MaxPasses (1,1) double {mustBePositive} = 200
        Eps       (1,1) double {mustBePositive} = 1e-8
    end

    methods
        function obj = SMOSolver(options)
            arguments
                options.Tol       (1,1) double {mustBePositive} = 1e-3
                options.MaxPasses (1,1) double {mustBePositive} = 1000
                options.Eps       (1,1) double {mustBePositive} = 1e-8
            end
            obj.Tol       = options.Tol;
            obj.MaxPasses = options.MaxPasses;
            obj.Eps       = options.Eps;
        end

        function [alpha, b, status] = solve(obj, K, Y, loss, options)
            % SOLVE  Run SMO to find alpha and b.
            arguments
                obj
                K       (:,:) double
                Y       (:,1) double
                loss    OceanStats.Loss.ILoss
                options struct = struct('C', 1.0)
            end
            
            % Enforce that SMO is only used with compatible losses
            if ~isa(loss, 'OceanStats.Loss.Classification.HingeLoss')
                error('OceanStats:KernelMethods:SMOSolver:IncompatibleLoss', ...
                    'SMOSolver currently only supports exact HingeLoss.');
            end
            
            N = length(Y);
            if ~isfield(options, 'C'), options.C = 1.0; end
            C = options.C;

            alpha   = zeros(N, 1);
            b       = 0;
            passes  = 0;
            E       = zeros(N, 1) - Y; % initially f=0 -> E=-Y

            totalSteps = 0;
            maxTotalSteps = obj.MaxPasses * N * 10;
            
            % Heuristically scale tolerance based on Kernel diagonal values
            % Polynomial kernels can hit 10^4 or higher; RBF is 1.0.
            K_trace = sum(diag(K));
            scaledEps = max(obj.Eps, obj.Eps * (K_trace / N));
            
            while passes < obj.MaxPasses && totalSteps < maxTotalSteps
                numChanged = 0;
                
                for i2 = 1:N
                    E(i2) = obj.decFunc_(alpha, Y, K, b, i2) - Y(i2);
                    r2    = E(i2) * Y(i2);
                    
                    % KKT Violated?
                    % Standard: r2 < -Tol (alpha < C) or r2 > Tol (alpha > 0)
                    % For numerical stability with large polynomials, scale Tol.
                    tol_i = obj.Tol * max(1, abs(E(i2)));

                    if (r2 < -tol_i && alpha(i2) < C) || ...
                       (r2 >  tol_i && alpha(i2) > 0)
                        
                        % Heuristic: select i1 as argmax |E_i1 - E_i2|
                        if E(i2) > 0
                            [~, i1] = min(E);
                        else
                            [~, i1] = max(E);
                        end

                        [alpha, b, E, stepped] = ...
                            obj.takeStep_(i1, i2, alpha, Y, K, b, E, N, C, scaledEps);
                        totalSteps = totalSteps + 1;

                        if stepped
                            numChanged = numChanged + 1;
                            continue;
                        end

                        % Fallback 1: non-bound alphas
                        % If N is large, randomize carefully. If step size (eta) is unstable, skip.
                        nonBound = find(alpha > obj.Eps & alpha < C - obj.Eps);
                        % Start from a random off-set to avoid deterministic cycles
                        if ~isempty(nonBound)
                            startIdx = randi(numel(nonBound));
                            perm = [nonBound(startIdx:end); nonBound(1:startIdx-1)];
                            for i1 = perm(:)'
                                if i1 == i2; continue; end
                                [alpha, b, E, stepped] = ...
                                    obj.takeStep_(i1, i2, alpha, Y, K, b, E, N, C, scaledEps);
                                totalSteps = totalSteps + 1;
                                if stepped
                                    numChanged = numChanged + 1;
                                    break;
                                end
                            end
                            if stepped; continue; end
                        end
                        
                        % Fallback 2: all alphas
                        % If first fallback fails, scan all alphas, but break if we cycle uselessly
                        startIdx = randi(N);
                        perm = [startIdx:N, 1:startIdx-1];
                        for i1 = perm
                            if i1 == i2; continue; end
                            [alpha, b, E, stepped] = ...
                                obj.takeStep_(i1, i2, alpha, Y, K, b, E, N, C, scaledEps);
                            totalSteps = totalSteps + 1;
                            if stepped
                                numChanged = numChanged + 1;
                                break;
                            end
                        end
                    end
                end

                if numChanged == 0
                    passes = passes + 1;
                else
                    passes = 0;
                end
            end
            
            if totalSteps >= maxTotalSteps
                warning('OceanStats:KernelMethods:SMOSolver:MaxStepsReached', ...
                        'SMO stopped early at %d steps.', maxTotalSteps);
                status.solved = false;
            else
                status.solved = true;
            end
            
            status.totalSteps = totalSteps;
            status.passes     = passes;
        end
        
        function name = getName(~)
            name = "Platt SMO Solver";
        end
    end
    
    methods (Access = private)
        function f_i = decFunc_(~, alpha, Y, K, b, i)
            f_i = (alpha .* Y)' * K(:, i) + b;
        end

        function [alpha, b, E, stepped] = takeStep_(obj, i1, i2, alpha, Y, K, b, E, N, C, scaledEps)
            stepped = false;
            if i1 == i2; return; end

            alph1 = alpha(i1);  y1 = Y(i1);
            alph2 = alpha(i2);  y2 = Y(i2);
            E1    = E(i1);
            E2    = E(i2);
            s     = y1 * y2;

        %alpha1_new  - alpha2_new = gamma where gamma = alpha1_old - alpha2_old
            if s < 0 %Opposite labels
                Lo = max(0,       alph2 - alph1);
                Hi = min(C,       C + alph2 - alph1);
            else %Same labels
                Lo = max(0,       alph1 + alph2 - C);
                Hi = min(C,       alph1 + alph2);
            end

            if Lo >= Hi; return; end
            % =========================================================================
            % DERIVATION OF THE SECOND DERIVATIVE (ETA)
            %
            % 1. The 2D Objective Function (freezing all other N-2 variables):
            %    W(a1, a2) = a1 + a2 - 0.5 * [K11*a1^2 + K22*a2^2 + 2*s*K12*a1*a2] - ...
            %    (Note: y1^2 = 1 and y2^2 = 1 are already applied)
            %
            % 2. The Equality Constraint (Handcuff):
            %    a1*y1 + a2*y2 = constant
            %    a1 = gamma - s*a2     (where s = y1*y2)
            %
            % 3. Substitute a1 to collapse W into a 1D parabola W(a2):
            %    To find the curvature, we only extract the quadratic terms (a2^2):
            %    From -0.5 * K11 * (-s*a2)^2  --> -0.5 * K11 * a2^2   (since s^2 = 1)
            %    From -0.5 * K22 * a2^2       --> -0.5 * K22 * a2^2
            %    From -s * K12 * (-s*a2)*a2   -->  1.0 * K12 * a2^2   (since s^2 = 1)
            %
            % 4. Combine quadratic coefficients (A):
            %    A = K12 - 0.5*K11 - 0.5*K22
            %
            % 5. The Second Derivative:
            %    For a parabola Ax^2 + Bx + C, the second derivative is 2A.
            %    d^2W / da2^2 = 2*K12 - K11 - K22 = eta
            %
            % THEOREM CHECK:
            % By Mercer's Theorem, the squared distance in feature space is:
            % ||Phi(x1) - Phi(x2)||^2 = K11 + K22 - 2*K12 >= 0
            % Therefore, eta is exactly the negative of this physical distance, 
            % mathematically guaranteeing eta <= 0 (a downward-facing parabola).
            % =========================================================================
            eta = 2*K(i1,i2) - K(i1,i1) - K(i2,i2);
            
            % If eta is identically 0 or positive due to numerical precision,
            % or if it's extremely close to 0, bounding logic is required.
            % However, for massive polynomials where K11=1e4, eta is also massive (-1e4).
            % The Newton step is d / eta. If eta is too large, the step vanishes.

            % =========================================================================
            % DERIVATION OF THE FIRST DERIVATIVE (GRADIENT) & NEWTON STEP
            %
            % 1. Partial Derivatives of the 2D Objective Function W(a1, a2):
            %    dW/da1 = 1 - K11*a1 - s*K12*a2 - y1*v1
            %    dW/da2 = 1 - K22*a2 - s*K12*a1 - y2*v2
            %
            % 2. Connection to the Prediction Error (E_i = f(x_i) - y_i):
            %    The primal decision function is f(x1) = a1*y1*K11 + a2*y2*K12 + v1 + b.
            %    Multiplying by y1 (and using y1^2 = 1, s = y1*y2) gives:
            %    y1*f(x1) = a1*K11 + a2*s*K12 + y1*v1 + y1*b
            %    
            %    Rearranging this into the dW/da1 equation yields:
            %    dW/da1 = 1 - (y1*f(x1) - y1*b)
            %
            %    Using the error definition E1 = f(x1) - y1, we get -y1*E1 = 1 - y1*f(x1).
            %    Substituting this completely removes the primal variables:
            %    dW/da1 = -y1*E1 + y1*b
            %    dW/da2 = -y2*E2 + y2*b   (by identical symmetry)
            %
            % 3. The Chain Rule (Enforcing the Equality Constraint):
            %    Because a1 = gamma - s*a2, changing a2 forces a1 to change.
            %    The total derivative of W with respect to a2 must account for this:
            %    dW_total/da2 = (dW/da1 * da1/da2) + dW/da2
            %
            %    Since da1/da2 = -s = -y1*y2, we substitute to find the total gradient:
            %    dW_total/da2 = (-y1*y2) * (-y1*E1 + y1*b) + (-y2*E2 + y2*b)
            %
            % 4. Algebraic Collapse:
            %    Distributing (-y1*y2) and applying y1^2 = 1:
            %    dW_total/da2 = y2*E1 - y2*b - y2*E2 + y2*b
            %    
            %    The global intercept terms (-y2*b and +y2*b) perfectly cancel out:
            %    dW_total/da2 = y2*(E1 - E2)
            %
            % 5. The Unconstrained Newton Step:
            %    To find the global maximum of the parabola: 
            %    a_new = a_old - (First Derivative / Second Derivative)
            %    a2_new = a2_old - [y2*(E1 - E2)] / eta
            % =========================================================================
            if eta < -scaledEps
                a2 = alph2 - y2*(E1 - E2) / eta; %Newton step
                %Ensure constraint
                a2 = min(max(a2, Lo), Hi);
            else
                a2 = obj.evalObjBound_(alph1, alph2, y1, y2, E1, E2, ...
                                        Lo, Hi, K, i1, i2, s);
            end

            % Check for exceedingly small learning progress relative to scale
            if abs(a2 - alph2) < scaledEps * (a2 + alph2 + scaledEps)
                return;
            end
            
            % Protection against wild gradient swings from Polynomial Kernels:
            % If the step throws alpha strictly out of bounds by a huge margin,
            % clamp it, but mark stepped=false if the clamped step is insignificant.
            if a2 == Lo || a2 == Hi
                if abs(a2 - alph2) < scaledEps
                    return;
                end
            end

            %Ensure global equality constraint sum (a_i * y_i) = 0
            a1 = alph1 + s*(alph2 - a2);
            a1 = min(max(a1, 0), C);

            alpha(i1) = a1;
            alpha(i2) = a2;

            %Assumption that hte point lies exactly on the boundary yi(f(xi)) = 1
            b1 = b - E1 - y1*(a1-alph1)*K(i1,i1) - y2*(a2-alph2)*K(i1,i2);
            b2 = b - E2 - y1*(a1-alph1)*K(i1,i2) - y2*(a2-alph2)*K(i2,i2);
            
            %Update bias, which one satisfy assumption
            if a1 > scaledEps && a1 < C - scaledEps
                b = b1;
            elseif a2 > scaledEps && a2 < C - scaledEps
                b = b2;
            else
                b = (b1 + b2) / 2;
            end

            E(i1) = obj.decFunc_(alpha, Y, K, b, i1) - y1;
            E(i2) = obj.decFunc_(alpha, Y, K, b, i2) - y2;
            stepped = true;
        end

        function a2 = evalObjBound_(~, alph1, alph2, y1, y2, E1, E2, Lo, Hi, K, i1, i2, s)
        % =====================================================================
        % DERIVATION OF THE OBJECTIVE FUNCTION EVALUATION (PSI)
        %
        % 1. Goal: Isolate the frozen contribution (v_i + b) from the rest of the 
        %    dataset (j = 3...N) to avoid O(N) summations.
        %    Definition: f(x1) = y1*a1*K11 + y2*a2*K12 + (v1 + b)
        %    Substitute f(x1) = E1 + y1 and multiply by y1 (using y1^2 = 1, s = y1*y2):
        %    y1*(E1 + y1) = a1*K11 + s*a2*K12 + y1*(v1 + b)
        %
        % 2. Extraction of Frozen Bias (f1, f2):
        %    f1 = y1*(v1 + b) = y1*(E1 + y1) - alph1*K11 - s*alph2*K12
        %    f2 = y2*(v2 + b) = y2*(E2 + y2) - s*alph1*K12 - alph2*K22
        % =====================================================================
        
        f1 = y1*(E1 + y1) - alph1*K(i1,i1) - s*alph2*K(i1,i2);
        f2 = y2*(E2 + y2) - s*alph1*K(i1,i2) - alph2*K(i2,i2);

        % Map the boundaries Lo/Hi of a2 to the corresponding a1 values 
        % using the equality constraint: a1_new = a1_old + s*(a2_old - a2_new)
        aLo1 = alph1 + s*(alph2 - Lo);
        aHi1 = alph1 + s*(alph2 - Hi);

        % =====================================================================
        % 3. Reconstruction of the Dual Objective W(a1, a2):
        %    The dual is: W = sum(a) - 0.5*sum(sum(y_i*y_j*a_i*a_j*K_ij))
        %    By substituting f1/f2 into the expanded W and ignoring constant terms 
        %    that do not depend on a1 or a2, we derive the proxy "Psi":
        %    Psi = a1*f1 + a2*f2 + 0.5*a1^2*K11 + 0.5*a2^2*K22 + s*a1*a2*K12
        %
        %    Note: This Psi is a simplified mathematical representation of the 
        %    negative objective. Minimizing Psi maximizes the SVM margin.
        % =====================================================================
        
        Psi_lo = aLo1*f1 + Lo*f2 + ...
                0.5*aLo1^2*K(i1,i1) + 0.5*Lo^2*K(i2,i2) + s*aLo1*Lo*K(i1,i2);
                
        Psi_hi = aHi1*f1 + Hi*f2 + ...
                0.5*aHi1^2*K(i1,i1) + 0.5*Hi^2*K(i2,i2) + s*aHi1*Hi*K(i1,i2);

        % =====================================================================
        % 4. Selection (Robustness Fallback):
        %    When eta >= 0, the function is either linear or an upward parabola.
        %    The maximum must lie at one of the endpoints (Lo or Hi).
        % =====================================================================
        if Psi_lo < Psi_hi
            a2 = Lo;
        elseif Psi_hi < Psi_lo
            a2 = Hi;
        else
            a2 = alph2; % No progress possible
        end
    end
    end
end
