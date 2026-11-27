function [S, lambda] = getSmootherMatrix(x, param_val, mode)
    %GET_SMOOTHER_MATRIX Computes the Smoothing Splines matrix S_lambda
    %Implements the derivation from ex.5.9. ESL: 
    % S_lambda = (I + lambda * K) ^-1
    %Theory:
    % Problem: Min 
    % RSS(f, lambda) = Sum_(i=1 to N){y_i - y(xi)}.^2 + lambda *
    % integration{ (f''(t)).^2 dt (eq. 5.9)
    %Solution: Proof by Ex 5.7 that Natural Cubic Spline is the unique with
    %smoothest.
    % Solution: f(x) = Sum_[(j=1 to N) (N_j(x) * theta_j)]
    %Substitue solution
    % Matrix form: RSS(theta, lambda) = (y - N*theta)^T ( y - N *theta) +
    % lambda theta^T omega_N * theta
    %Where {omega_N}_jk = integration N_j''(t) N_k''(t) dt
    % Take derivative
    %We have: Theta_hat = (N'N + lambda *omega_N)^-1 N' y
    % prediction f_hat = N * theta_hat
    %Subsitute: f_hat = N * (N'N + lambda *omega_N)^-1 N'
    %Define S_lambda = N * (N'N + lambda *omega_N)^-1 N'
    %So f_hat = S_lambda * y
    %   The Smoothing splines minimizes: ||y - f||^2 + lambda * f' K f
    %   S_lambda is symmetric, Positive Semidfinite
    %   Where K = Q * R^-1 * Q' is independent of lambda. Ex. 5.9 for
    %   derivation
    x = x(:);
    N = length(x);
    %Spacing h_i = x_{i+1} - x_i
    %Used in integral derivations: int_{x_i} ^ {x_{i+1}} ( f'')^2 dt
    h = diff(x);

    %Construct penalty matrix K ( Notice K is independent of lambda)
    %Instead of inverse (I + Lambda*K)^-1 for every guess lambda.
    %We construct K using (knots differences Q,R).
    % K u_k = d_k * u_k ( Notice d_k, u_k are fixed constant)
    % Compute S_lambda for any lamda
    % S_lambda = Sum_(k=1 to N) rho_k(lambda) u_k * u_k^T
    % rho_k(lambda) = 1 / ( 1 + lambda * d_k)  ->> Use this.
    

    %Penalty K = integration_(x1 to xN) { f''(t)^2 dt}
    %Solution: Using properties of Natural cubib spline (f'' is linear
    %between knots). Solve algebra..
    % Matrix form. m' R m (Problem: m are unknown curvatures)
    %Solution: Using constraint (smoothness definition) to related m to
    %heights of data f e.g slope left x^- = x^+ . Solve Algebra
    %We got. Matrix form Rm = Q' f -> m = R^-1 Q' f
    %Subsitute into K = m' R m -> K = Q R^-1 Q'
   
    %The Difference Matrix
    Q = zeros(N, N-2); % N x N-2
    for j = 1:N-2
        Q(j,   j) = 1/h(j);
        Q(j+1, j) = -(1/h(j) + 1/h(j+1));
        Q(j+2, j) = 1/h(j+1);
    end
    
    %Construct R (The integral Matrix)
    R = zeros(N-2, N-2); %N-2 x N-2
    for j = 1:N-2
        R(j, j) = (h(j) + h(j+1)) / 3;
        
        if j < N-2
            % Off-diagonal: h_i / 6
            R(j, j+1) = h(j+1) / 6;
            R(j+1, j) = h(j+1) / 6;
        end
    end
    
    K = Q * (R \ Q');
    
    % Enforce exact numerical symmetry (crucial for eig to return orthogonal U)
    K = (K + K') / 2;

    %Eigen-Decomposition (The Optimization)
    %Theory: S_lambda shares eigenvectors with K.
    %We do this ONCE
    [U, D_matrix] = eig(K);
    d = diag(D_matrix);

    %Determine Lambda
    if strcmp(mode, 'lambda')
        lambda = param_val;
    elseif strcmp(mode, 'df')
        target_df = param_val;
        
        % The maximum possible DF is bounded by the number of unique sample points
        max_df = sum(d < 1e10); % effectively N
        if target_df > max_df
            target_df = max_df;
        end

        % Instead of fzero which requires perfectly bracketed root signs,
        % use fminbnd on the squared difference. This is numerically safe
        % and guaranteed to find the closest possible lambda without crashing.
        loss_func = @(lam) (sum(1 ./ (1 + lam * d)) - target_df)^2;
        
        if abs(target_df - N) < 1e-3
            lambda = 0;
        else
            % Use fminsearch/fminbnd on a log scale for better numerical stability
            log_loss = @(log_lam) (sum(1 ./ (1 + (10^log_lam) * d)) - target_df)^2;
            log_lam_opt = fminbnd(log_loss, -15, 10);
            lambda = 10^log_lam_opt;
        end
        
    end
    %Construct S using eq 5.19
    rho = 1 ./ (1 + lambda * d);
    %Reconstruct matrix S = U * diag(rho) * U'
    S = U * diag(rho) * U';
end








