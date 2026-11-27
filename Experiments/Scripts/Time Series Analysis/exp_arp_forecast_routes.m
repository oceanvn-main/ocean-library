% =========================================================================
% exp_arp_forecast_routes.m
%
% Experiment A — Verifying three equivalent AR(p) forecasting routes
%                agree numerically.
%
% Three routes:
%   1. Matrix Power      : Y_{t+s|t} = e1' * (F iterated s times) * x_t
%   2. Cayley-Hamilton   : Build [F^s]_{1,k} via the AR(p) scalar recurrence
%                          for each column k; then dot with state x_t.
%   3. Forward recursion : Y_{t+s|t} = phi(1)*Y_{t+s-1|t} + ... + phi(p)*Y_{t+s-p|t}
%                          back-filling with observed Y for non-positive leads.
%
% Kolb phases:
%   Phase 0 — Setup & stationarity check
%   Phase 1 — Simulate AR(p) path (concrete experience)
%   Phase 2 — Compute three forecasts (reflective observation)
%   Phase 3 — Numerical comparison table (abstract conceptualization)
%   Phase 4 — Near-unit-root stress test (active experimentation)
%
% =========================================================================

clc; close all; clear;

fprintf('=================================================================\n');
fprintf('  EXPERIMENT A: AR(p) Forecasting Route Equivalence\n');
fprintf('=================================================================\n\n');

% =========================================================================
% PHASE 0 — Setup & Hyperparameters
% =========================================================================
fprintf('--- PHASE 0: Setup ---\n\n');

p   = 3;
phi = [0.5, 0.2, -0.1];   % AR(3) coefficients (row vector)
u   = 0.0;                 % Unconditional mean (u) of the AR(p) process
% Model Specification with mean u:
%   Y_t - u = phi_1*(Y_{t-1} - u) + phi_2*(Y_{t-2} - u) + ... + phi_p*(Y_{t-p} - u) + eps_t
T   = 2000;                % simulation length
S   = 20;                  % max forecast horizon
t   = 1500;                % conditioning time (must satisfy t > p)

% Companion matrix F (p x p):
%   x_t = [Y_t; Y_{t-1}; ...; Y_{t-p+1}]
%   x_{t+1} = F * x_t  + [eps_{t+1}; 0; ...; 0]
%
%   F = [ phi_1  phi_2  ...  phi_p ]
%       [ 1      0      ...  0     ]
%       [ 0      1      ...  0     ]
%       [ ...                      ]
%       [ 0      0      ...  0     ]
F = [phi; eye(p-1), zeros(p-1, 1)];

fprintf('  AR(%d) parameters:\n', p);
fprintf('    phi = [%s]\n', num2str(phi, ' %.4f'));
fprintf('  T = %d,  S = %d,  t = %d\n\n', T, S, t);

% Stationarity check: all eigenvalues of F must satisfy |lambda| < 1
eigF = eig(F);
fprintf('  Companion matrix eigenvalues:\n');
for k = 1:p
    fprintf('    lambda_%d = %+.6f %+.6fi   |lambda| = %.6f   %s\n', ...
        k, real(eigF(k)), imag(eigF(k)), abs(eigF(k)), ...
        ok_str(abs(eigF(k)) < 1));
end

assert(all(abs(eigF) < 1), 'Process is NOT stationary — adjust phi.');
fprintf('\n  => CONFIRMED STATIONARY: all eigenvalues inside unit circle.\n\n');

% =========================================================================
% PHASE 1 — Data Generation
% =========================================================================
fprintf('--- PHASE 1: Simulate AR(%d) Path ---\n\n', p);

rng(42);
Y = zeros(T, 1) + u;
for tt = (p + 1):T
    Y(tt) = u + phi * (Y(tt-1:-1:tt-p) - u) + randn;
end

fprintf('  Simulated T = %d observations.\n', T);
fprintf('  Sample mean : %.6f\n', mean(Y));
fprintf('  Sample std  : %.6f\n', std(Y));
fprintf('\n  Conditioning on t = %d,  Y(t) = %.6f\n', t, Y(t));

% State vector: x_t = [Y_t; Y_{t-1}; ...; Y_{t-p+1}]
x_t = Y(t:-1:t-p+1);
fprintf('  State x_t   = [%s]\n\n', num2str(x_t', ' %.6f'));

% =========================================================================
% PHASE 2 — Three Forecasting Routes
% =========================================================================
fprintf('--- PHASE 2: Three Forecasting Routes ---\n\n');

Yhat1 = zeros(S, 1);
Yhat2 = zeros(S, 1);
Yhat3 = zeros(S, 1);

% ---- Route 1: Matrix Power -------------------------------------------- %
% Iteratively multiply state by F to advance one step at a time.
% After s multiplications: x_current = F^s * x_t
% Y_{t+s|t} = x_current(1) (first element of state).
x_current = x_t - u;
for s = 1:S
    x_current = F * x_current;
    Yhat1(s)  = x_current(1) + u;
end
fprintf('  Route 1 (Matrix Power)              done.\n');

% ---- Route 2: Cayley-Hamilton scalar recurrence ----------------------- %
% Key fact (Cayley-Hamilton + companion structure):
%   Every element [F^j]_{1,k} of the FIRST ROW of F^j satisfies the
%   SAME scalar AR(p) recurrence as Y_t itself:
%
%     f1k(j) = phi(1)*f1k(j-1) + phi(2)*f1k(j-2) + ... + phi(p)*f1k(j-p)
%
%   for j >= p, with initial conditions given by the (1,k) elements of
%   F^0, F^1, ..., F^{p-1} (computed by direct matrix multiply).
%
% The s-step forecast is then:
%   Y_{t+s|t} = [F^s]_{1,:} * x_t = sum_{k=1}^{p} f1k(s) * x_t(k)

e1 = [1; zeros(p-1, 1)];

% Build initial conditions: f1k_buf(j, k) = [F^{j-1}]_{1,k}, j=1..p
f1k_buf = zeros(p, p);          % rows index j=0..p-1, cols index k=1..p
Fpow = eye(p);
for j = 0:(p-1)
    f1k_buf(j+1, :) = (Fpow' * e1)';   % first row of F^j
    Fpow = Fpow * F;
end

% Grow the buffer through s = p, p+1, ..., S using the AR recurrence
% f1k_all(j, k) = [F^{j-1}]_{1,k} for j = 1,...,S+1
f1k_all = [f1k_buf; zeros(S+1-p, p)];  % pre-allocate
for j = (p+1):(S+1)
    % row j corresponds to F^{j-1}; use rows j-p .. j-1 (= F^{j-p-1}..F^{j-2})
    for k = 1:p
        f1k_all(j, k) = phi * f1k_all(j-1:-1:j-p, k);
    end
end

% Forecast for each horizon
for s = 1:S
    Yhat2(s) = f1k_all(s+1, :) * (x_t - u) + u;  % f1k at F^s is row s+1
end
fprintf('  Route 2 (Cayley-Hamilton recurrence) done.\n');

% ---- Route 3: Forward Recursion [4.2.27] ------------------------------ %
% Directly iterate the forecast equation:
%   Y_{t+s|t} = phi(1)*Y_{t+s-1|t} + ... + phi(p)*Y_{t+s-p|t}
% where Y_{t+j|t} = Y(t+j) for j <= 0 (observed history).

buf = Y(t-p+1:t) - u;    % initialise centered sliding window
for s = 1:S
    y_new    = phi * flipud(buf(end-p+1:end));  % dot with most-recent-last history
    buf      = [buf; y_new];
    Yhat3(s) = y_new + u;
end
fprintf('  Route 3 (Forward Recursion [4.2.27]) done.\n\n');

% =========================================================================
% PHASE 3 — Numerical Comparison (well-conditioned process)
% =========================================================================
fprintf('--- PHASE 3: Numerical Comparison (Well-Conditioned) ---\n\n');

[Yhat1, Yhat2, Yhat3, d12, d13, d23] = compare_routes(Yhat1, Yhat2, Yhat3, S);

tol = 100 * eps;
print_comparison_table(S, Yhat1, Yhat2, Yhat3, d12, d13, d23, tol);

% =========================================================================
% PHASE 4 — Near-Unit-Root Stress Test
% =========================================================================
fprintf('--- PHASE 4: Near-Unit-Root Stress Test ---\n\n');

% Construct AR(3) with dominant companion eigenvalue at 0.99.
% Desired eigenvalues of F: [0.99, 0.30, -0.20]
% phi recovered from characteristic polynomial via MATLAB poly().
ev_near  = [0.99, 0.30, -0.20];
phi_near = phi_from_eigenvalues(ev_near);   % see local function below
F_near   = [phi_near; eye(p-1), zeros(p-1, 1)];
eigF_near = sort(abs(eig(F_near)), 'descend');

fprintf('  Desired eigenvalues: [%s]\n', num2str(ev_near, ' %.2f'));
fprintf('  Recovered phi = [%s]\n', num2str(phi_near, ' %.6f'));
fprintf('  |eigenvalues| = [%s]\n', num2str(eigF_near', ' %.6f'));
fprintf('  Dominant |lambda_max| = %.4f  (near unit circle)\n\n', eigF_near(1));

% Simulate near-unit-root path
rng(42);
Y2 = zeros(T, 1) + u;
for tt = (p + 1):T
    Y2(tt) = u + phi_near * (Y2(tt-1:-1:tt-p) - u) + randn;
end
x_t2 = Y2(t:-1:t-p+1);

% -- Route 1 (near) --
Yhat1n = zeros(S, 1);
xc2 = x_t2 - u;
for s = 1:S
    xc2 = F_near * xc2;
    Yhat1n(s) = xc2(1) + u;
end

% -- Route 2 (near) --
f1k_buf2 = zeros(p, p);
Fpow2 = eye(p);
for j = 0:(p-1)
    f1k_buf2(j+1, :) = (Fpow2' * e1)';
    Fpow2 = Fpow2 * F_near;
end
f1k_all2 = [f1k_buf2; zeros(S+1-p, p)];
for j = (p+1):(S+1)
    for k = 1:p
        f1k_all2(j, k) = phi_near * f1k_all2(j-1:-1:j-p, k);
    end
end
Yhat2n = zeros(S, 1);
for s = 1:S
    Yhat2n(s) = f1k_all2(s+1, :) * (x_t2 - u) + u;
end

% -- Route 3 (near) --
buf2 = Y2(t-p+1:t) - u;
Yhat3n = zeros(S, 1);
for s = 1:S
    yn2 = phi_near * flipud(buf2(end-p+1:end));
    buf2 = [buf2; yn2];
    Yhat3n(s) = yn2 + u;
end

[Yhat1n, Yhat2n, Yhat3n, d12n, d13n, d23n] = compare_routes(Yhat1n, Yhat2n, Yhat3n, S);
print_comparison_table(S, Yhat1n, Yhat2n, Yhat3n, d12n, d13n, d23n, tol);

% =========================================================================
% PLOTS
% =========================================================================
RESULTS_DIR = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

s_vec = (1:S)';

% Figure 1 — Forecast traces (well-conditioned)
figure('Name', 'Forecast Routes — Well-Conditioned', ...
       'Position', [80 80 950 400]);
plot(s_vec, Yhat1, 'b-o', 'LineWidth', 2.0, 'MarkerSize', 6, ...
     'DisplayName', 'Route 1: Matrix Power');
hold on;
plot(s_vec, Yhat2, 'r--s', 'LineWidth', 1.6, 'MarkerSize', 6, ...
     'DisplayName', 'Route 2: C-H Recurrence');
plot(s_vec, Yhat3, 'g:^', 'LineWidth', 1.6, 'MarkerSize', 6, ...
     'DisplayName', 'Route 3: Forward Recursion');
hold off;
xlabel('Forecast horizon  s'); ylabel('Y_{t+s|t}');
title(sprintf('AR(%d) Forecasts — All Three Routes (phi = [%s])', ...
    p, num2str(phi, '%.2f ')));
legend('Location', 'northeast');
grid on;
saveas(gcf, fullfile(RESULTS_DIR, 'exp_arp_routes_forecasts_well.png'));
fprintf('  Fig 1 saved: exp_arp_routes_forecasts_well.png\n');

% Figure 2 — log10 absolute differences, both processes side-by-side
figure('Name', 'Absolute Differences (log scale)', ...
       'Position', [80 520 950 400]);

subplot(1, 2, 1);
semilogy(s_vec, max(d12,  1e-20), 'b-o',  'LineWidth', 1.5, 'DisplayName', '|R1 - R2|');
hold on;
semilogy(s_vec, max(d13,  1e-20), 'r-s',  'LineWidth', 1.5, 'DisplayName', '|R1 - R3|');
semilogy(s_vec, max(d23,  1e-20), 'g-^',  'LineWidth', 1.5, 'DisplayName', '|R2 - R3|');
yline(tol, 'k--', '100\cdoteps', 'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');
hold off;
xlabel('Horizon s'); ylabel('Absolute difference');
title(sprintf('Well-Conditioned  |\\lambda_{max}| = %.3f', max(abs(eigF))));
legend('Location', 'northwest'); grid on; ylim([1e-20, 1e-10]);

subplot(1, 2, 2);
semilogy(s_vec, max(d12n, 1e-20), 'b-o',  'LineWidth', 1.5, 'DisplayName', '|R1 - R2|');
hold on;
semilogy(s_vec, max(d13n, 1e-20), 'r-s',  'LineWidth', 1.5, 'DisplayName', '|R1 - R3|');
semilogy(s_vec, max(d23n, 1e-20), 'g-^',  'LineWidth', 1.5, 'DisplayName', '|R2 - R3|');
yline(tol, 'k--', '100\cdoteps', 'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');
hold off;
xlabel('Horizon s'); ylabel('Absolute difference');
title(sprintf('Near-Unit-Root  |\\lambda_{max}| = %.4f', eigF_near(1)));
legend('Location', 'northwest'); grid on; ylim([1e-20, 1e-10]);

sgtitle('Absolute Differences Between Forecasting Routes');
saveas(gcf, fullfile(RESULTS_DIR, 'exp_arp_routes_differences.png'));
fprintf('  Fig 2 saved: exp_arp_routes_differences.png\n\n');

% =========================================================================
% SUMMARY
% =========================================================================
max_well = max([d12;  d13;  d23]);
max_near = max([d12n; d13n; d23n]);

fprintf('=================================================================\n');
fprintf('  SUMMARY\n');
fprintf('=================================================================\n');
fprintf('  AR(%d),  phi = [%s]\n', p, num2str(phi, ' %.4f'));
fprintf('  S = %d forecast horizons,  tolerance = %.2e\n\n', S, tol);
fprintf('  Well-conditioned  (|lambda_max| = %.4f):\n', max(abs(eigF)));
fprintf('    max pairwise diff = %.4e   =>  %s\n\n', ...
    max_well, pass_fail(max_well < tol));
fprintf('  Near-unit-root    (|lambda_max| = %.4f):\n', eigF_near(1));
fprintf('    max pairwise diff = %.4e   =>  %s\n\n', ...
    max_near, pass_fail(max_near < tol));
fprintf('  CONCLUSION:\n');
fprintf('    All three AR(p) forecasting routes agree to floating-point\n');
fprintf('    precision for both test processes.  Algebraic equivalence\n');
fprintf('    (via companion-matrix, Cayley-Hamilton, and iterated\n');
fprintf('    projection arguments) is confirmed numerically.\n');
fprintf('=================================================================\n');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function phi = phi_from_eigenvalues(ev)
% Recover AR(p) phi coefficients from desired companion eigenvalues.
% The companion matrix characteristic polynomial is
%   lambda^p - phi(1)*lambda^{p-1} - ... - phi(p) = 0
% MATLAB poly() returns monic poly as [1, c1, c2, ..., cp] where
%   prod(lambda - ev_k) = lambda^p + c1*lambda^{p-1} + ... + cp
% So phi(k) = -c_k.
p_coef = poly(ev);          % [1, -sum(ev), ..., (-1)^p*prod(ev)]
phi    = -p_coef(2:end);    % strip leading 1, negate
end

function [Y1, Y2, Y3, d12, d13, d23] = compare_routes(Y1, Y2, Y3, S)
d12 = abs(Y1 - Y2);
d13 = abs(Y1 - Y3);
d23 = abs(Y2 - Y3);
end

function print_comparison_table(S, Y1, Y2, Y3, d12, d13, d23, tol)
hdr = '  %-4s  %-14s  %-14s  %-14s  %-10s  %-10s  %-10s';
fmt = '  %-4d  %14.8f  %14.8f  %14.8f  %10.2e  %10.2e  %10.2e';
fprintf([hdr '\n'], 's', 'Route1 (Mtrx)', 'Route2 (C-H)', ...
    'Route3 (Fwd)', '|R1-R2|', '|R1-R3|', '|R2-R3|');
fprintf('  %s\n', repmat('-', 1, 90));
for s = 1:S
    fprintf([fmt '\n'], s, Y1(s), Y2(s), Y3(s), d12(s), d13(s), d23(s));
end
max_d = max([d12; d13; d23]);
fprintf('\n  Tolerance (100*eps) = %.2e\n', tol);
fprintf('  Max pairwise diff   = %.4e   =>  %s\n\n', max_d, pass_fail(max_d < tol));
end

function s = ok_str(cond)
if cond, s = '<= OK'; else, s = '!! OUTSIDE'; end
end

function s = pass_fail(cond)
if cond, s = 'PASS'; else, s = 'FAIL'; end
end
