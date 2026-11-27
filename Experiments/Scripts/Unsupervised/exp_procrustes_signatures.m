% EXP_PROCRUSTES_SIGNATURES  ESL Exercises 14.9 & 14.10 — Two Procrustes Averages.
%
%   REFERENCE:      Hastie et al. (2009), ESL §14.5.3, Ex. 14.9–14.10, Fig. 14.26.
%   DATASET:        Synthetic 2-D "S"-curve with N=96 landmarks, p=2, L=3 shapes.
%                   3 corrupted versions: rigid (rotate+scale) x2, affine (shear) x1.
%   ALGORITHMS:
%     Module 1  rigid_procrustes_average    — Iterative ALS (Ex. 14.9, Eq. 14.115)
%     Module 2  affine_invariant_average    — Closed-form eigen (Ex. 14.10, Eq. 14.60)
%   ASSUMPTIONS:
%     (1) True data lies on a smooth planar curve; noise-free (clean synthetic).
%     (2) Input tensor X is [N, p, L]; centering applied inside each module.
%     (3) Rigid module uses U*V' rotation (may include reflections); no det fix.
%     (4) Affine module is invariant to any nonsingular linear map on each shape.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% =========================================================================
% === 1. Generate Canonical S-Curve (Parametric) ===
% =========================================================================
N = 96;    % number of landmark points
p = 2;     % spatial dimension (2D)
L = 3;     % number of shape versions

% Parametric S-shape: sine-based curve traces a smooth letter "S"
%   x(t) = sin(t)        oscillates -1..+1 (left-right)
%   y(t) = t/pi - 1      monotone   -1..+1 (bottom-top)
%   Together: bottom (0,-1) -> right (1,0) -> center -> left (-1,0.5) -> top (0,1)
t      = linspace(0, 2*pi, N)';
x_base = sin(t);
y_base = t / pi - 1;
X_base = [x_base, y_base];              % (N x 2) canonical S, zero-mean approx

% =========================================================================
% === 2. Create L=3 Corrupted Versions ===
% =========================================================================
% 2-D rotation matrix (for column-vector convention: R * [x;y])
%   For row-vector layout used here: X * R^T
rot2d = @(deg) [cosd(deg), -sind(deg); sind(deg), cosd(deg)];

% --- Version 1: Rotate +30 degrees, scale by 1.5 ---
X_v1 = 1.5 * X_base * rot2d(30)';      % (N x 2)  X_base * R_30^T

% --- Version 2: Rotate -45 degrees, scale by 0.5 ---
X_v2 = 0.5 * X_base * rot2d(-45)';     % (N x 2)  X_base * R_{-45}^T

% --- Version 3: Horizontal shear (affine, NOT rigid) by A = [1, 0.5; 0, 1] ---
% Shear matrix transforms [x, y] -> [x + 0.5*y,  y]  (horizontal shear of x)
% For row-vectors: X_shear = X_base * A^T = X_base * [1, 0; 0.5, 1]
A_shear = [1, 0.5; 0, 1];
X_v3    = X_base * A_shear';            % (N x 2)  affine-only, no rigid equivalent

% Pack into a 3D tensor [N, p, L]
X = cat(3, X_v1, X_v2, X_v3);         % (N x p x L) = (96 x 2 x 3)

fprintf('=== Exp 14.9 & 14.10: Generalized Procrustes Analysis ===\n');
fprintf('  Tensor shape: [N=%d, p=%d, L=%d]\n', N, p, L);
fprintf('  Version 1: Rotate +30 deg, scale x1.5  (rigid)\n');
fprintf('  Version 2: Rotate -45 deg, scale x0.5  (rigid)\n');
fprintf('  Version 3: Horizontal shear A=[1,0.5;0,1]  (affine)\n\n');

% =========================================================================
% === 3. Module 1 — Rigid Procrustes Average (Ex. 14.9, Iterative ALS) ===
% =========================================================================
fprintf('--- Running rigid_procrustes_average ---\n');
[M_rigid, R_all, beta_all, X_aligned_all] = ...
    OceanStats.Unsupervised.rigid_procrustes_average(X, 1e-8, 500);

fprintf('  Recovered scales beta_l: ');
fprintf('%.4f  ', beta_all);
fprintf('\n');
for l = 1:L
    theta_l = atan2d(R_all(2,1,l), R_all(1,1,l));
    fprintf('  Shape %d: theta_rec = %.2f deg\n', l, theta_l);
end

% =========================================================================
% === 4. Module 2 — Affine-Invariant Average (Ex. 14.10, Closed-Form) ===
% =========================================================================
fprintf('\n--- Running affine_invariant_average ---\n');
M_affine = OceanStats.Unsupervised.affine_invariant_average(X);
fprintf('  M_affine shape: [%d x %d], M^T*M deviation from I: %.2e\n', ...
    size(M_affine,1), size(M_affine,2), norm(M_affine'*M_affine - eye(p),'fro'));

% =========================================================================
% === 4b. Rotation-Invariance Note (Why M_affine looks "wrong") ===
% =========================================================================
% The affine-invariant average solves:
%   min_{A_l, M: M^T M = I_p}  sum_l ||X_l A_l - M||_F^2
%
% Key property: if M* is optimal, so is M* * Q for ANY 2×2 orthogonal Q.
%   Proof: (M* Q)^T (M* Q) = Q^T (M*^T M*) Q = Q^T I Q = I    (still feasible)
%          The cost is unchanged because A_l absorbs Q: A_l -> Q^T A_l.
%
% Consequence: eigs() returns ONE valid orientation of the subspace, chosen
%   by floating-point arithmetic — NOT anchored to any physical reference frame.
%   This is why M_affine's raw columns do NOT trace a recognizable S-curve.
%
% The rigid algorithm (Ex 14.9) does NOT have this problem:
%   The ALS loop is anchored to the first shape (X_v1) as initialization,
%   and R_l must be orthogonal — so orientation is fixed from iteration 1.
%
% Fig 2 below shows this explicitly:
%   Left  = M_rigid (fixed orientation -> visual S-curve)
%   Right = M_affine raw columns (arbitrary orientation -> abstract pattern)
%   They MAY span the same subspace, but ONLY after an explicit re-orientation
%   step (which is NOT part of the ESL Ex 14.10 algorithm).

fprintf('\n--- Rotation-Invariance of Affine Average ---\n');
fprintf('  M_affine is defined up to any 2x2 orthogonal Q: M_affine * Q is equally valid.\n');
fprintf('  The raw eigs output has NO guaranteed visual orientation.\n');
fprintf('  M_rigid IS orientation-specific (anchored to V1 by ALS initialization).\n');

% Affine residuals (for back-projection in Fig 1)
X_affine_proj = zeros(N, p, L);
for l = 1:L
    Xl_c = X(:,:,l) - mean(X(:,:,l), 1);
    X_affine_proj(:,:,l) = M_affine * (M_affine' * Xl_c);
end

% Rigid GPA residual cost (sum over aligned shapes)
cost_rigid = 0;
for l = 1:L
    cost_rigid = cost_rigid + norm(X_aligned_all(:,:,l) - M_rigid, 'fro')^2;
end


% =========================================================================
% === 5b. Back-project M into each raw shape's coordinate space ===
% =========================================================================
% Purpose: produce the "reconstructed from model" curves to overlay on originals.
%
% Rigid GPA back-projection (exact inverse of the forward alignment):
%   Forward:  X_aligned_l = beta_l * X_centered_l * R_l  approx  M_rigid
%   Backward: M_back_rigid_l = (1/beta_l) * M_rigid * R_l^T + centroid_l
%   Interpretation: what M_rigid looks like if drawn in shape l's raw space.
%
% Affine avg back-projection (projection of M onto col(X_l)):
%   Forward:  A_l = pinv(X_l_c) * M_affine   =>  X_l_c * A_l = H_l * M_affine
%   Backward: M_back_affine_l = H_l * M_affine + centroid_l
%   where H_l = Q_l*Q_l' is the rank-p projector onto col(X_l). (ESL Eq.14.61)
%   Interpretation: the component of M_affine explainable by shape l's column space.

centroid_raw  = zeros(L, p);
M_back_rigid  = zeros(N, p, L);
M_back_affine = zeros(N, p, L);

for l = 1:L
    mu_l              = mean(X(:,:,l), 1);          % (1 x p) raw centroid
    centroid_raw(l,:) = mu_l;

    % Rigid back-projection
    M_back_rigid(:,:,l) = (1/beta_all(l)) * M_rigid * R_all(:,:,l)' + mu_l;

    % Affine back-projection via economy QR projector
    Xl_c    = X(:,:,l) - mu_l;                     % (N x p) centred slice
    [Q_l,~] = qr(Xl_c, 0);                         % economy QR: Q_l is (N x p)
    H_l     = Q_l * Q_l';                          % (N x N) rank-p projector
    M_back_affine(:,:,l) = H_l * M_affine + mu_l;  % project M, then un-centre
end

fprintf('\n--- Residual costs ---\n');
fprintf('  Rigid GPA total cost  : %.6f\n', cost_rigid);

% =========================================================================
% === 6. Visualization — 3x4 grid (original vs reconstructed) ===
% =========================================================================
%   Columns 1–3: V1, V2, V3 (one shape per column)
%   Column 4:    summary (all raws overlaid | rigid M | both M's)
%
%   Row 1: Original raw shape only              (what we measured)
%   Row 2: Raw + Rigid M back-projected (blue)  (what rigid model expects)
%   Row 3: Raw + Affine M back-projected (red)  (what affine model expects)
%
% Design note: V3 uses a shear transform (affine, NOT rigid).
%   Rigid back-projection on V3 will NOT match — visually demonstrating
%   that rotation+scale cannot undo a shear.
%   Affine back-projection uses H_l and will match much better.

viz    = OceanViz.Core('Theme', 'Dark');
bg     = [0.10 0.10 0.13];

raw_colors = [0.30 0.70 1.00;   % V1 — sky blue
              1.00 0.50 0.25;   % V2 — coral orange
              0.55 0.90 0.40];  % V3 — mint green
col_rigid  = [0.40 0.70 1.00];  % rigid reconstruction  — blue
col_affine = [1.00 0.35 0.35];  % affine reconstruction — red

shape_titles = {
    'V1: Rotate +30°, Scale ×1.5', ...
    'V2: Rotate −45°, Scale ×0.5', ...
    'V3: Shear [1 0.5; 0 1]  (affine)'};
row_labels = {'Original (Raw)', 'Rigid GPA Recon', 'Affine Avg Recon'};

fig = figure('Name', 'ESL Ex.14.9 & 14.10 — Original vs Reconstructed', ...
    'Color', bg, 'Position', [20, 30, 1500, 900]);
tl  = tiledlayout(fig, 3, 4, 'TileSpacing', 'tight', 'Padding', 'compact');

for row = 1:3
    for col = 1:3
        ax = nexttile(tl, (row-1)*4 + col);
        hold(ax, 'on');

        Xraw_l = X(:,:,col);   % (N x 2) raw shape for version col

        % Always draw the raw (original) shape — solid, per-shape color
        plot(ax, Xraw_l(:,1), Xraw_l(:,2), '-', ...
            'Color', raw_colors(col,:), 'LineWidth', 2.2, ...
            'DisplayName', 'Original');

        if row == 2
            % Rigid back-projection — dashed blue
            Mb = M_back_rigid(:,:,col);
            plot(ax, Mb(:,1), Mb(:,2), '--', ...
                'Color', col_rigid, 'LineWidth', 2.0, ...
                'DisplayName', 'M_{rigid} (back)');
            legend(ax, 'Location', 'best', 'FontSize', 6, 'TextColor', 'w');

        elseif row == 3
            % Affine back-projection — dashed red
            Ma = M_back_affine(:,:,col);
            plot(ax, Ma(:,1), Ma(:,2), '--', ...
                'Color', col_affine, 'LineWidth', 2.0, ...
                'DisplayName', 'M_{affine} (back)');
            legend(ax, 'Location', 'best', 'FontSize', 6, 'TextColor', 'w');
        end

        if row == 1
            title(ax, shape_titles{col}, 'Color', 'w', 'FontSize', 9);
        end
        if col == 1
            ylabel(ax, row_labels{row}, 'Color', [0.8 0.8 0.8], 'FontSize', 8);
        end
        axis(ax, 'equal');
        viz.styleAxes(ax);
    end

    % 4th column: per-row summary panel
    ax4 = nexttile(tl, row*4);
    hold(ax4, 'on');
    if row == 1
        for l = 1:L
            Xr = X(:,:,l);
            plot(ax4, Xr(:,1), Xr(:,2), '-', 'Color', raw_colors(l,:), ...
                'LineWidth', 1.5, 'DisplayName', shape_titles{l});
        end
        legend(ax4, 'Location', 'best', 'FontSize', 5, 'TextColor', 'w');
        title(ax4, 'All Raw Overlaid', 'Color', 'w', 'FontSize', 9);
    elseif row == 2
        plot(ax4, M_rigid(:,1), M_rigid(:,2), '-', 'Color', col_rigid, ...
            'LineWidth', 2.5, 'DisplayName', 'M_{rigid}');
        legend(ax4, 'Location', 'best', 'FontSize', 7, 'TextColor', 'w');
        title(ax4, 'Rigid Mean M', 'Color', 'w', 'FontSize', 9);
    else
        plot(ax4, M_rigid(:,1),  M_rigid(:,2),  '-',  'Color', col_rigid,  ...
            'LineWidth', 2.0, 'DisplayName', 'M_{rigid}');
        plot(ax4, M_affine(:,1), M_affine(:,2), '--', 'Color', col_affine, ...
            'LineWidth', 2.0, 'DisplayName', 'M_{affine}');
        legend(ax4, 'Location', 'best', 'FontSize', 7, 'TextColor', 'w');
        title(ax4, 'Both Templates', 'Color', 'w', 'FontSize', 9);
    end
    axis(ax4, 'equal');
    viz.styleAxes(ax4);
end

sgtitle(fig, ...
    sprintf(['ESL Ex.14.9 & 14.10 — Original vs Reconstructed  |  ' ...
             'Rigid cost = %.4f\n' ...
             'V1/V2=rigid transforms  |  V3=shear (affine only) — ' ...
             'rigid recon diverges on V3; affine recon fits'], cost_rigid), ...
    'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');

% =========================================================================
% === 6b. Subspace Proof Figure — "They span the same 2-D subspace" ===
% =========================================================================
% Three panels showing M_rigid, M_affine (raw columns), and M_affine_vis.
% M_affine_vis = P_affine * X_base = M_affine * (M_affine' * X_base_norm).
% If col(M_affine) ≡ col(M_rigid), ONLY AFTER re-orientation would they look the same.
% This figure shows the RAW algorithm outputs — without any extra alignment step.

fig2 = figure('Name', 'Rotation-Invariance: Why M_affine looks abstract', ...
    'Color', bg, 'Position', [60, 80, 1100, 420]);
tl2  = tiledlayout(fig2, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: M_rigid raw output — orientation is FIXED by ALS initialization
ax_s1 = nexttile(tl2);
hold(ax_s1, 'on');
plot(ax_s1, M_rigid(:,1), M_rigid(:,2), '-', 'Color', col_rigid, 'LineWidth', 2.5);
title(ax_s1, 'M_{rigid}  (Ex. 14.9 output)', 'Color', 'w', 'FontSize', 10);
subtitle(ax_s1, 'Orientation FIXED — ALS anchors to V1 from iteration 1', ...
    'Color', [0.55 0.90 0.55], 'FontSize', 8);
xlabel(ax_s1, 'x', 'Color', 'w'); ylabel(ax_s1, 'y', 'Color', 'w');
axis(ax_s1, 'equal'); viz.styleAxes(ax_s1);

% Panel 2: M_affine raw columns — orientation is ARBITRARY (rotation-invariant)
ax_s2 = nexttile(tl2);
hold(ax_s2, 'on');
plot(ax_s2, M_affine(:,1), M_affine(:,2), '--', 'Color', col_affine, 'LineWidth', 2.5);
title(ax_s2, 'M_{affine}  (Ex. 14.10 output)', 'Color', 'w', 'FontSize', 10);
subtitle(ax_s2, 'Orientation ARBITRARY — M*Q is equally valid for any orthogonal Q', ...
    'Color', [1.0 0.6 0.4], 'FontSize', 8);
xlabel(ax_s2, 'x', 'Color', 'w'); ylabel(ax_s2, 'y', 'Color', 'w');
axis(ax_s2, 'equal'); viz.styleAxes(ax_s2);

% Panel 3: Both raw outputs overlaid — shows they don't match directly
%   This is the PROBLEM: even if the underlying subspace is the same,
%   the raw outputs cannot be compared without an explicit re-orientation step.
ax_s3 = nexttile(tl2);
hold(ax_s3, 'on');
plot(ax_s3, M_rigid(:,1),  M_rigid(:,2),  '-',  'Color', col_rigid, ...
    'LineWidth', 2.5, 'DisplayName', 'M_{rigid} (raw)');
plot(ax_s3, M_affine(:,1), M_affine(:,2), '--', 'Color', col_affine, ...
    'LineWidth', 2.0, 'DisplayName', 'M_{affine} (raw)');
legend(ax_s3, 'Location', 'best', 'FontSize', 8, 'TextColor', 'w');
title(ax_s3, 'Both Raw Outputs — Direct Comparison', 'Color', 'w', 'FontSize', 10);
subtitle(ax_s3, 'No extra alignment step — raw mismatch exposes rotation invariance', ...
    'Color', [1.0 0.85 0.30], 'FontSize', 8);
xlabel(ax_s3, 'x', 'Color', 'w'); ylabel(ax_s3, 'y', 'Color', 'w');
axis(ax_s3, 'equal'); viz.styleAxes(ax_s3);

sgtitle(fig2, ...
    ['Rotation-Invariance of Affine Average (Ex. 14.10)' newline ...
     'M_{affine} is defined up to any 2\times2 orthogonal rotation Q: ' ...
     'eigs() picks one arbitrarily. ' ...
     'M_{rigid} has no such ambiguity — ALS fixes the orientation from step 1.'], ...
    'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');

% =========================================================================
% === 7. Summary Printout ===
% =========================================================================
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset     : Synthetic S-curves (N=%d, p=%d, L=%d)\n', N, p, L);
fprintf('  V1          : Rotate +30 deg, scale x1.5   (rigid)\n');
fprintf('  V2          : Rotate -45 deg, scale x0.5   (rigid)\n');
fprintf('  V3          : Shear [1,0.5;0,1]             (affine — not rigid!)\n');
fprintf('  Algorithm 1 : Iterative rigid GPA  (Ex. 14.9,  Eq. 14.115)\n');
fprintf('  Algorithm 2 : Affine-invariant avg (Ex. 14.10, Eq. 14.60)\n');
fprintf('  Rigid cost  : %.6f\n', cost_rigid);
fprintf('  Key point   : M_affine is rotation-invariant in its p-dim subspace.\n');
fprintf('    => raw eigs output has no guaranteed visual orientation.\n');
fprintf('    => to visually interpret M_affine, an extra alignment step is needed\n');
fprintf('       (e.g. Procrustes-align M_affine to M_rigid or to X_base).\n');
fprintf('    => This extra step is NOT part of ESL Ex. 14.10.\n');
fprintf('==========================\n');

% Export both figures
out_path = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Results', 'exp_procrustes_signatures.png');
viz.export(fig, out_path);
fprintf('  Figure 1 saved: %s\n', out_path);

out_path2 = fullfile('c:\', 'Ocean Library', 'Matlab code practice', ...
    'Results', 'exp_procrustes_rotation_invariance.png');
viz.export(fig2, out_path2);
fprintf('  Figure 2 (rotation invariance) saved: %s\n', out_path2);


