classdef KernelICAStrategy < OceanStats.Unsupervised.ICASolvers.IICASolver
    % KERNELICASTRATEGY  Kernel ICA (Bach & Jordan, 2002) — STUB.
    %
    % ESL §14.7 (discussion). Fundamentally different from FastICA/ProDenICA:
    % instead of maximizing negentropy of individual projections, KernelICA
    % minimizes a kernel-based measure of JOINT DEPENDENCE between all components.
    %
    % OBJECTIVE:
    %   min_W  KCI(s_1, ..., s_q)
    %   where KCI is the Kernel Canonical Correlation / HSIC criterion
    %   measuring statistical dependence in a Reproducing Kernel Hilbert Space.
    %
    % ALGORITHM (sketch):
    %   1. Construct centered kernel matrices K_1..K_q for each candidate IC
    %   2. Minimize sum of pairwise HSIC(K_i, K_j) over orthogonal W
    %   3. Gradient-based update on the Stiefel manifold (orthogonality constraint)
    %
    % WHY NOT DEFLATION:
    %   KernelICA optimizes ALL components jointly — sequential deflation would
    %   not minimize the joint dependence criterion. This makes the API subtly
    %   different from FastICA/ProDenICA.
    %
    % STATUS: Not yet implemented. Requires Stiefel manifold optimization.
    %         See Bach & Jordan (2002), JMLR 3:1-48 for the full algorithm.
    %
    % Author: Ocean Library | 2026-04-30

    properties (SetAccess = protected)
        Name = 'KernelICA'
    end

    methods
        function [W, converged, iters] = solve(~, ~, ~, ~)
            error('OceanStats:ICA:NotImplemented', ...
                ['KernelICA is not yet implemented.\n', ...
                 'KernelICA requires Stiefel manifold optimization and HSIC kernel matrices.\n', ...
                 'Reference: Bach & Jordan (2002) JMLR 3:1-48.\n', ...
                 'Use FastICAStrategy or ProDenICAStrategy instead.']);
        end
    end
end
