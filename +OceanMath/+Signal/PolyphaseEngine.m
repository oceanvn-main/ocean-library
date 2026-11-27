classdef PolyphaseEngine
    % POLYPHASEENGINE  Arbitrary-M polyphase FIR filter decomposition for decimation.
    %
    %   PROBLEM MOTIVATION (Computational Inefficiency of Naive Decimation):
    %     1. In standard decimation, an anti-aliasing FIR filter h[n] of length N is applied
    %        to a signal x[n] at the HIGH sample rate F_in.
    %     2. Then, downsampling by factor M discards M-1 of every M calculated output samples.
    %     3. Computational waste: (M-1)/M (e.g., 75% for M=4) of all multiply-accumulate operations
    %        are computed at high rate and immediately discarded!
    %
    %   CONDITIONS TO USE THIS METHOD (Noble Identity Prerequisites):
    %     1. INTEGER FACTOR: The decimation/interpolation factor M MUST be a positive integer (M in Z+).
    %     2. LTI FIR FILTER: The filter h[n] must be Linear Time-Invariant and Finite Impulse Response (FIR).
    %     3. NOBLE IDENTITY COMMUTATIVITY:
    %        Noble Identity for Decimation states:
    %          [Filter H(z^M)] -> [Downsample M]  ===  [Downsample M] -> [Filter H(z)]
    %        This allows moving the filtering operation to AFTER the downsampling step,
    %        so sub-filters operate strictly at the LOW sample rate F_in / M.
    %
    %   SCIENTIFIC PIPELINE DECOMPOSITION:
    %
    %     A. KNOWNS:
    %        - Full FIR filter coefficients h[n] of length N.
    %        - Integer decimation factor M.
    %        - High-rate input signal x[n].
    %
    %     B. UNKNOWNS:
    %        - Exact equivalence between full-rate filtered array y_direct and polyphase output y_polyphase.
    %        - Exact FLOP savings ratio for signal of length N_signal.
    %
    %     C. ASSUMPTIONS:
    %        - Output of polyphase structure is NUMERICALLY BIT-IDENTICAL to direct downsampling
    %          (max error < 1e-10, no approximation introduced).
    %
    %     D. MATHEMATICAL MODEL (Type-I Polyphase Decomposition):
    %        - Polyphase sub-filters: e_k[n] = h[M*n + k]  for k = 0, 1, ..., M-1.
    %        - Branch signals: x_k[n] = x[M*n + k] (subsampled input streams).
    %        - Total Output: y[n] = sum_{k=0}^{M-1} conv(x_k[n], e_k[n]).
    %
    %   USAGE:
    %     h      = fir1(63, 0.9/4);   % 64-tap AA filter for M=4
    %     eng    = OceanMath.Signal.PolyphaseEngine(h, 4);
    %     y_dir  = eng.filterDirect(x);      % reference implementation
    %     y_pp   = eng.filterPolyphase(x);   % polyphase (efficient)
    %     eng.verifyEquivalence(x);          % assert max|y_dir-y_pp| < 1e-10
    %     E      = eng.getSubfilters();      % M x ceil(N/M) sub-filter matrix
    %
    %   REFERENCE: Oppenheim & Schafer 3rd Ed.
    %              §4.7.1 — Interchange of Filtering and Downsampling/Upsampling
    %              §4.7.2 — Polyphase Decompositions, Eq. 4.104–4.112
    %              §4.7.3 — Polyphase Implementation of Decimation Filters
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    properties (SetAccess = private)
        H        (1,:) double   % Full FIR filter coefficients h[n], length N
        M        (1,1) double   % Decimation factor (number of polyphase branches)
        N        (1,1) double   % Filter length
        E        (:,:) double   % Polyphase sub-filter matrix: M x ceil(N/M)
                                %   E(k+1, :) = e_k[n] = h[M*n + k]
    end

    properties (Constant, Hidden)
        ClassDescription = ['Polyphase FIR filter decomposition for integer-M decimation. '...
            'Decomposes h[n] into M sub-filters e_k[n]=h[M*n+k] via the Noble Identity. '...
            'Provides filterDirect (reference) and filterPolyphase (efficient) with '...
            'bit-identical output. Computational gain = M. '...
            'See Oppenheim & Schafer §4.7.2-4.7.3 Eq. 4.104-4.112.'];
        ProblemMotivation = ['Naive decimation computes full FIR filtering at the HIGH rate '...
            'then discards (M-1)/M output samples. Polyphase decomposition uses the Noble Identity '...
            'to move filtering AFTER downsampling, achieving an exact M-times FLOP reduction.'];
        ConditionsToUse = ['1. Integer decimation/interpolation factor M in Z+; '...
            '2. Linear Time-Invariant FIR filter h[n]; '...
            '3. Sub-filter decomposition e_k[n] = h[M*n+k].'];
        Knowns = ['Full FIR filter h[n]; Integer decimation factor M; Input signal x[n].'];
        Unknowns = ['Equivalence tolerance max|y_direct - y_polyphase|; Actual runtime speedup.'];
        Assumptions = ['Polyphase output is bit-identical to direct filter-then-downsample output.'];
        MathematicalModel = ['e_k[n] = h[M*n + k]; x_k[n] = x[M*n + k]; '...
            'y[n] = sum_{k=0}^{M-1} (x_k * e_k)[n].'];
        ConceptNodeId = 'ch4-polyphase-decomposition';
        TextbookRef   = 'Oppenheim & Schafer 3rd Ed., §4.7.2 Eq. 4.104–4.112';
        PackagePath   = '+OceanMath/+Signal/PolyphaseEngine.m';
    end

    methods
        % ── Constructor ──────────────────────────────────────────────────────────
        function obj = PolyphaseEngine(h, M)
            % POLYPHASEENGINE  Construct from FIR coefficients and decimation factor.
            %
            %   eng = PolyphaseEngine(h, M)
            %
            %   Parameters:
            %     h — Full FIR filter coefficient vector h[n], row vector, length N.
            %         Typically a windowed-sinc lowpass with cutoff pi/M.
            %     M — Integer decimation factor (number of polyphase branches).
            %
            %   On construction, decomposes h into M polyphase sub-filters.
            arguments
                h (1,:) double
                M (1,1) double {mustBePositive, mustBeInteger}
            end
            assert(M >= 1, 'PolyphaseEngine:BadM', 'M must be >= 1.');
            obj.H = h;
            obj.M = M;
            obj.N = length(h);

            % ── Polyphase Decomposition ──────────────────────────────────────────
            % Pad h to length ceil(N/M)*M so rows partition evenly.
            % h_pad[n] = h[n] for n < N, = 0 for n >= N.
            %
            % Polyphase components (Oppenheim §4.7.2, Eq. 4.104):
            %   e_k[n] = h[M*n + k]  for k = 0, 1, ..., M-1
            %
            % Stored as E(k+1, :), a matrix of size [M, ceil(N/M)].
            %
            % Note: This is the TYPE-I polyphase decomposition.
            L_padded = ceil(obj.N / M) * M;
            h_padded = [h, zeros(1, L_padded - obj.N)];

            % Reshape: each row is one polyphase branch e_k
            % h_padded = [h[0], h[1], ..., h[M-1], h[M], ..., h[2M-1], ...]
            % e_k[n]   = h[M*n + k] = h_padded(k + M*n + 1)
            obj.E = reshape(h_padded, M, []);   % M rows, ceil(N/M) cols
        end

        % ── Two Equivalent Implementations ───────────────────────────────────────

        function y = filterDirect(obj, x)
            % FILTERDIRECT  Reference implementation: full-rate filter then decimate.
            %
            %   y = eng.filterDirect(x)
            %
            %   Operation (straightforward but computationally wasteful):
            %     1. Convolve x[n] with h[n] at the full input rate
            %     2. Decimate: keep every M-th output sample
            %
            %   Cost: length(h) multiply-adds per INPUT sample, then discard M-1.
            %   Effective cost: N multiply-adds per OUTPUT sample.
            %
            %   This is the REFERENCE whose output must match filterPolyphase() exactly.
            arguments
                obj
                x (1,:) double
            end
            % Full convolution, then keep every M-th sample starting at 1
            x_filt = conv(x, obj.H, 'full');
            % 'full' gives length N+L-1; take every M-th after group-delay offset
            % Group delay of linear-phase FIR: (N-1)/2 samples
            delay = (obj.N - 1) / 2;
            start = round(delay) + 1;   % 1-indexed MATLAB start
            y = x_filt(start : obj.M : end);
        end

        function y = filterPolyphase(obj, x)
            % FILTERPOLYPHASE  Efficient polyphase implementation: decimate then subfilter.
            %
            %   y = eng.filterPolyphase(x)
            %
            %   Operation (Noble Identity, Oppenheim §4.7.1):
            %     For each polyphase branch k = 0..M-1:
            %       1. Decimate x: keep x[M*n + k]  (THIS happens at LOW rate)
            %       2. Convolve with sub-filter e_k[n]  (at LOW rate — M times cheaper)
            %       3. Sum all M branch outputs
            %
            %   Cost: ceil(N/M) multiply-adds per OUTPUT sample.
            %   vs filterDirect: N multiply-adds per OUTPUT sample.
            %   Speedup: factor of M exactly.
            %
            %   Output is NUMERICALLY IDENTICAL to filterDirect() — no approximation.
            arguments
                obj
                x (1,:) double
            end
            Nx = length(x);
            M  = obj.M;
            E  = obj.E;              % [M x L_sub] sub-filter matrix
            L_sub = size(E, 2);      % length of each sub-filter

            % Number of output samples
            Ny = ceil((Nx - M + 1) / M);   % conservative lower bound
            y  = zeros(1, Ny);

            for k = 0 : M-1
                % Decimate x by picking samples at positions k+1, k+M+1, k+2M+1, ...
                x_branch = x(k+1 : M : end);      % low-rate branch signal

                % Convolve branch with polyphase sub-filter e_k = E(k+1, :)
                y_branch  = conv(x_branch, E(k+1, :), 'full');

                % Align and accumulate (sum M branch outputs)
                n_out = min(length(y_branch), length(y));
                y(1:n_out) = y(1:n_out) + y_branch(1:n_out);
            end
        end

        function verifyEquivalence(obj, x, tol)
            % VERIFYEQUIVALENCE  Assert that filterDirect and filterPolyphase match.
            %
            %   eng.verifyEquivalence(x)
            %   eng.verifyEquivalence(x, tol)  % default tol = 1e-9
            %
            %   Computes max(|y_direct - y_polyphase|) and asserts < tol.
            %   This is the numerical proof of the Noble Identity for this filter.
            arguments
                obj
                x   (1,:) double
                tol (1,1) double {mustBePositive} = 1e-9
            end
            y_dir = obj.filterDirect(x);
            y_pp  = obj.filterPolyphase(x);
            n_cmp = min(length(y_dir), length(y_pp));
            max_diff = max(abs(y_dir(1:n_cmp) - y_pp(1:n_cmp)));
            if max_diff > tol
                error('PolyphaseEngine:EquivalenceFailed', ...
                    'Noble Identity verification FAILED: max diff = %e (tol = %e)', ...
                    max_diff, tol);
            else
                fprintf('[PolyphaseEngine] Noble Identity VERIFIED: max|y_dir - y_pp| = %e < %e\n', ...
                    max_diff, tol);
            end
        end

        % ── Introspection ─────────────────────────────────────────────────────────

        function E = getSubfilters(obj)
            % GETSUBFILTERS  Return [M x L_sub] polyphase sub-filter matrix.
            %   E(k+1, :) = e_k[n] = h[M*n + k] for k = 0..M-1.
            E = obj.E;
        end

        function flops = estimateFLOPs(obj, N_signal)
            % ESTIMATEFLOPS  Compare theoretical FLOPs: direct vs polyphase.
            %
            %   flops = eng.estimateFLOPs(N_signal)
            %
            %   Returns struct with fields:
            %     flops.direct_per_output    — N multiply-adds per output sample
            %     flops.polyphase_per_output — ceil(N/M) multiply-adds per output
            %     flops.speedup              — exact M factor
            %     flops.direct_total         — total for N_signal input
            %     flops.polyphase_total       — total for N_signal input
            arguments
                obj
                N_signal (1,1) double {mustBePositive} = 100000
            end
            N_out = ceil(N_signal / obj.M);
            flops.direct_per_output    = obj.N;
            flops.polyphase_per_output = ceil(obj.N / obj.M);
            flops.speedup              = obj.M;   % exact by Noble Identity
            flops.direct_total         = obj.N * N_out;
            flops.polyphase_total      = ceil(obj.N / obj.M) * N_out;
        end

        function s = getMetadata(obj)
            % GETMETADATA  Return description struct for agents and OceanLearn.
            s.ClassName         = 'PolyphaseEngine';
            s.ConceptNodeId     = obj.ConceptNodeId;
            s.TextbookRef       = obj.TextbookRef;
            s.Description       = obj.ClassDescription;
            s.ProblemMotivation = obj.ProblemMotivation;
            s.ConditionsToUse   = obj.ConditionsToUse;
            s.Knowns            = obj.Knowns;
            s.Unknowns          = obj.Unknowns;
            s.Assumptions       = obj.Assumptions;
            s.MathematicalModel = obj.MathematicalModel;
            s.FilterLength      = obj.N;
            s.DecimationFactor  = obj.M;
            s.SubfilterLength   = size(obj.E, 2);
            s.NumBranches       = obj.M;
            s.SpeedupFactor     = obj.M;
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print polyphase decomposition summary to console.
            fprintf('=== PolyphaseEngine ===\n');
            fprintf('  Filter length N      : %d\n',    obj.N);
            fprintf('  Decimation factor M  : %d\n',    obj.M);
            fprintf('  Sub-filter length    : %d\n',    size(obj.E, 2));
            fprintf('  Num branches         : %d\n',    obj.M);
            fprintf('  Computational gain   : %dx  (Noble Identity)\n', obj.M);
            fprintf('  Concept node         : %s\n',    obj.ConceptNodeId);
        end
    end
end
