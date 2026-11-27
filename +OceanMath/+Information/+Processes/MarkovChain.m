classdef MarkovChain < OceanMath.Information.Processes.IStochasticProcess
    % MARKOVCHAIN  A finite, discrete-time, homogeneous Markov Chain.
    %
    % Encapsulates a row-stochastic transition matrix P where P(i,j) is the
    % probability of transitioning from state i to state j in one step.
    %
    % The class validates the matrix on construction, then provides efficient
    % implementations of all IStochasticProcess methods, including power-
    % iteration for the stationary distribution and spectral ergodicity check.
    %
    % Construction:
    %   mc = OceanMath.Information.Processes.MarkovChain(P)
    %   mc = OceanMath.Information.Processes.MarkovChain(P, 'StateLabels', {'Rain','Sun'})
    %
    % Usage:
    %   P  = [0.7 0.3; 0.4 0.6];
    %   mc = OceanMath.Information.Processes.MarkovChain(P);
    %   mu = mc.getStationaryDistribution();   % DiscreteDistribution
    %   p  = mc.getTransitionProbability(1,2); % 0.3
    %   tf = mc.isErgodic();                   % true
    %   seq = mc.simulate(100, 1);             % 101-step trajectory
    %
    % See also: OceanMath.Information.Processes.IStochasticProcess
    %           OceanMath.Information.Processes.RandomWalkGraph

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % P_           : (K x K) row-stochastic transition matrix
        % StateLabels_ : (K x 1) cell/numeric — human-readable state names
        % K_           : integer, number of states
        P_           (:,:) double
        StateLabels_ (:,1)
        K_           (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    % Cached stationary distribution (lazy-computed on first call)
    properties (Access = private)
        StationaryCache_ = []
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = MarkovChain(P, options)
            % MARKOVCHAIN  Construct from a (K x K) transition matrix.
            %
            % INPUTS:
            %   P - (K x K) double, row-stochastic. Each row must sum to 1
            %       and all entries must be non-negative.
            %
            % Name-Value Options:
            %   StateLabels (cell or numeric): human-readable names for the
            %       K states. Default: {1, 2, ..., K}.
            arguments
                P        (:,:) double {mustBeNonnegative}
                options.StateLabels = []
            end

            % --- Validation ---
            [r, c] = size(P);
            if r ~= c
                error('OceanMath:Information:MarkovChain:NotSquare', ...
                    'Transition matrix P must be square. Got %dx%d.', r, c);
            end
            rowSums = sum(P, 2);
            if any(abs(rowSums - 1) > 1e-9)
                badRows = find(abs(rowSums - 1) > 1e-9);
                error('OceanMath:Information:MarkovChain:NotStochastic', ...
                    'Rows must sum to 1. Rows %s do not (sums: %s).', ...
                    num2str(badRows'), num2str(rowSums(badRows)', '%.6f '));
            end

            obj.P_  = P;
            obj.K_  = r;

            % --- State labels ---
            if isempty(options.StateLabels)
                obj.StateLabels_ = (1:r)';
            else
                labels = options.StateLabels;
                if numel(labels) ~= r
                    error('OceanMath:Information:MarkovChain:LabelMismatch', ...
                        'StateLabels must have %d elements, got %d.', r, numel(labels));
                end
                obj.StateLabels_ = labels(:);
            end
        end

        % ---- IStochasticProcess Interface Implementation ---- %

        function dist = getStationaryDistribution(obj)
            % GETSTATIONARYDISTRIBUTION  Compute mu via power iteration.
            %
            % Uses the left-eigenvector method: solve mu * P = mu, sum(mu)=1.
            % Result is cached after the first call.
            %
            % OUTPUTS:
            %   dist - DiscreteDistribution over the K state labels

            if ~isempty(obj.StationaryCache_)
                dist = obj.StationaryCache_;
                return;
            end

            % Left eigenvectors: P' * v = lambda * v -> eigenvalue 1
            [V, D] = eig(obj.P_');
            eigenvalues = diag(D);
            % The stationary distribution corresponds to eigenvalue == 1
            [~, idx] = min(abs(eigenvalues - 1.0));
            mu = abs(real(V(:, idx)));
            mu = mu / sum(mu);   % Normalize to sum to 1

            dist = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF(...
                obj.StateLabels_, mu);

            obj.StationaryCache_ = dist;   % Cache it
        end

        function p = getTransitionProbability(obj, stateI, stateJ)
            % GETTRANSITIONPROBABILITY  Return P(i,j) from the matrix.
            %
            % INPUTS:
            %   stateI - integer index of source state (1-indexed)
            %   stateJ - integer index of destination state (1-indexed)
            %
            % OUTPUTS:
            %   p - scalar double in [0, 1]
            obj.validateStateIndex_(stateI, 'stateI');
            obj.validateStateIndex_(stateJ, 'stateJ');
            p = obj.P_(stateI, stateJ);
        end

        function P = getTransitionMatrix(obj)
            % GETTRANSITIONMATRIX  Return the full (K x K) matrix P.
            P = obj.P_;
        end

        function K = getNumStates(obj)
            % GETNUMSTATES  Return the number of states K.
            K = obj.K_;
        end

        function tf = isErgodic(obj)
            % ISERGODIC  Check irreducibility and aperiodicity via Wielandt bound.
            %
            % A finite Markov chain is ergodic (regular) if and only if:
            %   P^m > 0  (all entries strictly positive)
            % where m is the Wielandt bound:
            %   m = K^2 - 2K + 2
            %
            % This is the definitive algebraic criterion — no false positives.
            % Using (I + P)^K was incorrect because adding I inflates the
            % diagonal and can mask periodicity.
            %
            % Reference: Wielandt (1950) — tightest known bound for primitivity.
            m  = obj.K_^2 - 2*obj.K_ + 2;
            Pm = obj.P_^m;
            tf = all(Pm(:) > 1e-12);
        end

        % ---- Useful Utilities ---- %

        function disp(obj)
            % DISP  Pretty-print a chain summary.
            fprintf('MarkovChain  [%d states]\n', obj.K_);
            fprintf('  Ergodic  : %s\n', mat2str(obj.isErgodic()));
            mu = obj.getStationaryDistribution();
            fprintf('  Stationary distribution:\n');
            tbl = mu.toTable();
            disp(tbl);
        end

        function hRate = entropyRate(obj)
            % ENTROPYRATE  Compute H(X) = -sum_ij mu_i * P_ij * log2(P_ij)
            %
            % This is the shortcut formula for the entropy rate of an ergodic
            % Markov chain. It avoids computing the full joint entropy of long
            % sequences and is exact in the stationary regime.
            %
            % OUTPUTS:
            %   hRate - scalar double, entropy rate in bits per timestep
            mu = obj.getStationaryDistribution();
            muVec = mu.evaluate(mu.getSupport())';  % (K x 1)

            P  = obj.P_;
            % Safely compute P_ij * log2(P_ij), treating 0*log(0) as 0
            logP = zeros(size(P));
            mask = P > 0;
            logP(mask) = log2(P(mask));

            % hRate = -sum over i,j of mu_i * P_ij * log2(P_ij)
            hRate = -sum(muVec .* sum(P .* logP, 2));
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function validateStateIndex_(obj, idx, name)
            if ~isscalar(idx) || idx < 1 || idx > obj.K_ || mod(idx,1) ~= 0
                error('OceanMath:Information:MarkovChain:InvalidState', ...
                    '%s must be an integer in [1, %d]. Got %g.', name, obj.K_, idx);
            end
        end
    end
end
