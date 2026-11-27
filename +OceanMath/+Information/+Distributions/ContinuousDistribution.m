classdef ContinuousDistribution < OceanMath.Information.Distributions.IDistribution
    % CONTINUOUSDISTRIBUTION  A validated probability density function (PDF).
    %
    % Wraps a continuous distribution defined over a real-valued domain.
    % Provides the infrastructure for differential entropy calculations and
    % future metric strategies (e.g., KL Divergence between two PDFs).
    %
    % This class accepts a function handle representing the PDF, along with
    % its support interval [a, b]. The PDF is validated numerically by
    % verifying that the integral over the support is approximately 1.
    %
    % Construction — two modes:
    %   1) From a function handle -> ContinuousDistribution.fromPDF(f, support)
    %      For parametric PDFs like Gaussian, Beta, Exponential, etc.
    %
    %   2) From raw data (KDE)   -> ContinuousDistribution.fromData(samples)
    %      Automatically fits a Kernel Density Estimator. Useful for
    %      empirical distributions where the model is unknown.
    %
    % Usage:
    %   % Parametric: Standard Normal PDF
    %   f = @(x) (1/sqrt(2*pi)) * exp(-0.5 * x.^2);
    %   d = OceanMath.Information.Distributions.ContinuousDistribution.fromPDF(f, [-8, 8]);
    %   p_at_zero = d.evaluate(0);    % f(0) = 0.3989...
    %   h = d.entropy();              % bits
    %
    %   % Empirical: from raw data
    %   d2 = OceanMath.Information.Distributions.ContinuousDistribution.fromData(randn(500,1));
    %
    % See also: OceanMath.Information.Distributions.IDistribution
    %           OceanMath.Information.Distributions.DiscreteDistribution

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % PDF_        : function handle f(x) -> density value (>= 0)
        % Support_    : (1 x 2) double [lower_bound, upper_bound]
        % IsKDE_      : logical, true if built from data (empirical)
        % KDEData_    : raw samples stored for KDE queries (can be empty)
        % Bandwidth_  : KDE bandwidth (used if IsKDE_ is true)
        PDF_       function_handle
        Support_   (1,2) double
        IsKDE_     (1,1) logical = false
        KDEData_   (:,1) double = []
        Bandwidth_ (1,1) double = 0
    end

    % ------------------------------------------------------------------ %
    methods (Static)
        function obj = fromPDF(f, support, options)
            % FROMPDF  Wrap a parametric PDF function handle.
            %
            % INPUTS:
            %   f       - function handle: f(x) -> density, must accept vectors
            %   support - (1 x 2) double, [lower_bound, upper_bound]
            %
            % Name-Value Options:
            %   Validate (logical): if true (default), numerically checks
            %             integral(f, a, b) ~= 1. Set to false for speed.
            %
            % OUTPUTS:
            %   obj - ContinuousDistribution
            arguments
                f        (1,1) function_handle
                support  (1,2) double
                options.Validate (1,1) logical = true
            end
            if support(1) >= support(2)
                error('OceanMath:Information:ContinuousDistribution:InvalidSupport', ...
                    'Support must be [lower, upper] with lower < upper.');
            end
            if options.Validate
                % Numerical integration check (adaptive quadrature)
                integral_val = integral(f, support(1), support(2));
                if abs(integral_val - 1.0) > 1e-4
                    error('OceanMath:Information:ContinuousDistribution:InvalidPDF', ...
                        'PDF does not integrate to 1 over [%.4g, %.4g]. Got %.8f.', ...
                        support(1), support(2), integral_val);
                end
            end
            obj = OceanMath.Information.Distributions.ContinuousDistribution(f, support, false, [], 0);
        end

        function obj = fromData(samples, options)
            % FROMDATA  Fit a Kernel Density Estimator to raw samples.
            %
            % Uses Silverman's rule-of-thumb for the bandwidth:
            %   h = 1.06 * sigma * N^(-1/5)
            % and a Gaussian kernel. The resulting PDF approximates the
            % true underlying distribution in a non-parametric way.
            %
            % INPUTS:
            %   samples - (N x 1) numeric vector of real-valued observations
            %
            % Name-Value Options:
            %   Bandwidth (double): override Silverman's bandwidth estimate
            %
            % OUTPUTS:
            %   obj - ContinuousDistribution
            arguments
                samples  (:,1) double
                options.Bandwidth (1,1) double = 0   % 0 = auto
            end
            if options.Bandwidth > 0
                bw = options.Bandwidth;
            else
                % Silverman's rule
                n  = numel(samples);
                bw = 1.06 * std(samples) * n^(-0.2);
            end
            if bw == 0
                error('OceanMath:Information:ContinuousDistribution:ZeroBandwidth', ...
                    'All samples are identical; bandwidth is zero. Cannot build KDE.');
            end

            % Build Gaussian KDE as a function handle
            s   = samples;
            h   = bw;
            f   = @(x) OceanMath.Information.Distributions.ContinuousDistribution.gaussKDE_(x, s, h);

            % Support: use 4*bw padding beyond the data range
            support = [min(samples) - 4*bw, max(samples) + 4*bw];
            obj = OceanMath.Information.Distributions.ContinuousDistribution(f, support, true, samples, bw);
        end
    end

    % ------------------------------------------------------------------ %
    methods
        % ---- IDistribution Interface Implementation ---- %

        function p = evaluate(obj, x)
            % EVALUATE  Query the density f(x) at one or more points.
            %
            % INPUTS:
            %   x - scalar or vector of query points (double)
            %
            % OUTPUTS:
            %   p - density values f(x), same shape as x. Always >= 0.
            p = obj.PDF_(x(:)');
            p = max(p, 0);    % Numerical clip; prevents -eps artifacts
        end

        function tf = isValid(obj)
            % ISVALID  Numerically verify the PDF integrates to 1 over its support.
            integral_val = integral(obj.PDF_, obj.Support_(1), obj.Support_(2));
            tf = abs(integral_val - 1.0) < 1e-4;
        end

        function n = getSupportSize(obj)
            % GETSUPPORTSIZE  Always returns Inf for continuous distributions.
            n = Inf;
        end

        function supp = getSupport(obj)
            % GETSUPPORT  Return the support interval [lower, upper].
            supp = obj.Support_;
        end

        % ---- Efficient Override for entropy() ---- %

        function h = entropy(obj)
            % ENTROPY  Compute differential entropy h(X) in bits.
            %
            % h(X) = -integral( f(x) * log2(f(x)) ) dx
            %
            % Uses adaptive quadrature. Safely handles f(x)=0 regions
            % because lim_{p->0} p*log2(p) = 0.
            integrand = @(x) obj.safeEntropyIntegrand_(x);
            h = -integral(integrand, obj.Support_(1), obj.Support_(2));
        end

        % ---- Utilities ---- %

        function disp(obj)
            % DISP  Pretty-print a summary of the distribution.
            if obj.IsKDE_
                method = sprintf('KDE (h=%.4f, N=%d)', obj.Bandwidth_, numel(obj.KDEData_));
            else
                method = 'Parametric PDF';
            end
            fprintf('ContinuousDistribution [%s]\n', method);
            fprintf('  Support  : [%.4g, %.4g]\n', obj.Support_(1), obj.Support_(2));
            fprintf('  Entropy  : %.4f bits\n', obj.entropy());
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function obj = ContinuousDistribution(f, support, isKDE, data, bw)
            obj.PDF_       = f;
            obj.Support_   = support;
            obj.IsKDE_     = isKDE;
            obj.KDEData_   = data;
            obj.Bandwidth_ = bw;
        end

        function v = safeEntropyIntegrand_(obj, x)
            % For entropy integration: f(x) * log2(f(x)), with 0*log(0) = 0.
            p         = obj.PDF_(x);
            p         = max(p, 0);
            v         = zeros(size(p));
            mask      = p > 1e-15;
            v(mask)   = p(mask) .* log2(p(mask));
        end
    end

    % ------------------------------------------------------------------ %
    methods (Static, Access = private)
        function p = gaussKDE_(x, samples, h)
            % GAUSSKDE_  Evaluate Gaussian KDE at point(s) x.
            % x       : (1 x M) query points
            % samples : (N x 1) training samples
            % h       : bandwidth
            n = numel(samples);
            % Vectorized: (N x M) matrix of kernel evaluations
            z = (x - samples) / h;           % broadcast: samples col, x row
            K = exp(-0.5 * z.^2) / (h * sqrt(2*pi));
            p = sum(K, 1) / n;               % mean of kernels per query point
        end
    end
end
