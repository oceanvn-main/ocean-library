classdef ScatteredSpline < OceanMath.Interpolation.IScatteredInterpolant
    % SCATTEREDSPLINE  1-D scattered interpolation via Smoothing Spline.
    %
    %   Thin adapter around OceanStats.NonParametric.SmoothingSpline.
    %   Only valid for 1-D inputs (D=1).
    %
    %   Usage:
    %     ss = OceanMath.Interpolation.ScatteredSpline('Lambda', 1e-4);
    %     ss.fit(X, y);          % X must be (N x 1)
    %     y_hat = ss.predict(Xq);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        Lambda (1,1) double {mustBePositive} = 1e-4  % Regularization strength
        Df     (1,1) double = NaN   % Target degrees of freedom (NaN → use Lambda)
    end

    properties (Access = private)
        Model_   % SmoothingSpline instance
    end

    methods
        function obj = ScatteredSpline(opts)
            % SCATTEREDSPLINE Constructor.
            %
            %   obj = ScatteredSpline('Lambda', 1e-3)
            %   obj = ScatteredSpline('Df', 10)   % fix DoF, Lambda ignored

            arguments
                opts.Lambda (1,1) double {mustBePositive} = 1e-4
                opts.Df     (1,1) double                  = NaN
            end
            obj.Lambda = opts.Lambda;
            obj.Df     = opts.Df;
        end

        function name = getName(obj)
            if isnan(obj.Df)
                name = sprintf('SmoothSpline-lam%.2e', obj.Lambda);
            else
                name = sprintf('SmoothSpline-df%g', obj.Df);
            end
        end
    end

    methods (Access = protected)

        function fitCore_(obj, X, y)
            if size(X, 2) ~= 1
                error('ScatteredSpline:Only1D', ...
                    'ScatteredSpline only supports 1-D inputs (D=1). Got D=%d.', size(X,2));
            end

            if isnan(obj.Df)
                obj.Model_ = OceanStats.NonParametric.SmoothingSpline( ...
                    'Lambda', obj.Lambda);
            else
                obj.Model_ = OceanStats.NonParametric.SmoothingSpline( ...
                    'Df', obj.Df);
            end
            obj.Model_.fit(X, y);
        end

        function y_hat = predictCore_(obj, Xq)
            y_hat = obj.Model_.predict(Xq);
        end

    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Return pedagogical metadata for the GUI learner application.
            here = fileparts(mfilename('fullpath'));
            raw  = fileread(fullfile(here, 'meta', 'ScatteredSpline.meta.json'));
            meta = jsondecode(raw);
        end
    end
end

