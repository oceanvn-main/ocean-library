classdef ScatteredNatural < OceanMath.Interpolation.IScatteredInterpolant
    % SCATTEREDNATURAL  Thin adapter around MATLAB's built-in scatteredInterpolant.
    %
    %   Supports 'natural', 'linear', and 'nearest' interpolation methods
    %   backed by Delaunay triangulation. Works for 1D, 2D, and 3D inputs.
    %
    %   Usage:
    %     sn = OceanMath.Interpolation.ScatteredNatural('Method', 'natural');
    %     sn.fit(X, y);
    %     y_hat = sn.predict(Xq);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        Method (1,1) string {mustBeMember(Method, ["natural","linear","nearest"])} = "natural"
        % ExtrapolationMethod: value returned outside convex hull
        ExtrapolationMethod (1,1) string {mustBeMember(ExtrapolationMethod, ...
            ["none","nearest","linear"])} = "none"
    end

    properties (Access = private)
        Interp_   % MATLAB scatteredInterpolant object (stored after fit)
    end

    methods
        function obj = ScatteredNatural(opts)
            % SCATTEREDNATURAL Constructor.
            %
            %   obj = ScatteredNatural()
            %   obj = ScatteredNatural('Method', 'linear')
            %   obj = ScatteredNatural('Method', 'nearest', 'ExtrapolationMethod', 'nearest')

            arguments
                opts.Method              (1,1) string = "natural"
                opts.ExtrapolationMethod (1,1) string = "none"
            end
            obj.Method              = opts.Method;
            obj.ExtrapolationMethod = opts.ExtrapolationMethod;
        end

        function name = getName(obj)
            name = sprintf('Natural-%s', obj.Method);
        end
    end

    methods (Access = protected)

        function fitCore_(obj, X, y)
            % Build MATLAB's scatteredInterpolant from (X, y).
            % scatteredInterpolant requires columns as separate arguments.
            D = size(X, 2);

            colArgs = cell(1, D);
            for d = 1:D
                colArgs{d} = X(:, d);
            end

            obj.Interp_ = scatteredInterpolant(colArgs{:}, y, ...
                char(obj.Method), char(obj.ExtrapolationMethod));
        end

        function y_hat = predictCore_(obj, Xq)
            D = size(Xq, 2);
            colArgs = cell(1, D);
            for d = 1:D
                colArgs{d} = Xq(:, d);
            end
            y_hat = obj.Interp_(colArgs{:});
            % scatteredInterpolant returns NaN outside convex hull when Method='none'
            % Replace NaN with 0 for consistent metric computation
            y_hat(isnan(y_hat)) = 0;
        end

    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Return pedagogical metadata for the GUI learner application.
            here = fileparts(mfilename('fullpath'));
            raw  = fileread(fullfile(here, 'meta', 'ScatteredNatural.meta.json'));
            meta = jsondecode(raw);
        end
    end
end

