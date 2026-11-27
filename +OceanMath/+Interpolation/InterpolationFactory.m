classdef InterpolationFactory
    % INTERPOLATIONFACTORY  (Deprecated) Legacy wrapper for ScatteredInterpolant.
    %
    %   Prefer using OceanMath.Interpolation.ScatteredInterpolant directly:
    %     model = OceanMath.Interpolation.ScatteredInterpolant('Method', 'rbf', ...)
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19

    methods (Static)
        function model = create(type, varargin)
            % CREATE  Legacy factory wrapper (deprecated).
            warning('InterpolationFactory:Deprecated', ...
                'InterpolationFactory is deprecated. Use OceanMath.Interpolation.ScatteredInterpolant directly.');
            model = OceanMath.Interpolation.ScatteredInterpolant('Method', type, varargin{:});
        end

        function list = listTypes()
            list = {'rbf', 'kernel-regression', 'knn', 'natural', 'spline'};
        end
    end
end
