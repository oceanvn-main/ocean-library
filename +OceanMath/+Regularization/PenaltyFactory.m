classdef PenaltyFactory
    % PENALTYFACTORY  Static factory for constructing robust penalty strategies.
    %
    %   Package: OceanMath.Regularization
    %   Author: Ocean Library  |  2026-08-05

    methods (Static)
        function pen = create(name, varargin)
            % CREATE Construct penalty instance by string identifier.
            arguments
                name (1,1) string
            end
            arguments (Repeating)
                varargin
            end

            switch lower(name)
                case {"l2", "quadratic"}
                    pen = OceanMath.Regularization.QuadraticPenalty();
                case {"l1", "tv", "totalvariation"}
                    pen = OceanMath.Regularization.L1Penalty(varargin{:});
                case {"pnorm", "hyperlaplacian"}
                    pen = OceanMath.Regularization.PNormPenalty(varargin{:});
                case {"huber"}
                    pen = OceanMath.Regularization.HuberPenalty(varargin{:});
                case {"gnc", "clamped"}
                    pen = OceanMath.Regularization.GNCPenalty(varargin{:});
                otherwise
                    error('PenaltyFactory:UnknownPenalty', 'Unknown penalty name: %s', name);
            end
        end
    end
end
