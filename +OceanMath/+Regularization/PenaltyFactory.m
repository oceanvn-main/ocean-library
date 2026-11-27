classdef PenaltyFactory
    % PENALTYFACTORY  Static factory for constructing robust penalty strategies.
    %
    %   Provides a centralized mechanism to instantiate any IRobustPenalty by name,
    %   mirroring the pattern of OceanMath.Signal.Kernels.KernelFactory.
    %
    %   Supported names (case-insensitive):
    %       'quadratic', 'l2'          →  QuadraticPenalty()
    %       'l1', 'tv'                 →  L1Penalty()
    %       'pnorm', 'hyperlaplacian'  →  PNormPenalty('P', p)
    %       'huber'                    →  HuberPenalty('Sigma', sigma)
    %       'gnc', 'clamped'           →  GNCPenalty('V', V)
    %
    %   Additional key-value pairs are forwarded to the penalty constructor.
    %
    %   Usage:
    %       pen = OceanMath.Regularization.PenaltyFactory.create('L1');
    %       pen = OceanMath.Regularization.PenaltyFactory.create('PNorm','P',0.8);
    %       pen = OceanMath.Regularization.PenaltyFactory.create('Huber','Sigma',0.5);
    %       pen = OceanMath.Regularization.PenaltyFactory.create('GNC','V',1.0);
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
