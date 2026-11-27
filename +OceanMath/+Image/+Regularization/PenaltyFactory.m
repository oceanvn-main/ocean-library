classdef PenaltyFactory
    % PENALTYFACTORY  Static factory for OceanMath Image robust penalty functions.
    %
    %   Provides a centralized mechanism to instantiate any IRobustPenalty by name,
    %   mirroring the pattern of OceanMath.Signal.Kernels.KernelFactory.
    %
    %   Supported names (case-insensitive):
    %       'quadratic'  →  QuadraticPenalty()
    %       'l1', 'tv'   →  L1Penalty()
    %       'pnorm'      →  PNormPenalty('P', p)
    %       'huber'      →  HuberPenalty('Sigma', sigma)
    %       'gnc'        →  GNCPenalty('V', V)
    %
    %   Usage:
    %       pen = OceanMath.Image.Regularization.PenaltyFactory.create('L1');
    %       pen = OceanMath.Image.Regularization.PenaltyFactory.create('PNorm','P',0.8);
    %       pen = OceanMath.Image.Regularization.PenaltyFactory.create('Huber','Sigma',0.5);
    %       pen = OceanMath.Image.Regularization.PenaltyFactory.create('GNC','V',1.0);
    %
    %   Author: Ocean Library  |  2026-08-05

    methods (Static)
        function pen = create(name, varargin)
            % CREATE  Instantiate a named IRobustPenalty.
            %
            %   pen = PenaltyFactory.create(name)
            %   pen = PenaltyFactory.create(name, 'Key', value, ...)
            %
            %   Additional key-value pairs are forwarded to the penalty constructor.
            %
            %   Example:
            %       pen = PenaltyFactory.create('pnorm', 'P', 1.5)

            import OceanMath.Image.Regularization.*

            switch lower(name)
                case 'quadratic'
                    pen = QuadraticPenalty(varargin{:});
                case {'l1', 'tv', 'totalvariation'}
                    pen = L1Penalty(varargin{:});
                case 'pnorm'
                    pen = PNormPenalty(varargin{:});
                case 'huber'
                    pen = HuberPenalty(varargin{:});
                case 'gnc'
                    pen = GNCPenalty(varargin{:});
                otherwise
                    error('PenaltyFactory:UnknownPenalty', ...
                        ['Unknown penalty: "%s". Supported: ' ...
                         'quadratic, l1, tv, pnorm, huber, gnc.'], name);
            end
        end
    end
end
