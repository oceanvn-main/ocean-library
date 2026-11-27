classdef ActivationFactory
    % ACTIVATIONFACTORY  Create IActivation objects by name.
    %
    %   Centralizes activation construction so callers never hard-code
    %   class names. Follows the Factory pattern.
    %
    %   Supported names (case-insensitive):
    %     'linear'  — LinearActivation
    %     'relu'    — ReLUActivation
    %     'sigmoid' — SigmoidActivation
    %     'tanh'    — TanhActivation
    %     'softmax' — SoftmaxActivation
    %     'gelu'    — GELUActivation
    %     'swish'   — SwishActivation
    %     'sine'    — SineActivation  (NN-5, must be present for SIREN)
    %
    %   Usage:
    %     act = OceanDL.Activation.ActivationFactory.create('relu');
    %     A   = act.forward(Z);

    methods (Static)
        function act = create(name)
            % CREATE  Return an IActivation object for the given name.
            %
            %   act = ActivationFactory.create('gelu')
            import OceanDL.Activation.*

            switch lower(strtrim(name))
                case 'linear'
                    act = LinearActivation();
                case 'relu'
                    act = ReLUActivation();
                case 'sigmoid'
                    act = SigmoidActivation();
                case 'tanh'
                    act = TanhActivation();
                case 'softmax'
                    act = SoftmaxActivation();
                case 'gelu'
                    act = GELUActivation();
                case 'swish'
                    act = SwishActivation();
                case 'sine'
                    act = SineActivation();   % Added in NN-5
                otherwise
                    error('OceanDL:Activation:ActivationFactory:Unknown', ...
                        'Unknown activation "%s". Valid: linear, relu, sigmoid, tanh, softmax, gelu, swish, sine.', name);
            end
        end

        function names = validNames()
            % VALIDNAMES  Return cell of all recognised activation name strings.
            names = {'linear','relu','sigmoid','tanh','softmax','gelu','swish','sine'};
        end
    end
end
