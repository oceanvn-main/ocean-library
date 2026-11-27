classdef SineActivation < OceanDL.Activation.IActivation
    % SINEACTIVATION  Sinusoidal activation for SIREN networks (Sitzmann et al., 2020).
    %
    %   FORMULA:
    %     forward: A = sin(omega_0 * Z)
    %     grad:   dA = omega_0 * cos(omega_0 * Z)
    %
    %   WEIGHT INITIALIZATION (SIREN-specific, Supplement B):
    %     First layer:  W ~ U(-1/d_in, 1/d_in)
    %     Hidden layers: W ~ U(-sqrt(6/d_in)/omega_0, sqrt(6/d_in)/omega_0)
    %     This preserves the distribution of activations across layers.
    %
    %   omega_0 = 30 is the recommended default from the original paper.
    %   It controls the frequency of the signal the first layer can represent.
    %
    %   Ref: Sitzmann, V., Martel, J.N.P., Bergman, A.W., Lindell, D.B.,
    %        & Wetzstein, G. (2020). Implicit Neural Representations with
    %        Periodic Activation Functions. NeurIPS 2020. arXiv:2006.09661
    %
    %   Usage:
    %     act = OceanDL.Activation.SineActivation();
    %     act = OceanDL.Activation.SineActivation('Omega0', 30);
    %     A   = act.forward(Z);
    %     dA  = act.grad(Z);

    properties
        Omega0 (1,1) double = 30   % Frequency multiplier (omega_0 in the paper)
    end

    methods
        function obj = SineActivation(opts)
            arguments
                opts.Omega0 (1,1) double = 30
            end
            obj.Omega0 = opts.Omega0;
        end

        function A = forward(obj, Z)
            A = sin(obj.Omega0 * Z);
        end

        function dA = grad(obj, Z)
            dA = obj.Omega0 * cos(obj.Omega0 * Z);
        end

        function s = name(~)
            s = 'sine';
        end
    end
end
