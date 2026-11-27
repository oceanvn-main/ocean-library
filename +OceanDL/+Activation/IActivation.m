classdef (Abstract) IActivation < handle
    % IACTIVATION  Abstract interface for all activation functions.
    %
    %   Every activation function in the NeuralNet engine must implement
    %   this interface. Objects are stateless — they act as pure strategy
    %   objects (Strategy Pattern).
    %
    %   USAGE:
    %     act = OceanDL.Activation.ActivationFactory.create('relu');
    %     A   = act.forward(Z);    % apply activation
    %     dA  = act.grad(Z);       % elementwise gradient w.r.t. pre-activation Z
    %
    %   CONVENTION:
    %     Z — (d x N) pre-activation matrix  (column = sample, row = neuron)
    %     A — (d x N) post-activation matrix, same shape as Z
    %
    %   See also: OceanDL.Activation.ActivationFactory

    methods (Abstract)
        A  = forward(obj, Z)
        % FORWARD  Apply the activation function element-wise.
        %   Z — (d x N) pre-activation
        %   A — (d x N) post-activation, same shape

        dA = grad(obj, Z)
        % GRAD  Elementwise derivative d(activation)/dZ evaluated at Z.
        %   For Softmax: returns the diagonal of the Jacobian per sample.
        %   Z  — (d x N)
        %   dA — (d x N)

        s  = name(obj)
        % NAME  Return a lowercase string identifier, e.g. 'relu'.
    end
end
