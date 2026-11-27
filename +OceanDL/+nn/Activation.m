classdef Activation < OceanDL.nn.Module
    % ACTIVATION  Parameter-free nonlinearity as a module (for Sequential).
    %   Kind: "relu" | "gelu" | "tanh" | "sigmoid" | "softmax" | "logsoftmax"
    %   (softmax variants act over dimension 1 = features/classes).

    properties (SetAccess = protected)
        Kind (1,1) string
    end

    methods
        function obj = Activation(kind)
            arguments
                kind (1,1) string {mustBeMember(kind, ["relu", "gelu", "silu", "tanh", "sigmoid", "softmax", "logsoftmax"])}
            end
            obj.Kind = kind;
        end

        function y = forward(obj, x)
            F = OceanDL.Autograd.F;
            switch obj.Kind
                case "relu",       y = F.relu(x);
                case "gelu",       y = F.gelu(x);
                case "silu",       y = F.silu(x);
                case "tanh",       y = F.tanh(x);
                case "sigmoid",    y = F.sigmoid(x);
                case "softmax",    y = F.softmax(x, 1);
                case "logsoftmax", y = F.logSoftmax(x, 1);
            end
        end
    end
end
