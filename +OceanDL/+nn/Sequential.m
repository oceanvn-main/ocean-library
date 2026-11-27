classdef Sequential < OceanDL.nn.Module
    % SEQUENTIAL  Chain of modules: y = m_k(...m_2(m_1(x)))  (PyTorch nn.Sequential).
    %
    %   net = OceanDL.nn.Sequential( ...
    %            OceanDL.nn.Linear(2, 16), OceanDL.nn.Activation("tanh"), ...
    %            OceanDL.nn.Linear(16, 1));
    %   Parameter names: "Layers{1}.Weight", "Layers{3}.Bias", ...

    properties (SetAccess = protected)
        Layers cell = {}
    end

    methods
        function obj = Sequential(varargin)
            for k = 1:numel(varargin)
                obj.add(varargin{k});
            end
        end

        function add(obj, m)
            arguments
                obj
                m OceanDL.nn.Module
            end
            obj.Layers{end+1} = m;
        end

        function y = forward(obj, x)
            y = x;
            for k = 1:numel(obj.Layers)
                y = obj.Layers{k}.forward(y);
            end
        end
    end
end
