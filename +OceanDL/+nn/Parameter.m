classdef Parameter < OceanDL.Autograd.Tensor
    % PARAMETER  A trainable leaf Tensor (PyTorch torch.nn.Parameter).
    %
    %   Identical to a Tensor with RequiresGrad = true, but its TYPE marks it
    %   as model state: Module.parameters() collects every Parameter found in
    %   a module's properties, and optimizers update p.Data from p.Grad.

    methods
        function obj = Parameter(data, opts)
            arguments
                data = []
                opts.Name char = ''
            end
            obj@OceanDL.Autograd.Tensor(data, 'RequiresGrad', true, 'Name', opts.Name);
        end
    end
end
