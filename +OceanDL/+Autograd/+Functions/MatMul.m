classdef MatMul < OceanDL.Autograd.Function
    % MATMUL  Y = A * B   (matrix product, A: m x k, B: k x n)
    %
    %   VJP:  dL/dA = g * B.',   dL/dB = A.' * g
    %   (each entry Y_ij = sum_k A_ik B_kj, so dY_ij/dA_ik = B_kj -- summing over j gives g*B')

    methods
        function Y = forward(obj, A, B)
            obj.Saved.A = A;
            obj.Saved.B = B;
            Y = A * B;
        end

        function grads = backward(obj, g)
            grads = {g * obj.Saved.B.', obj.Saved.A.' * g};
        end
    end
end
