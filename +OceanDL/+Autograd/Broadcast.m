classdef Broadcast
    % BROADCAST  The one rule every broadcasting op needs in reverse mode.
    %
    %   MATLAB implicit expansion (like NumPy/PyTorch broadcasting) stretches
    %   a size-1 dimension to match the other operand, e.g. a bias b (H x 1)
    %   added to activations Z (H x N):  Y(:, n) = Z(:, n) + b  for every n.
    %   Because b was USED N times, its gradient is the SUM of the N copies:
    %     dL/db = sum_n dL/dY(:, n)
    %   In general: sum the incoming gradient over every dimension where the
    %   input had size 1 but the gradient does not, then reshape.
    %   Engine applies this to every input gradient, so individual ops can
    %   return gradients in the broadcast (output) shape.

    methods (Static)
        function g = reduceTo(g, targetSize)
            gs = size(g);
            if isequal(gs, targetSize)
                return;
            end
            nd = max(numel(gs), numel(targetSize));
            gs(end+1:nd) = 1;
            ts = targetSize;
            ts(end+1:nd) = 1;
            dims = find(ts == 1 & gs ~= 1);
            if ~isempty(dims)
                g = sum(g, dims);
            end
            if ~isequal(size(g, 1:nd), ts)
                error('OceanDL:Autograd:Broadcast:ShapeMismatch', ...
                    'Gradient of size [%s] cannot be reduced to input size [%s].', ...
                    num2str(size(g)), num2str(targetSize));
            end
            g = reshape(g, targetSize);
        end
    end
end
