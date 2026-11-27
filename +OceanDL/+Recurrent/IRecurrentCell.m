classdef (Abstract) IRecurrentCell < OceanDL.nn.Module
    % IRECURRENTCELL  One time step of a recurrent network (Goodfellow et al.
    %   2016, Sec. 10.1):   state_t = f(state_{t-1}, x_t; theta)
    %
    %   A CELL knows nothing about sequences: it maps one input column block
    %   x_t (D x N) and the previous state to the new state. Unrolling over
    %   time, masking, direction and truncation belong to RecurrentLayer --
    %   the cell/wrapper split used by PyTorch (LSTMCell / LSTM), Keras
    %   (RNN(cell)) and Flax (nn.RNN(cell)).
    %
    %   The same Parameters are used at EVERY step (parameter sharing across
    %   time); the autograd Engine sums their gradient contributions over the
    %   unrolled steps -- that sum IS backpropagation through time (Eq. 10.22-10.28).
    %
    % CONTRACT:
    %   state = cell.initialState(N, like)   struct of (HiddenSize x N) arrays/Tensors;
    %                                         field h is always the cell's output
    %   [h, state] = cell.step(x_t, state)   x_t: (InputSize x N), Tensor or array
    %   forward(x_t, state) == step(x_t, state)   (Module contract)
    %
    %   Layout: canonical "CB" per step (features x batch), see
    %   OceanDL.Layout.DimFormat.

    properties (SetAccess = protected)
        InputSize (1,1) double
        HiddenSize (1,1) double
    end

    methods (Abstract)
        [h, state] = step(obj, x, state)
    end

    methods
        function varargout = forward(obj, x, state)
            [varargout{1:nargout}] = obj.step(x, state);
        end

        function state = initialState(obj, N, like)
            % Zero state; subclasses with more state (LSTM's c) extend it.
            if nargin < 3, like = 0; end
            state = struct('h', zeros(obj.HiddenSize, N, 'like', OceanDL.Autograd.Tensor.raw(like)));
        end
    end

    methods (Static, Access = protected)
        function P = uniformParam(rows, cols, bound, stream, dataType)
            % U(-bound, bound) as a Parameter (PyTorch's recurrent-layer init
            % uses bound = 1/sqrt(HiddenSize) for every weight and bias).
            if isempty(stream)
                u = rand(rows, cols);
            else
                u = rand(stream, rows, cols);
            end
            P = OceanDL.nn.Parameter(cast(bound * (2 * u - 1), dataType));
        end

        function s = stream(seed)
            if isempty(seed)
                s = [];
            else
                s = RandStream('mt19937ar', 'Seed', seed);
            end
        end
    end
end
