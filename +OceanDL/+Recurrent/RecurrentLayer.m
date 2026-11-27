classdef RecurrentLayer < OceanDL.nn.Module
    % RECURRENTLAYER  Unrolls a cell over time (the unfolded graph of
    %   Goodfellow Sec. 10.1) on a canonical "CBT" sequence batch.
    %
    %   [Y, state] = layer.forward(X)
    %   [Y, state] = layer.forward(X, 'Mask', m, 'InitialState', s0)
    %     X  (InputSize x N x T)   array or Tensor (layout "CBT", DimFormat)
    %     m  (N x T) logical       right-padded validity mask (1...1 0...0)
    %     Y  (HiddenSize x N x T)  the output h_t at every step (0 at padded steps)
    %     state                    final state struct (h, and c for LSTM)
    %
    %   The loop is written out -- one cell.step per time step -- and the
    %   autograd Engine differentiates the unrolled graph, i.e. performs
    %   BPTT automatically (the shared cell Parameters receive the SUM of
    %   their per-step gradients).
    %
    % MASKING (variable-length sequences, right padding):
    %   at a padded step the state is CARRIED unchanged and the output is 0:
    %     state_t = m_t .* cell(state_{t-1}, x_t) + (1 - m_t) .* state_{t-1}
    %   so each sequence's final state is the state after its LAST VALID
    %   step. For Reverse = true the loop runs t = T..1: padded steps come
    %   first and leave the initial state untouched, then the valid steps
    %   run from the sequence's own end back to its start -- exactly the
    %   reversed valid sequence, with no per-sequence bookkeeping.
    %
    % TRUNCATED BPTT (long streams): feed chunk k with
    %   'InitialState', RecurrentLayer.detachState(finalStateOfChunk_{k-1}).
    %
    % DIAGNOSTICS: TrackStates = true keeps every step's hidden state (with
    %   retainGrad) in StepStates, so after backward() the gradient norms
    %   ||dL/dh_t|| can be inspected over t (Sec. 10.7) --
    %   see Recurrent.Diagnostics.gradientNormsOverTime.
    %
    % COST NOTE: when X is itself a Tensor that requires grad (e.g. CNN
    %   features), each step's slice X(:,:,t) back-propagates a gradient of
    %   the full size of X, i.e. O(T * numel(X)) work -- fine for T up to a
    %   few hundred; a multi-output split op would remove it.

    properties (SetAccess = protected)
        Cell
        Reverse (1,1) logical = false
    end

    properties
        TrackStates (1,1) logical = false
        StepStates cell = {}
    end

    methods
        function obj = RecurrentLayer(cell, opts)
            arguments
                cell OceanDL.Recurrent.IRecurrentCell
                opts.Reverse (1,1) logical = false
                opts.TrackStates (1,1) logical = false
            end
            obj.Cell = cell;
            obj.Reverse = opts.Reverse;
            obj.TrackStates = opts.TrackStates;
        end

        function [Y, state] = forward(obj, X, opts)
            arguments
                obj
                X
                opts.Mask = []
                opts.InitialState = []
            end
            [D, N, T] = size(X, 1:3);
            if D ~= obj.Cell.InputSize
                error('OceanDL:Recurrent:RecurrentLayer:InputSize', ...
                    ['Expected "CBT" input with %d features on axis 1, got size [%s]. ', ...
                     'Convert imported data with OceanDL.Layout.DimFormat.toCanonical.'], ...
                    obj.Cell.InputSize, num2str(size(X)));
            end
            mask = opts.Mask;
            if ~isempty(mask) && ~isequal(size(mask), [N T])
                error('OceanDL:Recurrent:RecurrentLayer:MaskSize', ...
                    'Mask must be N x T = [%d %d], got [%s].', N, T, num2str(size(mask)));
            end
            state = opts.InitialState;
            if isempty(state)
                state = obj.Cell.initialState(N, X);
            end

            outs = cell(1, T);
            obj.StepStates = cell(1, T);
            order = 1:T;
            if obj.Reverse
                order = T:-1:1;
            end
            for t = order
                xt = X(:, :, t);
                [h, newState] = obj.Cell.step(xt, state);
                if ~isempty(mask)
                    m = cast(mask(:, t).', 'like', OceanDL.Autograd.Tensor.raw(h));  % (1 x N)
                    fields = fieldnames(newState);
                    for k = 1:numel(fields)
                        f = fields{k};
                        newState.(f) = m .* newState.(f) + (1 - m) .* state.(f);
                    end
                    h = m .* h;
                end
                state = newState;
                outs{t} = h;
                if obj.TrackStates && isa(state.h, 'OceanDL.Autograd.Tensor')
                    state.h.retainGrad();
                    obj.StepStates{t} = state.h;
                end
            end
            Y = cat(3, outs{:});
        end
    end

    methods (Static)
        function s = detachState(state)
            % DETACHSTATE  Cut the graph between chunks (truncated BPTT).
            s = state;
            f = fieldnames(state);
            for k = 1:numel(f)
                v = state.(f{k});
                if isa(v, 'OceanDL.Autograd.Tensor')
                    s.(f{k}) = v.detach();
                end
            end
        end
    end
end
