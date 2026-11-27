classdef SequencePool < OceanDL.nn.Module
    % SEQUENCEPOOL  Many-to-one head input: summarise a (H x N x T) output
    %   sequence into one (H x N) vector per sequence (Goodfellow Sec. 10.2,
    %   "read an entire sequence, then produce a single output", Fig. 10.5).
    %
    %   Mode "last"  h at each sequence's LAST VALID step (lengths from the mask)
    %        "mean"  average over valid steps:  sum_t m_t h_t / sum_t m_t
    %        "max"   max over valid steps (padded steps can never win)
    %   Without a mask every step is valid.

    properties (SetAccess = protected)
        Mode (1,1) string = "last"
    end

    methods
        function obj = SequencePool(mode)
            arguments
                mode (1,1) string {mustBeMember(mode, ["last", "mean", "max"])} = "last"
            end
            obj.Mode = mode;
        end

        function P = forward(obj, Y, mask)
            if nargin < 3
                mask = [];
            end
            [H, N, T] = size(Y, 1:3);
            if isempty(mask)
                mask = true(N, T);
            end
            lengths = sum(mask, 2);
            if any(lengths == 0)
                error('OceanDL:Recurrent:SequencePool:EmptySequence', 'Every sequence needs at least one valid step.');
            end
            like = OceanDL.Autograd.Tensor.raw(Y);
            m = reshape(cast(mask, 'like', like), 1, N, T);            % broadcast over H
            switch obj.Mode
                case "last"
                    % Gather Y(:, n, lengths(n)) for every n with one linear index.
                    [hh, nn] = ndgrid(1:H, 1:N);
                    lin = sub2ind([H N T], hh, nn, repmat(lengths.', H, 1));
                    P = reshape(Y(lin(:)), H, N);
                case "mean"
                    P = sum(Y .* m, 3) ./ reshape(cast(lengths, 'like', like), 1, N);
                case "max"
                    P = max(Y + (1 - m) * cast(-1e30, 'like', like), [], 3);
            end
        end
    end
end
