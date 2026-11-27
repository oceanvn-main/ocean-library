classdef BatchMatMul < OceanDL.Autograd.Function
    % BATCHMATMUL  C = op(A) * op(B) for every PAGE (dims 3, 4, ...) at once:
    %   pagemtimes(A, TransA, B, TransB), op(X) = X or X.'.
    %   Page dims broadcast (implicit expansion), e.g. one weight page used
    %   for every sample. This is the workhorse of attention: one T x T score
    %   matrix per (head, sample) page, all in one call instead of a loop.
    %
    % VJP (G = dL/dC, per page; derived from C = A B and transposes):
    %   none/none   C = A B      dA = G B.'     dB = A.' G
    %   T/none      C = A.' B    dA = B G.'     dB = A G
    %   none/T      C = A B.'    dA = G B       dB = G.' A
    %   T/T         C = A.' B.'  dA = B.' G.'   dB = G.' A.'
    %   Each is itself a pagemtimes; gradients of broadcast pages are summed
    %   back by the Engine (Broadcast.reduceTo).

    properties (SetAccess = private)
        TransA (1,1) logical = false
        TransB (1,1) logical = false
    end

    methods
        function obj = BatchMatMul(transA, transB)
            if nargin > 0, obj.TransA = transA; end
            if nargin > 1, obj.TransB = transB; end
        end

        function C = forward(obj, A, B)
            obj.Saved.A = A;
            obj.Saved.B = B;
            C = pagemtimes(A, OceanDL.Autograd.Functions.BatchMatMul.flag(obj.TransA), ...
                           B, OceanDL.Autograd.Functions.BatchMatMul.flag(obj.TransB));
        end

        function grads = backward(obj, G)
            A = obj.Saved.A; B = obj.Saved.B;
            n = 'none'; t = 'transpose';
            if ~obj.TransA && ~obj.TransB
                dA = pagemtimes(G, n, B, t);  dB = pagemtimes(A, t, G, n);
            elseif obj.TransA && ~obj.TransB
                dA = pagemtimes(B, n, G, t);  dB = pagemtimes(A, n, G, n);
            elseif ~obj.TransA && obj.TransB
                dA = pagemtimes(G, n, B, n);  dB = pagemtimes(G, t, A, n);
            else
                dA = pagemtimes(B, t, G, t);  dB = pagemtimes(G, t, A, t);
            end
            grads = {dA, dB};
        end
    end

    methods (Static)
        function f = flag(tf)
            if tf, f = 'transpose'; else, f = 'none'; end
        end
    end
end
