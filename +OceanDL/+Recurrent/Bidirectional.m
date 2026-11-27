classdef Bidirectional < OceanDL.nn.Module
    % BIDIRECTIONAL  Two recurrent layers over the same sequence, one forward
    %   in time and one backward (Schuster & Paliwal 1997) -- Goodfellow
    %   Sec. 10.3: the output at step t sees the PAST through h_fwd and the
    %   FUTURE through h_bwd.
    %
    %     Y_t = [h_fwd_t ; h_bwd_t]       Merge "concat"  (2H x N x T)
    %     Y_t =  h_fwd_t + h_bwd_t        Merge "sum"     ( H x N x T)
    %
    %   The two directions have SEPARATE cells (their own Parameters), built
    %   from a factory so they share hyper-parameters:
    %     bi = Bidirectional(@() LSTMCell(D, H));

    properties (SetAccess = protected)
        ForwardLayer
        BackwardLayer
        Merge (1,1) string = "concat"
    end

    methods
        function obj = Bidirectional(cellFactory, opts)
            arguments
                cellFactory function_handle
                opts.Merge (1,1) string {mustBeMember(opts.Merge, ["concat", "sum"])} = "concat"
            end
            obj.ForwardLayer = OceanDL.Recurrent.RecurrentLayer(cellFactory());
            obj.BackwardLayer = OceanDL.Recurrent.RecurrentLayer(cellFactory(), 'Reverse', true);
            obj.Merge = opts.Merge;
        end

        function n = outputSize(obj)
            n = obj.ForwardLayer.Cell.HiddenSize * (1 + (obj.Merge == "concat"));
        end

        function [Y, states] = forward(obj, X, opts)
            arguments
                obj
                X
                opts.Mask = []
            end
            [Yf, sf] = obj.ForwardLayer.forward(X, 'Mask', opts.Mask);
            [Yb, sb] = obj.BackwardLayer.forward(X, 'Mask', opts.Mask);
            if obj.Merge == "concat"
                Y = cat(1, Yf, Yb);
            else
                Y = Yf + Yb;
            end
            states = struct('Forward', sf, 'Backward', sb);
        end
    end
end
