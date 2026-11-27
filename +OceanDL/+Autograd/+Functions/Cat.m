classdef Cat < OceanDL.Autograd.Function
    % CAT  Y = cat(dim, A1, A2, ..., Ak)   -- e.g. [h; x] inside an RNN cell
    %
    %   VJP: slice g back into the k pieces along dim, in order:
    %     dL/dAi = g( ..., offset_i + (1:size(Ai,dim)), ... )

    properties (SetAccess = private)
        Dim (1,1) double
    end

    methods
        function obj = Cat(dim)
            obj.Dim = dim;
        end

        function Y = forward(obj, varargin)
            obj.Saved.Lengths = cellfun(@(a) size(a, obj.Dim), varargin);
            Y = cat(obj.Dim, varargin{:});
        end

        function grads = backward(obj, g)
            len = obj.Saved.Lengths;
            grads = cell(1, numel(len));
            subs = repmat({':'}, 1, max(ndims(g), obj.Dim));
            offset = 0;
            for i = 1:numel(len)
                subs{obj.Dim} = offset + (1:len(i));
                grads{i} = g(subs{:});
                offset = offset + len(i);
            end
        end
    end
end
