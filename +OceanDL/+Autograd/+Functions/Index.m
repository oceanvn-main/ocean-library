classdef Index < OceanDL.Autograd.Function
    % INDEX  Y = A(subs{:})   -- slicing, gathering, x(:), embedding lookup W(:, ids)
    %
    %   VJP (scatter-ADD):  dL/dA = zeros(size(A));  dL/dA(subs) += g
    %   "+=" matters: when an index repeats (the same word twice in a
    %   sentence -> the same embedding column read twice), both reads
    %   contribute, so their gradients must be SUMMED. A plain assignment
    %   dA(subs) = g would silently keep only the last one. The sum is done
    %   with accumarray over linear indices.
    %
    %   Every subscript form MATLAB accepts is normalised to explicit index
    %   vectors: ':' -> 1:n, logical -> find(...), numeric -> as is. With
    %   fewer subscripts than dimensions, the last subscript spans the
    %   trailing dimensions (MATLAB's rule), handled by folding the size.

    properties (SetAccess = private)
        Subs cell
    end

    methods
        function obj = Index(subs)
            obj.Subs = subs;
        end

        function Y = forward(obj, A)
            Y = A(obj.Subs{:});
            obj.Saved.InSize = size(A);
            obj.Saved.Like = A([]);
        end

        function grads = backward(obj, g)
            inSize = obj.Saved.InSize;
            if OceanDL.Autograd.Functions.Index.hasNoRepeats(obj.Subs)
                % Fast path (slices such as X(:,:,t)): every element is read at
                % most once, so plain assignment equals scatter-add.
                gIn = zeros(inSize, 'like', g);
                gIn(obj.Subs{:}) = g;
                grads = {gIn};
                return;
            end
            lin = OceanDL.Autograd.Functions.Index.linearIndices(obj.Subs, inSize);
            acc = accumarray(lin(:), double(g(:)), [prod(inSize) 1]);
            grads = {reshape(cast(acc, 'like', obj.Saved.Like), inSize)};
        end
    end

    methods (Static)
        function tf = hasNoRepeats(subs)
            % True when no element can be selected twice: every subscript is
            % ':' , a logical mask, or a numeric vector without duplicates.
            tf = true;
            for d = 1:numel(subs)
                s = subs{d};
                if ischar(s) || isstring(s) || islogical(s) || isscalar(s)
                    continue;
                end
                if numel(unique(s(:))) ~= numel(s)
                    tf = false;
                    return;
                end
            end
        end

        function lin = linearIndices(subs, inSize)
            % LINEARINDICES  Column-major linear index of every selected element,
            %   in the same order as the elements of A(subs{:}).
            k = numel(subs);
            if k == 1
                folded = prod(inSize);                    % linear indexing A(idx)
            else
                folded = [inSize(1:k-1), prod(inSize(k:end))];
                folded(end+1:k) = 1;
            end
            vecs = cell(1, k);
            for d = 1:k
                s = subs{d};
                if ischar(s) || isstring(s)               % ':'
                    vecs{d} = 1:folded(d);
                elseif islogical(s)
                    vecs{d} = find(s(:)).';
                else
                    vecs{d} = s(:).';
                end
            end
            if k == 1
                lin = vecs{1};
                return;
            end
            grids = cell(1, k);
            [grids{:}] = ndgrid(vecs{:});
            lin = sub2ind(folded, grids{:});
        end
    end
end
