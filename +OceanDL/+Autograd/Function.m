classdef (Abstract) Function < handle
    % FUNCTION  A differentiable operation AND the graph node it leaves behind
    %   (PyTorch: torch.autograd.Function together with its grad_fn node).
    %
    %   MODERN AUTOGRAD IN ONE PARAGRAPH (define-by-run, reverse mode):
    %     Every op on Tensors is executed by a fresh Function object. Its
    %     forward() computes the result on RAW arrays and saves on itself
    %     whatever backward() will need (buffer-on-self, the repo's layer
    %     convention). If any input requires a gradient, the output Tensor
    %     points back to this Function (Tensor.GradFn) and the Function
    %     points to its input Tensors (Inputs). These pointers ARE the
    %     computation graph -- no global tape, nothing to reset: the graph
    %     lives exactly as long as the Tensors that reference it.
    %     Engine.backward() later walks the graph from the loss towards the
    %     leaves, calling backward() = the op's vector-Jacobian product (VJP):
    %       given  g = dL/dY,  return  dL/dX_k = g * dY/dX_k   for every input k.
    %
    %   WRITING A NEW OP (the whole extension contract):
    %     classdef MyOp < OceanDL.Autograd.Function
    %       methods
    %         function Y = forward(obj, A, B)        % raw arrays in, raw array out
    %           obj.Saved.A = A;  obj.Saved.B = B;  % keep what the VJP needs
    %           Y = ...;
    %         end
    %         function g = backward(obj, gY)         % one gradient per input
    %           g = {dLdA, dLdB};                    % [] for "no gradient"
    %         end
    %     end, end
    %     Y = MyOp().apply(tensorA, tensorB);        % constants are auto-wrapped
    %   Configuration that is NOT differentiated (a dim, a size, a stride)
    %   goes to the constructor, never to apply().
    %
    %   BROADCASTING: backward() may return a gradient in the BROADCAST
    %   (output) shape; the Engine sums it back to each input's shape
    %   (Broadcast.reduceTo). Ops therefore never re-implement that rule.
    %
    % See also: OceanDL.Autograd.Tensor, OceanDL.Autograd.Engine

    properties (SetAccess = protected)
        Inputs cell = {}               % input Tensors = incoming graph edges
        NeedsInputGrad logical = false(1, 0)
        Id (1,1) double = 0            % creation counter: a valid topological order
        Released (1,1) logical = false % saved buffers freed by a backward pass
    end

    properties (Access = protected)
        Saved struct = struct()        % buffers for backward(), freed after use
    end

    methods (Abstract)
        Y = forward(obj, varargin)
        grads = backward(obj, gradOut)
    end

    methods
        function out = apply(obj, varargin)
            % APPLY  Run forward() on the raw data of the inputs and, if any
            %   input requires grad (and grad mode is on), wire the graph.
            import OceanDL.Autograd.Tensor
            inputs = cell(size(varargin));
            raw = cell(size(varargin));
            needs = false(size(varargin));
            for k = 1:numel(varargin)
                t = Tensor.wrap(varargin{k});
                inputs{k} = t;
                raw{k} = t.Data;
                needs(k) = t.RequiresGrad;
            end
            Y = obj.forward(raw{:});
            if any(needs) && OceanDL.Autograd.GradMode.isEnabled()
                obj.Inputs = inputs;
                obj.NeedsInputGrad = needs;
                obj.Id = OceanDL.Autograd.Function.nextId();
                out = Tensor(Y, 'GradFn', obj);
            else
                obj.Saved = struct();            % nothing will ever call backward()
                out = Tensor(Y);
            end
        end

        function release(obj)
            % RELEASE  Free saved buffers after backward (PyTorch default,
            %   retain_graph=false): a second backward through this node errors.
            obj.Saved = struct();
            obj.Released = true;
        end

        function s = opName(obj)
            parts = split(class(obj), '.');
            s = char(parts{end});
        end
    end

    methods (Static)
        function id = nextId()
            persistent counter
            if isempty(counter)
                counter = 0;
            end
            counter = counter + 1;
            id = counter;
        end
    end
end
