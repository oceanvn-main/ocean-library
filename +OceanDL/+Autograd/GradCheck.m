classdef GradCheck
    % GRADCHECK  Compare autograd gradients with central finite differences
    %   (PyTorch torch.autograd.gradcheck).
    %
    %   For a function f of tensors returning Y, a fixed random projection R
    %   turns Y into the scalar  L = sum(Y .* R)  (so every output element is
    %   tested, not just their sum). Then for every input element x_i:
    %     numeric_i = (L(x + h e_i) - L(x - h e_i)) / (2h)       O(h^2) error
    %   is compared with the autograd value. Inputs are promoted to double.
    %
    % USAGE:
    %   r = OceanDL.Autograd.GradCheck.check(@(a, b) tanh(a * b), {randn(3,4), randn(4,2)});
    %   r.Passed, r.MaxAbsError, r.MaxRelError

    methods (Static)
        function v = evalProjected(fn, tensors, R)
            v = sum(OceanDL.Autograd.Tensor.raw(fn(tensors{:})) .* R, 'all');
        end

        function result = check(fn, inputs, opts)
            arguments
                fn function_handle
                inputs cell
                opts.Step (1,1) double = 1e-6
                opts.AbsTol (1,1) double = 1e-6
                opts.RelTol (1,1) double = 1e-5
                opts.Seed (1,1) double = 0
                opts.RequiresGrad logical = true(size(inputs))
            end
            import OceanDL.Autograd.*
            inputs = cellfun(@double, inputs, 'UniformOutput', false);

            % Autograd gradients.
            leaves = cell(size(inputs));
            for k = 1:numel(inputs)
                leaves{k} = Tensor(inputs{k}, 'RequiresGrad', opts.RequiresGrad(k));
            end
            Y = fn(leaves{:});
            R = randn(RandStream('mt19937ar', 'Seed', opts.Seed), size(Y.Data));
            L = sum(Y .* R, 'all');
            L.backward();

            % Numeric gradients (graph recording off: pure evaluation).
            guard = GradMode.disable(); %#ok<NASGU>
            asConst = @(vals) cellfun(@(v) Tensor(v), vals, 'UniformOutput', false);
            lossAt = @(vals) GradCheck.evalProjected(fn, asConst(vals), R);
            result = struct('Passed', true, 'MaxAbsError', 0, 'MaxRelError', 0, ...
                'Analytic', {cell(size(inputs))}, 'Numeric', {cell(size(inputs))});
            for k = find(opts.RequiresGrad)
                numeric = zeros(size(inputs{k}));
                for i = 1:numel(inputs{k})
                    plusVals = inputs;  plusVals{k}(i) = plusVals{k}(i) + opts.Step;
                    minusVals = inputs; minusVals{k}(i) = minusVals{k}(i) - opts.Step;
                    numeric(i) = (lossAt(plusVals) - lossAt(minusVals)) / (2 * opts.Step);
                end
                analytic = leaves{k}.Grad;
                if isempty(analytic)
                    analytic = zeros(size(numeric));
                end
                absErr = max(abs(analytic(:) - numeric(:)), [], 'all');
                relErr = max(abs(analytic(:) - numeric(:)) ./ max(1, abs(numeric(:))), [], 'all');
                if isempty(absErr), absErr = 0; relErr = 0; end
                result.MaxAbsError = max(result.MaxAbsError, absErr);
                result.MaxRelError = max(result.MaxRelError, relErr);
                result.Analytic{k} = analytic;
                result.Numeric{k} = numeric;
                if absErr > opts.AbsTol && relErr > opts.RelTol
                    result.Passed = false;
                end
            end
        end
    end
end
