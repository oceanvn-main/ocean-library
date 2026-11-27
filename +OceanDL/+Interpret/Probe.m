classdef Probe
    % PROBE  Read single units of a traceable model and project them back to
    %   the input (Szeliski 2nd ed. Sec. 5.4, Fig. 5.47 left; Zeiler & Fergus 2014).
    %
    %   a = Probe.activation(model, X, layer)            raw (H x W x C x N), no graph
    %   g = Probe.unitGradient(model, x, layer, c, r, k, rule)
    %       d a(r, k, c) / d x for ONE image x (H x W x Cin) under a backward rule:
    %       "standard" = saliency (the true gradient), "deconvnet", "guided"
    %       (OceanDL.Autograd.BackpropRule). With r = k = [] the target is the
    %       whole channel (sum over positions) instead of one unit; channel = []
    %       targets ALL channels at that position (used for receptive fields).
    %   parts = Probe.partsFor(layer)   "encoder" when only the bottom-up pass is needed.

    methods (Static)
        function p = partsFor(layer)
            if startsWith(string(layer), "enc."), p = "encoder"; else, p = "all"; end
        end

        function a = activation(model, X, layer)
            layer = char(layer);
            acts = OceanDL.Autograd.GradMode.noGrad(@() model.trace(X, 'Parts', OceanDL.Interpret.Probe.partsFor(layer)));
            a = OceanDL.Autograd.Tensor.raw(acts(layer));
        end

        function g = unitGradient(model, x, layer, channel, row, col, rule)
            arguments
                model
                x
                layer
                channel double
                row = []
                col = []
                rule (1,1) string = "standard"
            end
            g = OceanDL.Autograd.BackpropRule.run(rule, ...
                @() OceanDL.Interpret.Probe.gradient(model, x, char(layer), channel, row, col));
        end
    end

    methods (Static, Access = private)
        function g = gradient(model, x, layer, channel, row, col)
            X = OceanDL.Autograd.Tensor(reshape(x, size(x, 1), size(x, 2), size(x, 3), 1), 'RequiresGrad', true);
            acts = model.trace(X, 'Parts', OceanDL.Interpret.Probe.partsFor(layer));
            a = acts(layer);
            seed = zeros(size(OceanDL.Autograd.Tensor.raw(a)), 'like', OceanDL.Autograd.Tensor.raw(a));
            if isempty(channel), channel = 1:size(seed, 3); end
            if isempty(row), seed(:, :, channel, 1) = 1; else, seed(row, col, channel, 1) = 1; end
            a.backward(seed);
            g = X.Grad;
            if isempty(g), g = zeros(size(OceanDL.Autograd.Tensor.raw(X)), 'like', OceanDL.Autograd.Tensor.raw(X)); end   % unit independent of x
            OceanDL.Interpret.Probe.clearParamGrads(model);
        end

        function clearParamGrads(model)
            if isa(model, 'OceanDL.nn.Module')
                P = model.parameters();
                for k = 1:numel(P), P{k}.zeroGrad(); end
            end
        end
    end
end
