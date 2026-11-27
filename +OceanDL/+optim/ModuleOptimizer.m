classdef ModuleOptimizer < handle
    % MODULEOPTIMIZER  Drive the existing per-parameter optimizers
    %   (AdamOptimizer, SGDOptimizer, RMSPropOptimizer, SGDMomentumOptimizer)
    %   over a list of autograd Parameters -- PyTorch's
    %   optimizer.zero_grad(); loss.backward(); optimizer.step() loop.
    %
    %   One optimizer instance per Parameter (each keeps its own moments),
    %   created from a factory so all share hyper-parameters:
    %     opt = OceanDL.optim.ModuleOptimizer(model.parameters(), ...
    %               @() OceanStats.Optim.AdamOptimizer('LR', 1e-3));
    %     opt.zeroGrad();  L = lossFn(model.forward(x));  L.backward();  opt.step();
    %
    %   Parameters without a gradient (not reached by this loss) are skipped
    %   and their optimizer state is left untouched.

    properties (SetAccess = private)
        Params cell
    end

    properties (SetAccess = {?OceanDL.optim.ModuleOptimizer})
        Optimizers cell
    end

    methods
        function obj = ModuleOptimizer(params, factory)
            arguments
                params cell
                factory function_handle
            end
            obj.Params = params;
            obj.Optimizers = cell(size(params));
            for k = 1:numel(params)
                obj.Optimizers{k} = factory();
            end
        end

        function step(obj)
            for k = 1:numel(obj.Params)
                p = obj.Params{k};
                if isempty(p.Grad)
                    continue;
                end
                p.Data = cast(obj.Optimizers{k}.step(p.Data, p.Grad), 'like', p.Data);
            end
        end

        function setLR(obj, lr)
            % SETLR  Set the learning rate of every per-parameter optimizer
            %   (driven by a schedule such as OceanDL.optim.WarmupCosine).
            for k = 1:numel(obj.Optimizers)
                obj.Optimizers{k}.LR = lr;
            end
        end

        function zeroGrad(obj)
            for k = 1:numel(obj.Params)
                obj.Params{k}.zeroGrad();
            end
        end

        function s = describeGroups(obj)
            s = sprintf('%d parameters, %d without weight decay', numel(obj.Params), ...
                sum(cellfun(@(o) isprop(o, 'WeightDecay') && o.WeightDecay == 0, obj.Optimizers)));
        end

        function s = describe(obj)
            if isempty(obj.Optimizers)
                s = 'ModuleOptimizer over 0 parameters';
                return;
            end
            s = sprintf('ModuleOptimizer over %d parameters: %s', numel(obj.Params), ...
                obj.Optimizers{1}.describe());
        end
    end
    methods (Static)
        function opt = forModule(module, factory, opts)
            % FORMODULE  One optimizer per named parameter, built by
            %   factory(weightDecay) -- weight decay is switched OFF for
            %   parameters that should not shrink towards 0: biases, LayerNorm
            %   gains/shifts, position embeddings and the [CLS] token (the
            %   standard AdamW recipe for Transformers). Everything else
            %   gets opts.WeightDecay.
            %     opt = OceanDL.optim.ModuleOptimizer.forModule(vit, ...
            %             @(wd) OceanStats.Optim.AdamWOptimizer('LR', 1e-3, 'WeightDecay', wd), 'WeightDecay', 0.05);
            arguments
                module OceanDL.nn.Module
                factory function_handle
                opts.WeightDecay (1,1) double = 0.05
                opts.NoDecay function_handle = @OceanDL.optim.ModuleOptimizer.isNoDecay
            end
            named = module.namedParameters();
            params = {named.Param};
            opt = OceanDL.optim.ModuleOptimizer(params, @() []);
            for k = 1:numel(named)
                wd = opts.WeightDecay;
                if opts.NoDecay(named(k).Name, named(k).Param)
                    wd = 0;
                end
                opt.Optimizers{k} = factory(wd);
            end
        end

        function tf = isNoDecay(name, ~)
            % Default rule, by parameter name (Module.namedParameters paths):
            %   biases "...Bias", any LayerNorm "...Norm...", position embeddings
            %   "...PosEmbed..." and the "ClsToken" are NOT decayed.
            name = string(name);
            tf = endsWith(name, "Bias") || contains(name, "Norm") || contains(name, "PosEmbed") ...
                || contains(name, "ClsToken");
        end
    end
end
