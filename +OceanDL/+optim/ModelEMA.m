classdef ModelEMA < handle
    % MODELEMA  Exponential moving average of a module's parameters (Polyak
    %   averaging) -- standard for diffusion models: samples are drawn with
    %   the EMA weights, which are much smoother than the last iterate.
    %
    %     theta_ema <- d theta_ema + (1 - d) theta          after every optimiser step
    %   'Warmup' true uses d_n = min(Decay, (1 + n) / (10 + n)) so early
    %   averages are not dominated by the random initialisation.
    %
    %   ema = OceanDL.optim.ModelEMA(net, 'Decay', 0.999);
    %   ... opt.step(); ema.update();
    %   ema.applyTo(net); X = sample(...); ema.restore(net);   % sample with EMA weights

    properties (SetAccess = private)
        Decay (1,1) double
        Warmup (1,1) logical
        Updates (1,1) double = 0
        Shadow              % containers.Map name -> array
        Backup = []
    end

    properties (Access = private)
        Module
    end

    methods
        function obj = ModelEMA(module, opts)
            arguments
                module OceanDL.nn.Module
                opts.Decay (1,1) double {mustBeInRange(opts.Decay, 0, 1)} = 0.999
                opts.Warmup (1,1) logical = false
            end
            obj.Module = module;
            obj.Decay = opts.Decay;
            obj.Warmup = opts.Warmup;
            obj.Shadow = module.stateDict();
        end

        function d = currentDecay(obj)
            d = obj.Decay;
            if obj.Warmup, d = min(d, (1 + obj.Updates) / (10 + obj.Updates)); end
        end

        function update(obj)
            d = obj.currentDecay();
            named = obj.Module.namedParameters();
            for k = 1:numel(named)
                key = named(k).Name;
                obj.Shadow(key) = d * obj.Shadow(key) + (1 - d) * named(k).Param.Data;
            end
            obj.Updates = obj.Updates + 1;
        end

        function applyTo(obj, module)
            % Swap the EMA weights in (keeps a backup for restore()).
            if nargin < 2, module = obj.Module; end
            obj.Backup = module.stateDict();
            module.loadStateDict(obj.Shadow);
        end

        function restore(obj, module)
            if nargin < 2, module = obj.Module; end
            if isempty(obj.Backup)
                error('OceanDL:optim:ModelEMA:NoBackup', 'restore() without a preceding applyTo().');
            end
            module.loadStateDict(obj.Backup);
            obj.Backup = [];
        end
    end
end
