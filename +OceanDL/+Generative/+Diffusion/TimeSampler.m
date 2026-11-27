classdef TimeSampler
    % TIMESAMPLER  Distribution of training times t (where the loss is spent):
    %     "uniform"        t ~ U(TMin, 1)                         (DDPM)
    %     "logitnormal"    t = sigmoid(Mean + Std n)              (SD3: m = 0, s = 1;
    %                      concentrates on mid noise levels)
    %     "lognormal"      ln sigma ~ N(Mean, Std^2), t = tFromSigma  (EDM: -1.2, 1.2)
    %   Samples are clipped to [TMin, 1] (t = 0 has no noise to learn).
    %   'Discrete' L > 0 (with "uniform"): t = i / L with i ~ U{1, ..., L} -- the
    %   textbook DDPM training of Lai et al. Eq. (2.2.10) / Ho et al. 2020
    %   (integer steps), instead of continuous t.

    properties (SetAccess = private)
        Kind (1,1) string
        Mean (1,1) double
        Std (1,1) double
        TMin (1,1) double
        Discrete (1,1) double      % 0 = continuous t; L = integer steps i / L
    end

    methods
        function obj = TimeSampler(kind, opts)
            arguments
                kind (1,1) string {mustBeMember(kind, ["uniform", "logitnormal", "lognormal"])} = "uniform"
                opts.Mean (1,1) double = NaN
                opts.Std (1,1) double = NaN
                opts.TMin (1,1) double = 1e-5
                opts.Discrete (1,1) double {mustBeNonnegative, mustBeInteger} = 0
            end
            if opts.Discrete > 0 && kind ~= "uniform"
                error('OceanDL:Generative:Diffusion:TimeSampler:Discrete', '''Discrete'' applies to "uniform" only.');
            end
            defaults = struct('uniform', [0 1], 'logitnormal', [0 1], 'lognormal', [-1.2 1.2]);
            d = defaults.(kind);
            obj.Kind = kind;
            obj.Mean = d(1); if ~isnan(opts.Mean), obj.Mean = opts.Mean; end
            obj.Std = d(2);  if ~isnan(opts.Std), obj.Std = opts.Std; end
            obj.TMin = opts.TMin;
            obj.Discrete = opts.Discrete;
        end

        function t = sample(obj, n, stream, schedule)
            switch obj.Kind
                case "uniform"
                    if obj.Discrete > 0
                        t = randi(stream, obj.Discrete, 1, n) / obj.Discrete;      % i / L, i in 1..L
                        return;
                    end
                    t = rand(stream, 1, n);
                case "logitnormal"
                    t = 1 ./ (1 + exp(-(obj.Mean + obj.Std * randn(stream, 1, n))));
                case "lognormal"
                    if ~ismethod(schedule, 'tFromSigma')
                        error('OceanDL:Generative:Diffusion:TimeSampler:Schedule', ...
                            '"lognormal" samples sigma and needs a schedule with tFromSigma (EDM).');
                    end
                    t = schedule.tFromSigma(exp(obj.Mean + obj.Std * randn(stream, 1, n)));
            end
            t = min(1, max(obj.TMin, t));
        end
    end
end
