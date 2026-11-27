classdef DeltaFeatures < OceanNLP.Speech.Core.IFeatureTransform
    % DELTAFEATURES  Stack static features with their regression (delta) coefficients.
    %
    %   t  = OceanNLP.Speech.Features.DeltaFeatures()                 % Width 2, Order 2
    %   t  = OceanNLP.Speech.Features.DeltaFeatures('Width', N, 'Order', R)
    %   fs = t.forward(featureSequence)    % (R+1)F x T : [c; dc; ddc; ...]
    %
    %   HTK REGRESSION FORMULA (HTK Book sec. 5.6; Furui 1986):
    %     d_t = sum_{n=1}^{N} n (c_{t+n} - c_{t-n}) / (2 sum_{n=1}^{N} n^2)
    %   with EDGE REPLICATION (c_t = c_1 for t < 1, c_t = c_T for t > T).
    %   Order r applies the same formula to the order r-1 output (acceleration =
    %   delta of delta, each stage replicating its own edges), as HTK does.
    %   Equals librosa.feature.delta(c, width=2N+1, order=1, mode='nearest')
    %   applied r times.
    %
    %   Input must be a FeatureSequence (':InputType' otherwise); time axis and
    %   FrameTimes are unchanged; FrequencyHz (if any) is repeated per block.
    %   T = 0 returns an empty (R+1)F x 0 sequence.

    properties (SetAccess = immutable)
        Width (1,1) double
        Order (1,1) double
    end

    methods
        function obj = DeltaFeatures(opts)
            arguments
                opts.Width (1,1) double {mustBeInteger, mustBePositive} = 2
                opts.Order (1,1) double {mustBeInteger, mustBeNonnegative} = 2
            end
            obj.Width = opts.Width;
            obj.Order = opts.Order;
        end

        function fs = forward(obj, input)
            arguments
                obj
                input
            end
            if ~isa(input, 'OceanNLP.Speech.Core.FeatureSequence')
                error('OceanNLP:Speech:Features:DeltaFeatures:InputType', ...
                    'DeltaFeatures.forward expects an OceanNLP.Speech.Core.FeatureSequence, got %s.', class(input));
            end
            c = input.Values;
            blocks = cell(1, obj.Order + 1);
            blocks{1} = c;
            for r = 1:obj.Order
                blocks{r + 1} = OceanNLP.Speech.Features.DeltaFeatures.regression(blocks{r}, obj.Width);
            end
            cfg = obj.config();
            cfg.Input = input.Config;
            f = input.FrequencyHz;
            if ~isempty(f)
                f = repmat(f, obj.Order + 1, 1);
            end
            fs = input.derive(vertcat(blocks{:}), 'FrequencyHz', f, ...
                'Units', input.Units + sprintf(" with %d delta order(s)", obj.Order), 'Config', cfg);
        end

        function cfg = config(obj)
            cfg = struct('Type', "DeltaFeatures", 'Width', obj.Width, 'Order', obj.Order, ...
                'Formula', "HTK regression, edge replication");
        end

        function d = outputDim(obj, inputDim)
            arguments
                obj
                inputDim (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            d = inputDim * (obj.Order + 1);
        end
    end

    methods (Static)
        function d = regression(c, N)
            % REGRESSION  One delta pass along dim 2 (time) with edge replication.
            arguments
                c (:,:) double
                N (1,1) double {mustBeInteger, mustBePositive}
            end
            T = size(c, 2);
            d = zeros(size(c));
            if T == 0
                return;
            end
            denom = 2 * sum((1:N) .^ 2);
            t = 1:T;
            for n = 1:N
                d = d + n * (c(:, min(t + n, T)) - c(:, max(t - n, 1)));
            end
            d = d / denom;
        end
    end

    methods (Static)
        function f = fromConfig(cfg)
            % FROMCONFIG  Inverse of config().
            arguments
                cfg (1,1) struct
            end
            f = OceanNLP.Speech.Features.DeltaFeatures('Width', cfg.Width, 'Order', cfg.Order);
        end
    end
end
