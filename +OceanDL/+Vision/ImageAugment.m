classdef ImageAugment < handle
    % IMAGEAUGMENT  Random training-time image transformations on canonical
    %   "SSCB" batches (H x W x C x N). Transformers lack the CNN's built-in
    %   locality prior and overfit small datasets; augmentation supplies
    %   invariances through data instead (DeiT, Touvron et al. 2021).
    %
    %   Each image independently:
    %     RandomCrop   pad by Padding pixels (edge replicate), crop back to H x W
    %                  at a random offset -> small translations
    %     FlipProb     horizontal mirror with this probability (faces: yes;
    %                  digits/text: NO -- a mirrored 2 is not a 2)
    %     Brightness   add U(-b, b) (images in [-1,1] or [0,1] scale)
    %     Contrast     scale deviations from the image mean by U(1-c, 1+c)
    %   Labels are unchanged. Apply only to training batches.
    %
    %   aug = OceanDL.Vision.ImageAugment('Padding', 4, 'FlipProb', 0.5, 'Seed', 1);
    %   Xb = aug.apply(Xb);

    properties (SetAccess = private)
        Padding (1,1) double = 0
        FlipProb (1,1) double = 0
        Brightness (1,1) double = 0
        Contrast (1,1) double = 0
        Stream
    end

    methods
        function obj = ImageAugment(opts)
            arguments
                opts.Padding (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.FlipProb (1,1) double {mustBeInRange(opts.FlipProb, 0, 1)} = 0
                opts.Brightness (1,1) double {mustBeNonnegative} = 0
                opts.Contrast (1,1) double {mustBeNonnegative} = 0
                opts.Seed = []
            end
            obj.Padding = opts.Padding; obj.FlipProb = opts.FlipProb;
            obj.Brightness = opts.Brightness; obj.Contrast = opts.Contrast;
            if isempty(opts.Seed)
                obj.Stream = RandStream.getGlobalStream();
            else
                obj.Stream = RandStream('mt19937ar', 'Seed', opts.Seed);
            end
        end

        function X = apply(obj, X)
            [H, W, ~, N] = size(X, 1:4);
            s = obj.Stream;
            p = obj.Padding;
            if p > 0
                rows = [ones(1, p), 1:H, H * ones(1, p)];            % edge replicate
                cols = [ones(1, p), 1:W, W * ones(1, p)];
                Xp = X(rows, cols, :, :);
                dy = randi(s, [0 2*p], 1, N); dx = randi(s, [0 2*p], 1, N);
                for n = 1:N
                    X(:, :, :, n) = Xp(dy(n) + (1:H), dx(n) + (1:W), :, n);
                end
            end
            if obj.FlipProb > 0
                flipMask = rand(s, 1, N) < obj.FlipProb;
                X(:, :, :, flipMask) = X(:, end:-1:1, :, flipMask);
            end
            if obj.Contrast > 0
                c = 1 + obj.Contrast * (2 * rand(s, 1, 1, 1, N) - 1);
                m = mean(X, [1 2 3]);
                X = (X - m) .* c + m;
            end
            if obj.Brightness > 0
                X = X + obj.Brightness * (2 * rand(s, 1, 1, 1, N) - 1);
            end
        end
    end
end
