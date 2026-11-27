classdef DigitStrings
    % DIGITSTRINGS  Synthesise multi-digit "house number"-style images from a
    %   set of single-digit images: the data a CRNN + CTC reads left to right.
    %
    %   Each output image concatenates L ~ U{MinLength..MaxLength} digit
    %   images horizontally, separated by random gaps of 0..MaxGap background
    %   columns, starting at a random left offset, right-padded to a fixed
    %   width. Labels are the digit sequence in reading order -- NO per-pixel
    %   or per-column alignment is provided, which is exactly the setting CTC
    %   was designed for.
    %
    %   Input and output use the repo's canonical image layout (H x W x C x N,
    %   "SSCB"); the source images must already be imported/validated.
    %
    %   [X, labels, info] = DigitStrings.compose(images, digitLabels, 1000, ...
    %       'MinLength', 3, 'MaxLength', 5, 'Width', 96, 'Stream', RandStream('mt19937ar','Seed',1));
    %   X: 16 x 96 x 1 x 1000;  labels{m}: row of digit values (e.g. [4 0 7])

    methods (Static)
        function [X, labels, info] = compose(images, digitLabels, M, opts)
            arguments
                images (:,:,1,:) double
                digitLabels (:,1) double
                M (1,1) double {mustBePositive, mustBeInteger}
                opts.MinLength (1,1) double {mustBePositive, mustBeInteger} = 3
                opts.MaxLength (1,1) double {mustBePositive, mustBeInteger} = 5
                opts.MaxGap (1,1) double {mustBeNonnegative, mustBeInteger} = 3
                opts.Width (1,1) double {mustBePositive, mustBeInteger} = 96
                opts.Background (1,1) double = -1
                opts.Stream = RandStream.getGlobalStream()
            end
            [H, W, ~, N] = size(images, 1:4);
            if numel(digitLabels) ~= N
                error('OceanData:Generators:DigitStrings:LabelCount', '%d labels for %d images.', numel(digitLabels), N);
            end
            worst = opts.MaxLength * W + (opts.MaxLength - 1) * opts.MaxGap + opts.MaxGap;
            if worst > opts.Width
                error('OceanData:Generators:DigitStrings:TooNarrow', ...
                    'Width %d cannot hold %d digits of width %d with gaps (needs %d).', opts.Width, opts.MaxLength, W, worst);
            end
            s = opts.Stream;
            X = opts.Background * ones(H, opts.Width, 1, M);
            labels = cell(1, M);
            sources = cell(1, M);
            for m = 1:M
                L = randi(s, [opts.MinLength opts.MaxLength]);
                pick = randi(s, N, 1, L);
                x = 1 + randi(s, [0 opts.MaxGap]);
                for k = 1:L
                    X(:, x:x+W-1, 1, m) = images(:, :, 1, pick(k));
                    x = x + W + randi(s, [0 opts.MaxGap]);
                end
                labels{m} = digitLabels(pick).';
                sources{m} = pick;
            end
            info = struct('Sources', {sources}, 'DigitWidth', W);
        end
    end
end
