function D = gen_load_data(name, quick, ckptDir)
% GEN_LOAD_DATA  Data for the generative-model comparisons (images "SSCB",
%   single precision, intensities in [0, 1]).
%
%   "usps"    real: USPS handwritten digits 16 x 16 (7291 train / 2007 test),
%             imported through UspsDigitsReader + DimFormat, rescaled [-1,1] -> [0,1].
%   "shapes"  synthetic challenge: ComplexShapeImages 32 x 32 (spirals, stars,
%             rings with a gap, crescents, crosses, hearts; random pose/scale),
%             8000 train (seed 1) / 2000 test (seed 2); cached via Checkpoint.
%   quick = true shrinks the sets for smoke runs.

switch name
    case "usps"
        [Xtr, ytr] = importUsps("train");
        [Xte, yte] = importUsps("test");
        classes = string(0:9);
    case "shapes"
        G = OceanData.Generators.Shapes2D.ComplexShapeImages;
        ntr = 8000; nte = 2000;
        if quick, ntr = 600; nte = 200; end
        data = OceanDL.Training.Checkpoint.run(ckptDir, sprintf('data_shapes32_%d_%d', ntr, nte), @() makeShapes(G, ntr, nte));
        Xtr = data.Xtr; ytr = data.ytr; Xte = data.Xte; yte = data.yte;
        classes = G.ClassNames;
    otherwise
        error('gen_load_data:Name', 'Unknown dataset "%s".', name);
end
if quick && name == "usps"
    Xtr = Xtr(:, :, :, 1:600); ytr = ytr(1:600); Xte = Xte(:, :, :, 1:200); yte = yte(1:200);
end
OceanDL.Layout.DimFormat.validate(Xtr, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(ytr));   % 1 grey channel
D = struct('Name', name, 'Xtr', single(Xtr), 'ytr', ytr(:), 'Xte', single(Xte), 'yte', yte(:), ...
    'ClassNames', classes, 'Size', size(Xtr, 1));
end

function [X, y] = importUsps(split)
[rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
X = OceanDL.Layout.DimFormat.toCanonical(OceanDL.Layout.DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape), fmt.Format);
X = (X + 1) / 2;
y = y + 1;
end

function s = makeShapes(G, ntr, nte)
[Xtr, ytr] = G.generate(ntr, 'Size', 32, 'Seed', 1);
[Xte, yte] = G.generate(nte, 'Size', 32, 'Seed', 2);
s = struct('Xtr', single(Xtr), 'ytr', ytr, 'Xte', single(Xte), 'yte', yte);
end
