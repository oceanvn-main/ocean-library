classdef AffectNetReader
    % AFFECTNETREADER  Facial-expression images (this repo's AffectNet copy:
    %   96 x 96 RGB, 8 classes) with an explicit DATA-QUALITY step.
    %
    %   Layout of Data/Datasets/AffectNet:
    %     Train/<class>/imageNNN.jpg, Test/<Class>/imageNNN.jpg  (folder case differs)
    %     labels.csv: columns pth ("<class>/<file>"), label, relFCs (a confidence)
    %
    %   MEASURED PROBLEMS (2026-09-23), handled here, reported in `report`:
    %     1. LEAKAGE: 1,688 Test files are byte-identical to Train files.
    %        -> 'DedupTest' (default true) drops them from Test by MD5 of the
    %           file bytes, so test accuracy is measured on unseen images.
    %     2. LABEL CONFLICT: labels.csv disagrees with the folder name for 37%
    %        of its rows (with the same confidence as agreeing rows). Which is
    %        right is unknown, so the source is an explicit option:
    %           "agreeing" (default) keep images where folder == csv label
    %           "folder"   folder name is the label (all images)
    %           "csv"      labels.csv is the label (images without a row dropped)
    %     3. labels.csv covers only part of the images and its pth has no split;
    %        rows are matched by lower-case "<class>/<file>".
    %     4. MIXED SOURCES (confound): 'happy' and 'neutral' are almost entirely
    %        FFHQ images (ffhq_*.png: train 2295/2340 and 2630/2758, test 100%),
    %        all other classes AffectNet JPEGs. A model can separate those two
    %        classes by image SOURCE (compression, alignment) instead of
    %        expression. 'Sources' selects which sources to keep
    %        ("affectnet", "ffhq"; default both) and report.PerClassSource shows
    %        the composition, so the confound is never silent.
    %
    %   PROTOCOL 'Split':
    %     "provided"  the folders' Train/Test (test de-duplicated against train)
    %     "resplit"   pool Train + Test, keep ONE copy of every distinct image
    %                 (MD5), then a seeded stratified split (TestFraction per
    %                 class). Leak-free by construction; restores classes whose
    %                 provided test images were all copies (sad).
    %   Clean 6-class protocol (the repo default for face experiments):
    %     read(split, 'Split', "resplit", 'Sources', "affectnet", 'Labels', "agreeing", ...
    %          'Classes', ["anger" "contempt" "disgust" "fear" "sad" "surprise"])
    %
    %   IMAGE SIZE is an option ('ImageSize', [h w]; [] = native 96 x 96),
    %   resampled with the anti-aliased OceanMath ImageResampler (one batched
    %   call). Output is canonical "SSCB" (H x W x C x N) single in [0,1] --
    %   imread's H x W x C per image already has the canonical axis meaning.
    %
    %   [X, y, info] = OceanData.Adapters.AffectNetReader.read("train", 'ImageSize', [48 48]);
    %   info.Classes, info.Report

    properties (Constant)
        Classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    end

    methods (Static)
        function p = cleanProtocol()
            % Name-value options of the clean 6-class protocol (see header):
            %   p = AffectNetReader.cleanProtocol();
            %   X = AffectNetReader.read("train", p{:}, 'ImageSize', [16 16]);
            p = {'Split', "resplit", 'Sources', "affectnet", 'Labels', "agreeing", ...
                 'Classes', ["anger" "contempt" "disgust" "fear" "sad" "surprise"], 'TestFraction', 0.2, 'SplitSeed', 0};
        end

        function [X, y, info] = read(split, opts)
            arguments
                split (1,1) string {mustBeMember(split, ["train", "test"])}
                opts.Labels (1,1) string {mustBeMember(opts.Labels, ["agreeing", "folder", "csv"])} = "agreeing"
                opts.Sources (1,:) string {mustBeMember(opts.Sources, ["affectnet", "ffhq"])} = ["affectnet", "ffhq"]
                opts.Classes (1,:) string = OceanData.Adapters.AffectNetReader.Classes   % subset; y indexes THIS list
                opts.Split (1,1) string {mustBeMember(opts.Split, ["provided", "resplit"])} = "provided"
                opts.TestFraction (1,1) double {mustBeInRange(opts.TestFraction, 0, 1, 'exclusive')} = 0.2
                opts.SplitSeed (1,1) double = 0
                opts.ImageSize double = []                   % [] = native
                opts.ResizeFilter (1,1) string = "Lanczos"
                opts.Grayscale (1,1) logical = false
                opts.DedupTest (1,1) logical = true
                opts.MaxImages (1,1) double = Inf
                opts.Seed (1,1) double = 0                   % subsampling order when MaxImages < all
                opts.Root (1,1) string = ""
                opts.CacheDir (1,1) string = ""              % "" = no cache
            end
            root = opts.Root;
            if root == ""
                repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
                root = fullfile(repo, 'Data', 'Datasets', 'AffectNet');
            end
            cacheFile = "";
            if opts.CacheDir ~= ""
                key = sprintf('affectnet_%s_%s_%s_%s_%s_%g_%d_%s_%s_%d_%d_%g_%d.mat', split, opts.Split, opts.Labels, ...
                    strjoin(opts.Sources, '+'), strjoin(extractBefore(opts.Classes, 3), ''), opts.TestFraction, opts.SplitSeed, ...
                    mat2str(opts.ImageSize), opts.ResizeFilter, opts.Grayscale, opts.DedupTest, opts.MaxImages, opts.Seed);
                key = regexprep(key, '[\[\]\s;]', '_');
                cacheFile = fullfile(opts.CacheDir, key);
                if isfile(cacheFile)
                    S = load(cacheFile); X = S.X; y = S.y; info = S.info;
                    return;
                end
            end

            unknown = setdiff(opts.Classes, OceanData.Adapters.AffectNetReader.Classes);
            if ~isempty(unknown)
                error('OceanData:Adapters:AffectNetReader:UnknownClass', 'Unknown class(es): %s', strjoin(unknown, ', '));
            end
            if opts.Split == "resplit"
                % Pool BOTH folders, keep one copy of every distinct image (first
                % occurrence by MD5), split later -- leakage is impossible by construction.
                [f1, l1, r1] = OceanData.Adapters.AffectNetReader.listSplit(root, "train");
                [f2, l2, r2] = OceanData.Adapters.AffectNetReader.listSplit(root, "test");
                files = [f1; f2]; folderLabel = [l1; l2]; rel = [r1; r2];
                report = struct('Split', split, 'Protocol', "resplit", 'Labels', opts.Labels, 'Listed', numel(files));
                [~, first] = unique(OceanData.Adapters.AffectNetReader.md5(files), 'stable');
                report.DuplicatesRemoved = numel(files) - numel(first);
                files = files(first); folderLabel = folderLabel(first); rel = rel(first);
            else
                [files, folderLabel, rel] = OceanData.Adapters.AffectNetReader.listSplit(root, split);
                report = struct('Split', split, 'Protocol', "provided", 'Labels', opts.Labels, 'Listed', numel(files));
                report.DuplicatesRemoved = 0;
            end

            % 1. Leakage (provided split): drop test files byte-identical to any training file.
            if opts.Split == "provided" && split == "test" && opts.DedupTest
                trFiles = OceanData.Adapters.AffectNetReader.listSplit(root, "train");
                trHash = OceanData.Adapters.AffectNetReader.md5(trFiles);
                teHash = OceanData.Adapters.AffectNetReader.md5(files);
                dup = ismember(teHash, trHash);
                report.DuplicatesRemoved = nnz(dup);
                files = files(~dup); folderLabel = folderLabel(~dup); rel = rel(~dup);
            end

            % 2. Label source.
            T = readtable(fullfile(root, 'labels.csv'), 'TextType', 'string');
            [found, loc] = ismember(rel, lower(T.pth));
            csvLabel = strings(size(rel));
            csvLabel(found) = lower(T.label(loc(found)));
            report.WithCsvRow = nnz(found);
            report.FolderCsvDisagree = nnz(found & csvLabel ~= folderLabel);
            switch opts.Labels
                case "folder"
                    keep = true(size(files)); label = folderLabel;
                case "csv"
                    keep = found; label = csvLabel;
                case "agreeing"
                    keep = found & csvLabel == folderLabel; label = folderLabel;
            end
            files = files(keep); label = label(keep);
            source = OceanData.Adapters.AffectNetReader.sourceOf(files);
            report.RemovedBySource = nnz(~ismember(source, opts.Sources));
            files = files(ismember(source, opts.Sources)); label = label(ismember(source, opts.Sources));
            source = source(ismember(source, opts.Sources));
            inClasses = ismember(label, opts.Classes);
            report.RemovedByClass = nnz(~inClasses);
            files = files(inClasses); label = label(inClasses); source = source(inClasses);
            [~, y] = ismember(label, opts.Classes);

            % Stratified re-split: per class, a seeded TestFraction goes to test.
            if opts.Split == "resplit"
                rs = RandStream('mt19937ar', 'Seed', opts.SplitSeed);
                isTest = false(size(y));
                for c = 1:numel(opts.Classes)
                    idx = find(y == c);
                    idx = idx(randperm(rs, numel(idx)));
                    isTest(idx(1:round(opts.TestFraction * numel(idx)))) = true;
                end
                part = isTest == (split == "test");
                files = files(part); y = y(part); source = source(part);
            end
            report.PerClassSourceEligible = OceanData.Adapters.AffectNetReader.composition(y, source, opts.Classes);
            if isfinite(opts.MaxImages) && numel(files) > opts.MaxImages
                pick = sort(randperm(RandStream('mt19937ar', 'Seed', opts.Seed), numel(files), opts.MaxImages));
                files = files(pick); y = y(pick); source = source(pick);
            end
            report.PerClassSource = OceanData.Adapters.AffectNetReader.composition(y, source, opts.Classes);

            % 3. Pixels: imread (H x W x C) stacked -> SSCB, optional resize.
            X = OceanData.Adapters.AffectNetReader.loadImages(files, opts);
            y = y(:);
            report.Kept = numel(y);
            report.PerClass = sum(report.PerClassSource{:, :}, 2).';
            info = struct('Classes', opts.Classes, 'Report', report, ...
                'Format', "SSCB", 'Range', [0 1], 'Files', {files});
            if cacheFile ~= ""
                if ~isfolder(opts.CacheDir), mkdir(opts.CacheDir); end
                save(cacheFile, 'X', 'y', 'info', '-v7.3');
            end
        end
    end

    methods (Static, Access = private)
        function [files, folderLabel, rel] = listSplit(root, split)
            base = fullfile(root, char(upper(extractBefore(split, 2)) + extractAfter(split, 1)));
            d = [dir(fullfile(base, '*', '*.jpg')); dir(fullfile(base, '*', '*.jpeg')); dir(fullfile(base, '*', '*.png'))];
            files = string(fullfile({d.folder}, {d.name})).';
            [~, cls] = fileparts(string({d.folder}).');
            folderLabel = lower(cls);
            rel = folderLabel + "/" + lower(string({d.name}).');
        end

        function T = composition(y, source, classes)
            % Images per class (rows) and source (columns); robust to empty sets.
            K = numel(classes);
            edges = 0.5:1:K + 0.5;
            T = array2table([histcounts(y(source == "affectnet"), edges).', histcounts(y(source == "ffhq"), edges).'], ...
                'VariableNames', {'affectnet', 'ffhq'}, 'RowNames', cellstr(classes));
        end

        function src = sourceOf(files)
            [~, name] = fileparts(files);
            src = repmat("affectnet", size(files));
            src(startsWith(lower(name), "ffhq")) = "ffhq";
        end

        function h = md5(files)
            md = java.security.MessageDigest.getInstance('MD5');
            h = strings(numel(files), 1);
            for k = 1:numel(files)
                fid = fopen(files(k), 'r'); bytes = fread(fid, Inf, '*uint8'); fclose(fid);
                md.reset();
                d = typecast(md.digest(bytes), 'uint8');
                h(k) = string(sprintf('%02x', d));
            end
        end

        function X = loadImages(files, opts)
            N = numel(files);
            first = imread(files(1));
            [H, W, C] = size(first, 1:3);
            if opts.Grayscale, Cout = 1; else, Cout = 3; end
            X = zeros(H, W, Cout, N, 'single');
            for k = 1:N
                I = single(imread(files(k))) / 255;
                if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
                if opts.Grayscale
                    I = 0.2989 * I(:, :, 1) + 0.5870 * I(:, :, 2) + 0.1140 * I(:, :, 3);
                end
                X(:, :, :, k) = I;
            end
            if ~isempty(opts.ImageSize) && ~isequal(opts.ImageSize(:).', [H W])
                rs = OceanMath.Image.Resampling.ImageResampler('TargetSize', opts.ImageSize, 'FilterType', opts.ResizeFilter);
                X = reshape(rs.apply(reshape(double(X), H, W, [])), opts.ImageSize(1), opts.ImageSize(2), Cout, N);
                X = single(min(1, max(0, X)));
            end
        end
    end
end
