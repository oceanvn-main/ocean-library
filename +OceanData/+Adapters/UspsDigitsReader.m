classdef UspsDigitsReader
    % USPSDIGITSREADER  Read the USPS handwritten digits (Hastie, Tibshirani &
    %   Friedman, ESL data "zip.train" / "zip.test") EXACTLY as stored.
    %
    %   File format: one observation per text line: the digit label (0..9)
    %   followed by 256 grey values in [-1, 1] of a 16 x 16 image stored
    %   ROW BY ROW (row-major, pixel k = 16*r + c, 0-based).
    %
    %   This reader deliberately does NOT convert anything: it returns the
    %   source representation (observations as rows, row-major pixels) plus
    %   a description of it. Converting to the repo's canonical image layout
    %   (H x W x C x N, "SSCB") is the separate, explicit import step, e.g.
    %     [rows, y, fmt] = UspsDigitsReader.read("train");
    %     buf = reshape(rows.', 1, []);                       % whole set as one C-order buffer
    %     X = DimFormat.fromRowMajor(buf, fmt.Shape);         % (N, 16, 16, 1): index meaning restored
    %     X = DimFormat.toCanonical(X, fmt.Format);           % -> 16 x 16 x 1 x N
    %
    %   (Note: OceanData.Generators.ZipGenerator points at Data\zip, but the
    %   files live in Data\Datasets\zip -- this reader resolves the path.)

    methods (Static)
        function [rows, labels, fmt] = read(split, opts)
            arguments
                split (1,1) string {mustBeMember(split, ["train", "test"])} = "train"
                opts.Root (1,1) string = ""
            end
            root = opts.Root;
            if root == ""
                repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
                root = fullfile(repo, 'Data', 'Datasets', 'zip');
            end
            path = fullfile(root, "zip." + split, "zip." + split);
            if ~isfile(path)
                error('OceanData:Adapters:UspsDigitsReader:NotFound', 'USPS file not found: %s', path);
            end
            data = readmatrix(path, 'FileType', 'text');
            data = data(:, ~all(isnan(data), 1));                 % trailing-delimiter column, if any
            if size(data, 2) ~= 257
                error('OceanData:Adapters:UspsDigitsReader:Format', ...
                    'Expected 257 columns (label + 256 pixels), found %d.', size(data, 2));
            end
            labels = data(:, 1);
            rows = data(:, 2:end);
            N = size(rows, 1);
            fmt = struct( ...
                'Description', "rows: N x 256, one image per row, pixels row-major (k = 16 r + c)", ...
                'Shape', [N 16 16 1], ...          % the C-order shape of the whole buffer
                'Format', "BSSC", ...              % axis meaning: observation, row (H), column (W), channel
                'Range', [-1 1]);
        end
    end
end
