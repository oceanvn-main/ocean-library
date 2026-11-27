classdef JpegSyntheticCompressor
    % JPEGSYNTHETICCOMPRESSOR  Software JPEG encoder/decoder with user-defined quantization.
    %
    %   Bypasses camera quality constraints by applying custom Quantization Tables
    %   directly to 8x8 DCT blocks. Produces severe, controllable blocking artifacts
    %   suitable as degradation inputs to POCS deblocking experiments.
    %
    %   Pipeline per block (encoding):
    %     1. Extract 8x8 spatial block
    %     2. Forward 2-D DCT-II
    %     3. Divide by Q table element-wise and round → integer indices q
    %
    %   Pipeline per block (decoding):
    %     1. Multiply q by Q table element-wise → quantized DCT coefficients c_hat
    %     2. Inverse 2-D DCT-II
    %     3. Clip to valid range
    %
    %   Usage:
    %     comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', 10);
    %     [img_degraded, q_indices, Q_table] = comp.compress(img_clean);
    %     img_reconstructed = comp.decompress(q_indices, Q_table);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19
    %   Ref:    Szeliski "Computer Vision" 2e, §4.3 (JPEG quantization, p. 234)

    properties
        % QualityFactor: JPEG quality [1=worst, 100=best]. Scales the base Q table.
        QualityFactor   (1,1) double {mustBeInRange(QualityFactor, 1, 100)} = 50

        % QTable: Custom 8x8 quantization table. If non-empty, overrides QualityFactor.
        QTable          double = []

        % BlockSize: DCT block size (standard JPEG uses 8).
        BlockSize       (1,1) double {mustBeMember(BlockSize, [4, 8, 16])} = 8
    end

    properties (Constant, Access = private)
        % Standard JPEG luminance quantization table (JPEG standard Annex K)
        BASE_QTABLE_LUMA = [
            16  11  10  16  24  40  51  61;
            12  12  14  19  26  58  60  55;
            14  13  16  24  40  57  69  56;
            14  17  22  29  51  87  80  62;
            18  22  37  56  68 109 103  77;
            24  35  55  64  81 104 113  92;
            49  64  78  87 103 121 120 101;
            72  92  95  98 112 100 103  99
        ]

        % Standard JPEG chrominance quantization table (JPEG standard Annex K)
        BASE_QTABLE_CHROMA = [
            17  18  24  47  99  99  99  99;
            18  21  26  66  99  99  99  99;
            24  26  56  99  99  99  99  99;
            47  66  99  99  99  99  99  99;
            99  99  99  99  99  99  99  99;
            99  99  99  99  99  99  99  99;
            99  99  99  99  99  99  99  99;
            99  99  99  99  99  99  99  99
        ]
    end

    methods
        function obj = JpegSyntheticCompressor(opts)
            % JPEGSYNTHETICCOMPRESSOR Construct with optional name-value arguments.
            arguments
                opts.QualityFactor  (1,1) double = 50
                opts.QTable         double = []
                opts.BlockSize      (1,1) double = 8
            end
            obj.QualityFactor = opts.QualityFactor;
            obj.QTable        = opts.QTable;
            obj.BlockSize     = opts.BlockSize;
        end

        function [img_out, q_all, Q_table] = compress(obj, img)
            % COMPRESS  Compress image using software DCT quantization.
            %
            %   [img_out, q_all, Q_table] = compress(img)
            %
            %   Inputs:
            %     img      — H x W or H x W x C numeric image (uint8 or double [0,1])
            %
            %   Outputs:
            %     img_out  — Degraded image (same class as input)
            %     q_all    — Cell array {C} of (nH x nW x 8 x 8) quantized DCT indices
            %     Q_table  — 8x8 quantization table used
            arguments
                obj
                img {mustBeNumeric}
            end

            is_uint8 = isa(img, 'uint8');

            % Work in double, range [0, 255] for DCT
            if is_uint8
                img_d = double(img);
            else
                % Assume [0, 1] double — scale to [0, 255] for standard DCT
                img_d = double(img) * 255.0;
            end

            [H, W, C] = size(img_d);
            B = obj.BlockSize;

            % Pad image to be divisible by block size
            H_pad = ceil(H / B) * B;
            W_pad = ceil(W / B) * B;
            img_padded = zeros(H_pad, W_pad, C);
            img_padded(1:H, 1:W, :) = img_d;
            % Replicate-pad right and bottom edges
            if H_pad > H
                img_padded(H+1:end, 1:W, :) = repmat(img_d(end, :, :), [H_pad-H, 1, 1]);
            end
            if W_pad > W
                img_padded(:, W+1:end, :) = repmat(img_padded(:, W, :), [1, W_pad-W, 1]);
            end

            % Get quantization table
            Q_table = obj.getQTable();

            nBH = H_pad / B;
            nBW = W_pad / B;

            img_out_padded = zeros(H_pad, W_pad, C);
            q_all = cell(1, C);

            for ch = 1:C
                q_ch = zeros(nBH, nBW, B, B);

                for bi = 1:nBH
                    for bj = 1:nBW
                        ri = (bi-1)*B + 1;
                        ci_start = (bj-1)*B + 1;

                        % Extract block, shift to [-128, 127] (standard JPEG level shift)
                        blk = img_padded(ri:ri+B-1, ci_start:ci_start+B-1, ch) - 128.0;

                        % Forward 2-D DCT
                        C_dct = dct2(blk);

                        % Quantize
                        q = round(C_dct ./ Q_table);
                        q_ch(bi, bj, :, :) = q; %Store interger q

                        % Dequantize → inverse DCT
                        C_hat = q .* Q_table;
                        blk_rec = idct2(C_hat) + 128.0;

                        img_out_padded(ri:ri+B-1, ci_start:ci_start+B-1, ch) = blk_rec;
                    end
                end

                q_all{ch} = q_ch;
            end

            % Crop back to original size
            img_out_d = img_out_padded(1:H, 1:W, :);

            % Restore original type
            if is_uint8
                img_out = uint8(max(0, min(255, round(img_out_d))));
            else
                img_out = max(0.0, min(1.0, img_out_d / 255.0));
            end
        end

        function img_out = decompress(obj, q_all, Q_table)
            % DECOMPRESS  Reconstruct image from stored quantized DCT indices.
            %
            %   img_out = decompress(q_all, Q_table)
            %
            %   Inputs:
            %     q_all    — Cell array from compress() output
            %     Q_table  — 8x8 quantization table from compress() output
            %
            %   Outputs:
            %     img_out  — Reconstructed image in double [0, 1]
            arguments
                obj
                q_all   cell
                Q_table (8,8) double
            end

            C = numel(q_all);
            B = obj.BlockSize;
            [nBH, nBW, ~, ~] = size(q_all{1});
            H_pad = nBH * B;
            W_pad = nBW * B;

            img_out_d = zeros(H_pad, W_pad, C);

            for ch = 1:C
                q_ch = q_all{ch};
                for bi = 1:nBH
                    for bj = 1:nBW
                        ri = (bi-1)*B + 1;
                        ci_start = (bj-1)*B + 1;

                        q = squeeze(q_ch(bi, bj, :, :));
                        C_hat = q .* Q_table;
                        blk_rec = idct2(C_hat) + 128.0;

                        img_out_d(ri:ri+B-1, ci_start:ci_start+B-1, ch) = blk_rec;
                    end
                end
            end

            img_out = max(0.0, min(1.0, img_out_d / 255.0));
        end

        function Q_table = getQTable(obj)
            % GETQTABLE  Return the effective 8x8 quantization table.
            %
            %   If QTable property is non-empty, returns it directly.
            %   Otherwise, scales the standard JPEG Annex K luma table by
            %   QualityFactor using the JPEG standard quality scaling formula.
            if ~isempty(obj.QTable)
                Q_table = obj.QTable;
                return;
            end

            % Standard JPEG quality scaling formula
            %   Quality 50 → scale 1.0 (unchanged base table)
            %   Quality  1 → scale 99.0 (extreme coarsening)
            %   Quality 100 → scale ~0.01 (near-lossless)
            if obj.QualityFactor < 50
                scale = 50.0 / obj.QualityFactor;
            else
                scale = 2.0 - obj.QualityFactor / 50.0;
            end

            Q_raw = floor(obj.BASE_QTABLE_LUMA * scale + 0.5);
            Q_table = max(1, min(255, Q_raw));
        end
    end
end
