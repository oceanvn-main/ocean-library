function info = analyzeJPEGLayers(filepath)
%ANALYZEJPEGLAYERS  Decode a JPEG file into its 4 internal representation layers.
%
%  Exposes the full decoding stack of a JPEG image:
%
%  Layer 1 — Raw compressed bitstream bytes from disk
%  Layer 2 — Quantization table (luma/chroma) + quantized DCT coefficients
%             of the first 8x8 luminance block
%  Layer 3 — Decoded pixel matrix (first 8x8 block, uint8 RGB)
%  Layer 4 — Binary + hexadecimal encoding of raw pixel memory
%
%  Syntax:
%    info = analyzeJPEGLayers(filepath)
%    analyzeJPEGLayers(filepath)          % print report, no output
%
%  Input:
%    filepath  - char | string  Full path to a JPEG file (.jpg, .jpeg)
%
%  Output:
%    info  - struct with fields:
%      .Layer1.FileBytes      uint8 vector of all raw file bytes
%      .Layer1.FileSizeBytes  scalar
%      .Layer1.HexPreview     char  first 32 bytes as 'FF D8 FF E0 ...'
%
%      .Layer2.QuantLuma      8x8 double  luma quantization table
%      .Layer2.QuantChroma    8x8 double  chroma quantization table (if present)
%      .Layer2.DCT_Raw        8x8 double  DCT coefficients of first 8x8 luma block
%      .Layer2.DCT_Quantized  8x8 double  quantized DCT coefficients
%
%      .Layer3.Pixels8x8      8x8x3 uint8  first 8x8 pixel block (RGB)
%      .Layer3.FullImage      H x W x C uint8  full decoded image
%
%      .Layer4.PixelRGB       1x3 uint8  first pixel [R G B]
%      .Layer4.BinaryR        char  8-bit binary string for R
%      .Layer4.BinaryG        char  8-bit binary string for G
%      .Layer4.BinaryB        char  8-bit binary string for B
%      .Layer4.RawHexBytes    char  hex pairs of first 32 pixel bytes
%
%  Example:
%    info = analyzeJPEGLayers('my_photo.jpg');
%
%  See also: imread, dct2, imfinfo

% ---- Validate input -------------------------------------------------------
arguments
    filepath (1,1) string
end

if ~isfile(filepath)
    error('analyzeJPEGLayers:fileNotFound', 'File not found: %s', filepath);
end
[~, ~, ext] = fileparts(filepath);
if ~ismember(lower(ext), {'.jpg', '.jpeg'})
    warning('analyzeJPEGLayers:notJPEG', ...
        'File extension is not .jpg/.jpeg — results may be incorrect.');
end

% ===========================================================================
% LAYER 1 — Raw compressed bytes
% ===========================================================================
fid  = fopen(filepath, 'rb');
rawBytes = fread(fid, '*uint8');
fclose(fid);

info.Layer1.FileBytes      = rawBytes;
info.Layer1.FileSizeBytes  = numel(rawBytes);
info.Layer1.HexPreview     = formatHexBytes_(rawBytes(1:min(32, end)));

% ===========================================================================
% LAYER 2 — Quantization tables + DCT coefficients
% ===========================================================================
[quantLuma, quantChroma] = parseQuantTables_(rawBytes);
info.Layer2.QuantLuma    = quantLuma;
info.Layer2.QuantChroma  = quantChroma;

% Decode image, convert to YCbCr, extract first 8x8 luma block, apply DCT
fullImage = imread(filepath);
info.Layer3.FullImage = fullImage;

% Pad image to at least 8x8 if very small
if size(fullImage,1) < 8 || size(fullImage,2) < 8
    fullImage = padarray(fullImage, [8,8], 'replicate', 'post');
end

% Convert to YCbCr and extract Y (luma) channel
if size(fullImage,3) == 3
    ycbcr   = rgb2ycbcr(fullImage);
    lumaFull = double(ycbcr(1:8, 1:8, 1));
else
    lumaFull = double(fullImage(1:8, 1:8, 1));
end

% Centre-shift (JPEG level shifts by 128 before DCT)
lumaShifted = lumaFull - 128;

% 2D DCT of first 8x8 luma block
dctRaw = dct2(lumaShifted);
info.Layer2.DCT_Raw = dctRaw;

% Quantize — element-wise division then round (JPEG encoder step)
if ~isempty(quantLuma)
    dctQuantized = round(dctRaw ./ quantLuma);
else
    % Fallback: standard JPEG baseline luma table (quality ~50)
    dctQuantized = round(dctRaw ./ defaultQuantLuma_());
end
info.Layer2.DCT_Quantized = dctQuantized;

% ===========================================================================
% LAYER 3 — Decoded pixel matrix (first 8x8 block)
% ===========================================================================
decoded = imread(filepath);
if size(decoded,1) < 8 || size(decoded,2) < 8
    decoded = padarray(decoded, [8,8], 'replicate', 'post');
end
if size(decoded,3) == 1
    % Grayscale — replicate to 3 channels for consistent shape
    decoded = repmat(decoded, [1,1,3]);
end
info.Layer3.Pixels8x8 = decoded(1:8, 1:8, :);

% ===========================================================================
% LAYER 4 — Binary encoding + raw memory bytes
% ===========================================================================
px = squeeze(info.Layer3.Pixels8x8(1,1,:));   % first pixel [R G B]
info.Layer4.PixelRGB  = uint8(px);
info.Layer4.BinaryR   = dec2bin(px(1), 8);
info.Layer4.BinaryG   = dec2bin(px(2), 8);
info.Layer4.BinaryB   = dec2bin(px(3), 8);

% Raw memory bytes of the 8x8 pixel block as continuous buffer
block   = info.Layer3.Pixels8x8;       % 8x8x3 uint8
rawMem  = typecast(block(:), 'uint8'); % flat byte array
info.Layer4.RawHexBytes = formatHexBytes_(rawMem(1:min(32, end)));

% ===========================================================================
% Print report (always; suppress with nargout check if needed)
% ===========================================================================
printReport_(info, filepath);

end % analyzeJPEGLayers

% ===========================================================================
% Local helpers
% ===========================================================================

function [quantLuma, quantChroma] = parseQuantTables_(rawBytes)
%PARSEQUANTTABLES_  Extract DQT segments from raw JPEG bytes.
%
%  JPEG DQT marker: FF DB
%  Segment layout:
%    [FF][DB][Len Hi][Len Lo][Pq|Tq][64 bytes zigzag-ordered table] ...
%
%  Pq = precision (0 = 8-bit values, 1 = 16-bit values)
%  Tq = table ID  (0 = luma, 1 = chroma)

quantLuma   = [];
quantChroma = [];

% Zigzag-to-raster index map (1-based, for an 8x8 block)
zigzagIdx = zigzagOrder_();

i = 1;
n = numel(rawBytes);
while i <= n - 1
    if rawBytes(i) == 0xFF && rawBytes(i+1) == 0xDB
        % Found DQT marker
        if i + 3 > n, break; end
        segLen = double(rawBytes(i+2)) * 256 + double(rawBytes(i+3));
        % segLen includes the 2-byte length field itself
        segEnd = i + 1 + segLen;          % last byte of segment
        ptr    = i + 4;                   % first byte after length

        while ptr < segEnd && ptr <= n
            pqtq = rawBytes(ptr);
            pq   = bitshift(pqtq, -4);    % upper nibble: precision
            tq   = bitand(pqtq, 0x0F);    % lower nibble: table ID
            ptr  = ptr + 1;

            if pq == 0
                % 8-bit table values
                if ptr + 63 > n, break; end
                tableZigzag = double(rawBytes(ptr : ptr+63));
                ptr = ptr + 64;
            else
                % 16-bit table values (2 bytes per entry)
                if ptr + 127 > n, break; end
                hi  = double(rawBytes(ptr : 2 : ptr+126));
                lo  = double(rawBytes(ptr+1 : 2 : ptr+127));
                tableZigzag = hi * 256 + lo;
                ptr = ptr + 128;
            end

            % Reorder from zigzag to raster 8x8
            table8x8 = zeros(8, 8);
            for k = 1:64
                [r, c] = ind2sub([8, 8], zigzagIdx(k));
                table8x8(r, c) = tableZigzag(k);
            end

            if tq == 0
                quantLuma   = table8x8;
            elseif tq == 1
                quantChroma = table8x8;
            end
        end
        i = segEnd + 1;
    else
        i = i + 1;
    end
end

% Fallback defaults if no DQT found
if isempty(quantLuma)
    quantLuma = defaultQuantLuma_();
end
if isempty(quantChroma)
    quantChroma = defaultQuantChroma_();
end
end


function idx = zigzagOrder_()
%ZIGZAGORDER_  Return the JPEG zigzag scan order as linear indices (1-based).
%  idx(k) is the linear index in an 8x8 row-major matrix for zigzag position k.
%
%  JPEG standard Annex K zigzag:
%   0  1  5  6 14 15 27 28
%   2  4  7 13 16 26 29 42
%   3  8 12 17 25 30 41 43
%   9 11 18 24 31 40 44 53
%  10 19 23 32 39 45 52 54
%  20 22 33 38 46 51 55 60
%  21 34 37 47 50 56 59 61
%  35 36 48 49 57 58 62 63

% 0-based (row,col) pairs in zigzag order — JPEG standard
pairs0 = [ ...
    0,0; 0,1; 1,0; 2,0; 1,1; 0,2; 0,3; 1,2; ...
    2,1; 3,0; 4,0; 3,1; 2,2; 1,3; 0,4; 0,5; ...
    1,4; 2,3; 3,2; 4,1; 5,0; 6,0; 5,1; 4,2; ...
    3,3; 2,4; 1,5; 0,6; 0,7; 1,6; 2,5; 3,4; ...
    4,3; 5,2; 6,1; 7,0; 7,1; 6,2; 5,3; 4,4; ...
    3,5; 2,6; 1,7; 2,7; 3,6; 4,5; 5,4; 6,3; ...
    7,2; 7,3; 6,4; 5,5; 4,6; 3,7; 4,7; 5,6; ...
    6,5; 7,4; 7,5; 6,6; 5,7; 6,7; 7,6; 7,7];

% Convert to 1-based linear indices (column-major for MATLAB sub2ind)
idx = zeros(64, 1);
for k = 1:64
    r = pairs0(k,1) + 1;  % row 1-based
    c = pairs0(k,2) + 1;  % col 1-based
    idx(k) = sub2ind([8,8], r, c);
end
end


function Q = defaultQuantLuma_()
%DEFAULTQUANTLUMA_  JPEG standard baseline luma quantization table (quality ~50).
Q = [ ...
    16, 11, 10, 16, 24,  40,  51,  61; ...
    12, 12, 14, 19, 26,  58,  60,  55; ...
    14, 13, 16, 24, 40,  57,  69,  56; ...
    14, 17, 22, 29, 51,  87,  80,  62; ...
    18, 22, 37, 56, 68,  109, 103, 77; ...
    24, 35, 55, 64, 81,  104, 113, 92; ...
    49, 64, 78, 87, 103, 121, 120, 101; ...
    72, 92, 95, 98, 112, 100, 103, 99];
end


function Q = defaultQuantChroma_()
%DEFAULTQUANTCHROMA_  JPEG standard baseline chroma quantization table (quality ~50).
Q = [ ...
    17, 18, 24, 47, 99, 99, 99, 99; ...
    18, 21, 26, 66, 99, 99, 99, 99; ...
    24, 26, 56, 99, 99, 99, 99, 99; ...
    47, 66, 99, 99, 99, 99, 99, 99; ...
    99, 99, 99, 99, 99, 99, 99, 99; ...
    99, 99, 99, 99, 99, 99, 99, 99; ...
    99, 99, 99, 99, 99, 99, 99, 99; ...
    99, 99, 99, 99, 99, 99, 99, 99];
end


function hex = formatHexBytes_(bytes)
%FORMATHEXBYTES_  Format uint8 array as 'XX XX XX ...' string.
hex = strtrim(sprintf('%02X ', bytes));
end


function printReport_(info, filepath)
%PRINTREPORT_  Print a formatted 4-layer JPEG inspection report to stdout.

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║           JPEG Layer Inspector — %s\n', filepath);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n\n');

% ---- Layer 1 ------------------------------------------------------------
fprintf('─── Layer 1 — Raw Compressed Bitstream ─────────────────────────────\n');
fprintf('  File size  : %d bytes\n', info.Layer1.FileSizeBytes);
fprintf('  Header     : %s\n', info.Layer1.HexPreview(1:min(23,end)));
fprintf('  First 32B  : %s\n\n', info.Layer1.HexPreview);

% ---- Layer 2 ------------------------------------------------------------
fprintf('─── Layer 2 — Quantization Tables + Quantized DCT Coefficients ─────\n');
fprintf('  Quant table (luma):\n');
printMatrix8x8_(info.Layer2.QuantLuma, '%4d');
fprintf('\n  DCT coefficients, first 8x8 luma block (after IDCT-shift):\n');
printMatrix8x8_(info.Layer2.DCT_Raw, '%7.1f');
fprintf('\n  Quantized DCT coefficients (DCT / quant_table, rounded):\n');
printMatrix8x8_(info.Layer2.DCT_Quantized, '%5d');
fprintf('\n');

% ---- Layer 3 ------------------------------------------------------------
fprintf('─── Layer 3 — Decoded Pixel Matrix (first 8x8 block, uint8 RGB) ────\n');
pix = info.Layer3.Pixels8x8;
[h, w, ~] = size(pix);
fprintf('  Block shape : %d x %d x 3\n', h, w);
fprintf('  Pixel [1,1] : R=%3d  G=%3d  B=%3d\n', ...
    pix(1,1,1), pix(1,1,2), pix(1,1,3));
fprintf('  Red channel (8x8):\n');
printMatrix8x8_(double(pix(:,:,1)), '%4d');
fprintf('  Green channel (8x8):\n');
printMatrix8x8_(double(pix(:,:,2)), '%4d');
fprintf('  Blue channel (8x8):\n');
printMatrix8x8_(double(pix(:,:,3)), '%4d');
fprintf('\n');

% ---- Layer 4 ------------------------------------------------------------
fprintf('─── Layer 4 — Binary Encoding + Raw Memory Bytes ───────────────────\n');
px = info.Layer4.PixelRGB;
fprintf('  Pixel [1,1] RGB : [%d, %d, %d]\n', px(1), px(2), px(3));
fprintf('  R = %-3d  ->  %s\n', px(1), info.Layer4.BinaryR);
fprintf('  G = %-3d  ->  %s\n', px(2), info.Layer4.BinaryG);
fprintf('  B = %-3d  ->  %s\n', px(3), info.Layer4.BinaryB);
fprintf('  Raw memory bytes (hex, first 32 of 8x8 block):\n');
fprintf('  %s\n\n', info.Layer4.RawHexBytes);
fprintf('════════════════════════════════════════════════════════════════════\n\n');
end


function printMatrix8x8_(M, fmt)
%PRINTMATRIX8X8_  Print an 8x8 matrix with row prefixes.
for r = 1:size(M, 1)
    fprintf('  [');
    fprintf(fmt, M(r, :));
    fprintf(' ]\n');
end
end
