classdef BoundaryPadder < OceanMath.Image.Pixels.IPixelTransform
    % BOUNDARYPADDER  Pads image boundaries to handle border effects.
    %
    %   Pads an input [H x W x C] image array to size [(H + 2*Pr) x (W + 2*Pc) x C]
    %   using one of the following methods:
    %
    %     'zero'      - Pads with 0 (constant 0).
    %     'constant'  - Pads with a custom constant value (scalar or RGB vector).
    %     'replicate' - (or 'clamp') Repeats the border pixels indefinitely.
    %     'symmetric' - Symmetrical reflection, repeating the edge pixel.
    %     'circular'  - (or 'wrap') Periodic toroidal tiling.
    %     'reflect'   - (or 'mirror') Symmetrical reflection, without repeating the edge.
    %     'extend'    - Extrapolates signal by subtracting mirrored version from edge.
    %
    %   Constructor:
    %       BoundaryPadder('PadSize', [Pr, Pc], 'Boundary', border, 'ConstantValue', val)
    %
    %   Static usage:
    %       I_pad = BoundaryPadder.pad(I_in, pad_size, boundary, constant_val)
    %
    %   Author: Antigravity
    %   Date: 2026-06-12
    
    properties (Access = public)
        PadSize       (1,2) double {mustBeInteger, mustBeNonnegative} = [1, 1]
        Boundary            = 'replicate'
        ConstantValue       = 0.0
    end
    
    methods
        function obj = BoundaryPadder(options)
            % BOUNDARYPADDER Construct a BoundaryPadder transform.
            arguments
                options.PadSize       (1,2) double {mustBeInteger, mustBeNonnegative} = [1, 1]
                options.Boundary            = 'replicate'
                options.ConstantValue       = 0.0
            end
            obj.PadSize = options.PadSize;
            obj.Boundary = options.Boundary;
            obj.ConstantValue = options.ConstantValue;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Applies padding to the input image.
            I_out = BoundaryPadder.pad(I_in, obj.PadSize, obj.Boundary, obj.ConstantValue);
        end
    end
    
    methods (Static)
        function I_pad = pad(I_in, pad_size, boundary, constant_val)
            % PAD Static method to perform boundary padding.
            arguments
                I_in         {mustBeNumeric}
                pad_size     (1,2) double {mustBeInteger, mustBeNonnegative}
                boundary     = 'replicate'
                constant_val = 0.0
            end
            
            % Normalize pad sizes
            pad_rows = pad_size(1);
            pad_cols = pad_size(2);
            
            if pad_rows == 0 && pad_cols == 0
                I_pad = I_in;
                return;
            end
            
            % Normalize boundary mode to lowercase char
            if isstring(boundary)
                boundary = char(boundary);
            end
            
            [H, W, C] = size(I_in);
            
            % Handle constant/zero scalar or vector padding
            if (isnumeric(boundary) && isscalar(boundary)) || ...
               strcmpi(boundary, 'constant') || strcmpi(boundary, 'zero')
                
                if strcmpi(boundary, 'zero')
                    val = 0.0;
                elseif strcmpi(boundary, 'constant')
                    val = constant_val;
                else
                    val = boundary;
                end
                
                % If val is a vector, pad channel-by-channel
                if ~isscalar(val) && length(val) == C
                    I_pad = zeros(H + 2*pad_rows, W + 2*pad_cols, C, 'like', I_in); % (H + 2*pad_rows x W + 2 *pad_cols). 'like' inherit type from I_in
                    for ch = 1:C
                        I_pad(:,:,ch) = val(ch);
                    end
                else %All channel get the same
                    I_pad = ones(H + 2*pad_rows, W + 2*pad_cols, C, 'like', I_in) * val(1);
                end
                I_pad(pad_rows + (1:H), pad_cols + (1:W), :) = I_in;
                return;
            end
            
            % Resolve alias modes
            mode = lower(boundary);
            if strcmp(mode, 'clamp')
                mode = 'replicate';
            elseif strcmp(mode, 'wrap')
                mode = 'circular';
            elseif strcmp(mode, 'mirror')
                mode = 'reflect';
            end
            
            % Handle Extend Padding (Linear Extrapolation)
            if strcmp(mode, 'extend')
                H_pad = H + 2*pad_rows;
                W_pad = W + 2*pad_cols;
                
                % Setup 1D row coordinates relative to the input bounds
                r_in = (1 : H_pad) - pad_rows;
                r_ref = min(max(r_in, 1), H);
                
                r_mir = r_in;
                r_mir(r_in < 1) = 2 - r_in(r_in < 1);
                r_mir(r_in > H) = 2*H - r_in(r_in > H);
                
                % Setup 1D col coordinates relative to the input bounds
                c_in = (1 : W_pad) - pad_cols;
                c_ref = min(max(c_in, 1), W);
                
                c_mir = c_in;
                c_mir(c_in < 1) = 2 - c_in(c_in < 1);
                c_mir(c_in > W) = 2*W - c_in(c_in > W);
                
                % Generate 2D coordinate grids
                [R_ref, C_ref] = ndgrid(r_ref, c_ref);
                [R_mir, C_mir] = ndgrid(r_mir, c_mir);
                
                % Convert to linear indices and vectorize across channels
                channel_offsets = reshape((0 : C - 1) * (H * W), 1, 1, C);
                idx_ref_3d = (R_ref + (C_ref - 1)*H) + channel_offsets;
                idx_mir_3d = (R_mir + (C_mir - 1)*H) + channel_offsets;
                
                % Compute extrapolation: 2 * Edge - Mirrored
                I_pad = 2 * double(I_in(idx_ref_3d)) - double(I_in(idx_mir_3d));
                
                % Restore original class type
                if isa(I_in, 'uint8')
                    I_pad = uint8(I_pad);
                elseif isa(I_in, 'single')
                    I_pad = single(I_pad);
                end
                return;
            end
            
            % Generate coordinate mapping indices for standard modes
            switch mode
                case 'replicate'
                    r_idx = getReplicateIndices(H, pad_rows);
                    c_idx = getReplicateIndices(W, pad_cols);
                case 'symmetric'
                    r_idx = getSymmetricIndices(H, pad_rows);
                    c_idx = getSymmetricIndices(W, pad_cols);
                case 'circular'
                    r_idx = getCircularIndices(H, pad_rows);
                    c_idx = getCircularIndices(W, pad_cols);
                case 'reflect'
                    r_idx = getReflectIndices(H, pad_rows);
                    c_idx = getReflectIndices(W, pad_cols);
                otherwise
                    % Default fallback: zero padding
                    I_pad = zeros(H + 2*pad_rows, W + 2*pad_cols, C, 'like', I_in);
                    I_pad(pad_rows + (1:H), pad_cols + (1:W), :) = I_in;
                    return;
            end
            
            I_pad = I_in(r_idx, c_idx, :);
        end
    end
end

% --- Helper Functions ---

% GETREPLICATEINDICES computes indices for 'clamp'/'replicate' padding.
% Values outside boundaries are mapped to the edge elements themselves.
% Example (len=4, pad=2):
%   Original: [1, 2, 3, 4]
%   Padded:   [1, 1,  1, 2, 3, 4,  4, 4]
function idx = getReplicateIndices(len, pad)
    idx = zeros(1, len + 2*pad);
    for i = 1:(len + 2*pad)
        k = i - pad;
        if k < 1
            idx(i) = 1;      % Clamp to left edge
        elseif k > len
            idx(i) = len;    % Clamp to right edge
        else
            idx(i) = k;      % Keep original index
        end
    end
end

% GETSYMMETRICINDICES computes indices for 'symmetric' padding.
% Reflections mirror the signal across the outer boundary of the border pixel,
% repeating the edge pixel itself (e.g. 1 is followed by 1, len by len).
% Example (len=4, pad=2):
%   Original: [1, 2, 3, 4]
%   Padded:   [2, 1,  1, 2, 3, 4,  4, 3]
function idx = getSymmetricIndices(len, pad)
    idx = zeros(1, len + 2*pad);
    for i = 1:(len + 2*pad)
        k = i - pad;
        val = k;
        % Dynamically bounce coordinates back and forth within [1, len]
        % to support large padding sizes where pad > len
        while true
            if val < 1
                val = 1 - val;          % Reflect left: 0 -> 1, -1 -> 2, etc.
            elseif val > len
                val = 2*len - val + 1;  % Reflect right: len+1 -> len, len+2 -> len-1, etc.
            else
                break;                  % Within bounds
            end
        end
        idx(i) = val;
    end
end

% GETCIRCULARINDICES computes indices for periodic 'wrap'/'circular' padding.
% The signal wraps around periodically from end-to-beginning and vice versa.
% Example (len=4, pad=2):
%   Original: [1, 2, 3, 4]
%   Padded:   [3, 4,  1, 2, 3, 4,  1, 2]
function idx = getCircularIndices(len, pad)
    idx = zeros(1, len + 2*pad);
    for i = 1:(len + 2*pad)
        k = i - pad;
        % Convert to 0-based indexing, apply modulo, and convert back to 1-based indexing
        idx(i) = mod(k - 1, len) + 1;
    end
end

% GETREFLECTINDICES computes indices for 'reflect'/'mirror' padding.
% Reflections mirror the signal around the border pixel itself,
% without repeating the edge pixel (e.g. 1 is followed by 2, len by len-1).
% Example (len=4, pad=2):
%   Original: [1, 2, 3, 4]
%   Padded:   [3, 2,  1, 2, 3, 4,  3, 2]
function idx = getReflectIndices(len, pad)
    idx = zeros(1, len + 2*pad);
    if len == 1
        idx(:) = 1;
        return;
    end
    for i = 1:(len + 2*pad)
        k = i - pad;
        val = k;
        % Dynamically bounce coordinates back and forth within [1, len]
        % without replicating the edge pixels
        while true
            if val < 1
                val = 2 - val;      % Reflect left: 0 -> 2, -1 -> 3, etc.
            elseif val > len
                val = 2*len - val;  % Reflect right: len+1 -> len-1, len+2 -> len-2, etc.
            else
                break;              % Within bounds
            end
        end
        idx(i) = val;
    end
end
