classdef NpyReader
    % NPYREADER  Read NumPy .npy files (format versions 1.0, 2.0, 3.0) into
    %   MATLAB with the SAME INDEX MEANING as in Python:
    %       A(i1, ..., ik) == np_array[i1-1, ..., ik-1]
    %
    %   .npy layout: magic "\x93NUMPY", version (2 bytes), header length
    %   (2 bytes for v1, 4 bytes for v2/v3, little-endian), then an ASCII
    %   Python dict such as
    %       {'descr': '<f4', 'fortran_order': False, 'shape': (64, 3, 16), }
    %   followed by the raw element buffer.
    %
    %   MEMORY ORDER: with fortran_order False (NumPy's default, C order) the
    %   LAST index varies fastest, the opposite of MATLAB. The buffer is read
    %   with the reversed shape and the axes are reversed back (see
    %   OceanDL.Layout.DimFormat.fromRowMajor for the same rule).
    %   With fortran_order True the buffer is already column-major.
    %
    %   The result has the SOURCE axis meaning (e.g. PyTorch batch-first);
    %   converting to the repo's canonical layout is a separate, explicit
    %   step: DimFormat.toCanonical(A, "BTC").
    %
    % USAGE:
    %   [A, info] = OceanData.Adapters.NpyReader.read('batch.npy');   % info: Descr, FortranOrder, Shape

    methods (Static)
        function [A, info] = read(path)
            arguments
                path (1,1) string
            end
            fid = fopen(path, 'r', 'ieee-le');
            if fid < 0
                error('OceanData:Adapters:NpyReader:Open', 'Cannot open %s.', path);
            end
            cleaner = onCleanup(@() fclose(fid));
            magic = fread(fid, 6, '*uint8').';
            if ~isequal(magic, [147 double('NUMPY')])
                error('OceanData:Adapters:NpyReader:Magic', '%s is not a .npy file.', path);
            end
            version = fread(fid, 2, '*uint8').';
            if version(1) == 1
                headerLen = fread(fid, 1, 'uint16');
            else
                headerLen = fread(fid, 1, 'uint32');
            end
            header = char(fread(fid, headerLen, '*uint8').');
            info = OceanData.Adapters.NpyReader.parseHeader(header);
            [precision, bytes, byteOrder] = OceanData.Adapters.NpyReader.dtype(info.Descr);
            n = prod(info.Shape);
            data = fread(fid, n, ['*' precision], 0, byteOrder);
            if numel(data) ~= n
                error('OceanData:Adapters:NpyReader:Truncated', ...
                    'Expected %d elements (%d bytes each), read %d.', n, bytes, numel(data));
            end
            if strcmp(precision, 'uint8') && startsWith(info.Descr, '|b1')
                data = logical(data);
            end
            shape = info.Shape;
            if isempty(shape)                              % 0-d array (scalar)
                A = data;
            elseif info.FortranOrder
                A = reshape(data, [shape, 1]);
            else
                k = numel(shape);
                if k == 1
                    A = data(:);
                else
                    A = permute(reshape(data, fliplr(shape)), k:-1:1);
                end
            end
        end

        function info = parseHeader(header)
            descr = regexp(header, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
            fortran = regexp(header, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
            shape = regexp(header, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
            if isempty(descr) || isempty(fortran) || isempty(shape)
                error('OceanData:Adapters:NpyReader:Header', 'Unparseable .npy header: %s', header);
            end
            dims = str2double(strsplit(strtrim(shape{1}), ','));
            dims = dims(~isnan(dims));
            info = struct('Descr', descr{1}, 'FortranOrder', strcmp(fortran{1}, 'True'), ...
                'Shape', reshape(dims, 1, []));
        end

        function [precision, bytes, byteOrder] = dtype(descr)
            % Supported NumPy dtype strings -> fread precision.
            byteOrder = 'ieee-le';
            if descr(1) == '>'
                byteOrder = 'ieee-be';
            end
            code = descr(2:end);
            table = {'f4', 'single', 4; 'f8', 'double', 8; 'i1', 'int8', 1; 'u1', 'uint8', 1; ...
                     'i2', 'int16', 2; 'u2', 'uint16', 2; 'i4', 'int32', 4; 'u4', 'uint32', 4; ...
                     'i8', 'int64', 8; 'u8', 'uint64', 8; 'b1', 'uint8', 1};
            row = find(strcmp(table(:, 1), code), 1);
            if isempty(row)
                error('OceanData:Adapters:NpyReader:DType', 'Unsupported dtype %s.', descr);
            end
            precision = table{row, 2};
            bytes = table{row, 3};
        end
    end
end
