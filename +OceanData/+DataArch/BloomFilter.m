classdef BloomFilter < handle
    % BLOOMFILTER  Probabilistic membership test for row-level deduplication.
    %
    %   A Bloom filter answers "has this item been inserted?" in O(1) with:
    %     - ZERO false negatives  (if it says "new", it IS new)
    %     - Small false positive rate  (configurable, e.g. 0.1%)
    %     - No disk I/O for lookups  (pure in-memory bit array)
    %
    %   Auto-sizes the bit array (m) and hash count (k) from the expected
    %   number of items (n) and desired false positive rate (p):
    %     m = -(n * ln(p)) / (ln(2))^2
    %     k = (m / n) * ln(2)
    %
    %   Uses SHA-256 with salt prefixes for k independent hash functions.
    %
    %   USAGE:
    %     bf = OceanData.DataArch.BloomFilter(10000, 0.001);
    %     bf.add('experiment_LDA_seed42');
    %     bf.mightContain('experiment_LDA_seed42')   % true
    %     bf.mightContain('experiment_LDA_seed43')   % false (almost certainly)
    %
    %     % Struct convenience:
    %     row = struct('model','LDA','error',0.12,'seed',42);
    %     bf.addStruct(row);
    %     bf.mightContainStruct(row)                 % true
    %
    %   See also: OceanData.DataArch.DataWarehouse

    % ================================================================== %
    properties (SetAccess = private)
        BitArray    logical         % 1 x m bit vector
        NumHashes   uint32          % k — number of hash functions
        NumBits     uint32          % m — size of bit array
        Count       uint32 = 0     % number of items inserted
        FPRate      double          % target false positive rate
    end

    % ================================================================== %
    methods

        function obj = BloomFilter(expectedItems, fpRate)
            % BLOOMFILTER  Create a Bloom filter sized for n items at FP rate p.
            %
            %   bf = BloomFilter(10000, 0.001)  — expects 10k items, 0.1% FP
            arguments
                expectedItems (1,1) double {mustBePositive}
                fpRate        (1,1) double {mustBePositive, mustBeLessThan(fpRate, 1)} = 0.01
            end

            obj.FPRate = fpRate;

            % Optimal bit array size: m = -(n * ln(p)) / (ln(2))^2
            m = ceil(-(expectedItems * log(fpRate)) / (log(2)^2));
            m = max(m, 64);   % minimum 64 bits
            obj.NumBits = uint32(m);

            % Optimal hash count: k = (m / n) * ln(2)
            k = round((double(m) / expectedItems) * log(2));
            k = max(k, 1);    % at least 1 hash
            obj.NumHashes = uint32(k);

            obj.BitArray = false(1, m);
        end

        % -------------------------------------------------------------- %
        function add(obj, key)
            % ADD  Insert a string key into the filter.
            %
            %   bf.add('my_unique_key')
            arguments
                obj
                key (1,1) string
            end
            indices = obj.hashKey_(key);
            obj.BitArray(indices) = true;
            obj.Count = obj.Count + 1;
        end

        % -------------------------------------------------------------- %
        function tf = mightContain(obj, key)
            % MIGHTCONTAIN  Test if a key was previously added.
            %
            %   Returns TRUE  if the key was (probably) added.
            %   Returns FALSE if the key was definitely NOT added.
            arguments
                obj
                key (1,1) string
            end
            indices = obj.hashKey_(key);
            tf = all(obj.BitArray(indices));
        end

        % -------------------------------------------------------------- %
        function addStruct(obj, s)
            % ADDSTRUCT  Serialize a struct to a canonical string and add it.
            %
            %   bf.addStruct(struct('model','LDA','seed',42))
            arguments
                obj
                s (1,1) struct
            end
            obj.add(obj.structToKey_(s));
        end

        % -------------------------------------------------------------- %
        function tf = mightContainStruct(obj, s)
            % MIGHTCONTAINSTRUCT  Test membership for a struct row.
            arguments
                obj
                s (1,1) struct
            end
            tf = obj.mightContain(obj.structToKey_(s));
        end

        % -------------------------------------------------------------- %
        function reset(obj)
            % RESET  Clear all bits and reset count.
            obj.BitArray(:) = false;
            obj.Count = uint32(0);
        end

        % -------------------------------------------------------------- %
        function saveToFile(obj, filepath)
            % SAVETOFILE  Persist the Bloom filter state to a .mat file.
            %
            %   bf.saveToFile('/path/to/bloom.mat')
            arguments
                obj
                filepath (1,1) string
            end
            bloomState.BitArray  = obj.BitArray;  %#ok<STRNU>
            bloomState.NumHashes = obj.NumHashes;
            bloomState.NumBits   = obj.NumBits;
            bloomState.Count     = obj.Count;
            bloomState.FPRate    = obj.FPRate;
            save(filepath, 'bloomState');
        end

        % -------------------------------------------------------------- %
        function info = describe(obj)
            % DESCRIBE  Return a summary struct with filter statistics.
            info.NumBits     = obj.NumBits;
            info.NumHashes   = obj.NumHashes;
            info.Count       = obj.Count;
            info.FPRate      = obj.FPRate;
            info.FillRatio   = sum(obj.BitArray) / double(obj.NumBits);
            % Estimated actual FP rate: (fillRatio)^k
            info.EstimatedFP = info.FillRatio ^ double(obj.NumHashes);
            info.MemoryBytes = numel(obj.BitArray);  % 1 byte per logical in MATLAB
        end

    end

    % ================================================================== %
    methods (Static)

        function obj = loadFromFile(filepath)
            % LOADFROMFILE  Restore a Bloom filter from a .mat file.
            %
            %   bf = OceanData.DataArch.BloomFilter.loadFromFile('bloom.mat')
            arguments
                filepath (1,1) string
            end
            if ~isfile(filepath)
                error('OceanData:DataArch:BloomFilter:FileNotFound', ...
                    'Bloom filter file not found: %s', filepath);
            end
            loaded = load(filepath, 'bloomState');
            st = loaded.bloomState;

            % Reconstruct — create with dummy args, then overwrite
            obj = OceanData.DataArch.BloomFilter(1, st.FPRate);
            obj.BitArray  = st.BitArray;
            obj.NumHashes = st.NumHashes;
            obj.NumBits   = st.NumBits;
            obj.Count     = st.Count;
            obj.FPRate    = st.FPRate;
        end

    end

    % ================================================================== %
    methods (Access = private)

        function indices = hashKey_(obj, key)
            % Compute k hash positions in [1, m] using SHA-256 + salt prefixes.
            md = java.security.MessageDigest.getInstance('SHA-256');
            k = double(obj.NumHashes);
            m = double(obj.NumBits);
            indices = zeros(1, k);

            keyBytes = uint8(char(key));
            for i = 1:k
                md.reset();
                % Salt: prepend the hash index as bytes
                salt = uint8(sprintf('__salt_%d__', i));
                md.update([salt, keyBytes]);
                digest = typecast(md.digest(), 'uint8');

                % Use first 8 bytes as a uint64 hash value
                hashVal = uint64(0);
                for b = 1:8
                    hashVal = bitor(bitshift(hashVal, 8), uint64(digest(b)));
                end

                % Map to bit position [1, m]
                indices(i) = double(mod(hashVal, uint64(m))) + 1;
            end
        end

    end

    % ================================================================== %
    methods (Static, Access = private)

        function key = structToKey_(s)
            % Serialize a struct to a canonical, deterministic string.
            % Sort field names for order-independence.
            fnames = sort(fieldnames(s));
            parts  = cell(1, numel(fnames));
            for i = 1:numel(fnames)
                fn  = fnames{i};
                val = s.(fn);
                if isnumeric(val) || islogical(val)
                    parts{i} = sprintf('%s=%.17g', fn, double(val));
                elseif ischar(val) || isstring(val)
                    parts{i} = sprintf('%s=%s', fn, char(val));
                else
                    % Fallback: use mat2str for arrays, class name for objects
                    try
                        parts{i} = sprintf('%s=%s', fn, mat2str(val));
                    catch
                        parts{i} = sprintf('%s=<%s>', fn, class(val));
                    end
                end
            end
            key = string(strjoin(parts, '|'));
        end

    end

end
