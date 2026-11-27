classdef StreamRegistry < handle
    % STREAMREGISTRY  One run seed -> one independent random stream per named purpose.
    % Every source of randomness in a run ("env" episode seeds, each agent's action draws, ...) gets its own
    % mrg32k3a substream, chosen by hashing the purpose name. Because the substream index depends only on the
    % purpose string (not on creation order), adding a new purpose never changes another purpose's draws (AP38),
    % and the order in which agents are evaluated cannot change what any agent draws (C03).
    %
    %   reg = OceanAgents.Runtime.StreamRegistry(42);
    %   rs  = reg.stream("agent:alice:action");    % same handle on every call with this purpose
    %   s   = reg.nextSeed("env");                 % integer in [0, 2^32-1] drawn from the "env" stream
    %
    % Substream index:  k = 1 + mod(FNV-1a-32(UTF-8 bytes of purpose), 2^31),  NumStreams = 2^31.
    % Two purposes with the same k -> OceanAgents:Runtime:StreamCollision (never silently shared).
    % The global stream (rng / RandStream.getGlobalStream) is never read or modified.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.3.

    properties (SetAccess = private)
        RunSeed double = 0
    end

    properties (Constant)
        Generator = "mrg32k3a"
        NumStreams = 2^31
    end

    properties (Access = private)
        Streams_       % containers.Map: purpose (char) -> RandStream
        IndexOwner_    % containers.Map: substream index (double) -> purpose (char)
    end

    methods
        function obj = StreamRegistry(runSeed)
            arguments
                runSeed (1,1) double {mustBeInteger, mustBeNonnegative, mustBeLessThan(runSeed, 4294967296)}
            end
            obj.RunSeed = runSeed;
            obj.Streams_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.IndexOwner_ = containers.Map('KeyType', 'double', 'ValueType', 'any');
        end

        function rs = stream(obj, purpose)
            % STREAM  The RandStream for purpose (created on first request, then the same handle).
            arguments
                obj
                purpose (1,1) string {mustBeNonzeroLengthText}
            end
            key = char(purpose);
            if isKey(obj.Streams_, key)
                rs = obj.Streams_(key);
                return
            end
            k = obj.indexOf(purpose);
            if isKey(obj.IndexOwner_, k)
                fail_('StreamCollision', 'Purposes "%s" and "%s" both hash to substream %d; rename one.', ...
                    obj.IndexOwner_(k), key, k);
            end
            rs = RandStream.create(char(obj.Generator), 'Seed', obj.RunSeed, ...
                'NumStreams', obj.NumStreams, 'StreamIndices', k);
            obj.Streams_(key) = rs;
            obj.IndexOwner_(k) = key;
        end

        function s = nextSeed(obj, purpose)
            % NEXTSEED  A fresh integer seed in [0, 2^32-1], drawn from stream(purpose) (e.g. one per episode).
            arguments
                obj
                purpose (1,1) string {mustBeNonzeroLengthText}
            end
            s = randi(obj.stream(purpose), [0, 2^32 - 1]);
        end

        function purposes = purposes(obj)
            % PURPOSES  Purposes created so far (sorted).
            purposes = string(keys(obj.Streams_));
        end

        function meta = getMetadata(obj)
            meta = pillarMeta_(obj, {sprintf('run seed %d; %s, NumStreams 2^31; index = 1 + mod(FNV-1a-32(purpose), 2^31)', ...
                obj.RunSeed, obj.Generator)}, "L'Ecuyer (1999) MRG32k3a substreams; FNV-1a hash (Fowler-Noll-Vo)");
        end

        function spec = getPipelineSpec(obj)
            spec = pillarSpec_(obj, strings(1, 0), strings(1, 0));
        end
    end

    methods (Static)
        function k = indexOf(purpose)
            % INDEXOF  Substream index of a purpose: 1 + mod(FNV-1a-32(UTF-8 bytes), 2^31).
            arguments
                purpose (1,1) string
            end
            k = 1 + mod(OceanAgents.Runtime.StreamRegistry.fnv1a32(purpose), 2^31);
        end

        function h = fnv1a32(text)
            % FNV1A32  32-bit FNV-1a hash of the UTF-8 bytes of text, returned as a double in [0, 2^32-1].
            %   h = offset basis;  for each byte b:  h = (h XOR b) * FNV prime  (mod 2^32)
            % uint64 keeps the product exact: h < 2^32 and prime < 2^25, so h*prime < 2^57.
            arguments
                text (1,1) string
            end
            offsetBasis = uint64(2166136261);
            fnvPrime = uint64(16777619);
            mask32 = uint64(4294967295);
            h = offsetBasis;
            bytes = uint64(unicode2native(char(text), 'UTF-8'));
            for b = bytes
                h = bitand(bitxor(h, b) * fnvPrime, mask32);
            end
            h = double(h);
        end
    end
end
