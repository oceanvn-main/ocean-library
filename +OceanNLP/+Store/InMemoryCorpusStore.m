classdef InMemoryCorpusStore < OceanNLP.Store.ICorpusStore
    % INMEMORYCORPUSSTORE Flat in-memory text chunk and metadata store.
    %
    %   Stores text chunks and their source document IDs in memory. Provides
    %   fast lookup by chunk index following vector similarity search.

    properties (SetAccess=private)
        Chunks    string = strings(1,0) % Array of text chunk strings
        SourceIds double = zeros(1,0)   % Mapping of chunk index to source document ID
    end

    methods
        function fit(obj, chunks, sourceIds)
            % FIT Load chunks and source document IDs into memory.
            arguments
                obj
                chunks    string
                sourceIds double
            end
            if numel(chunks) ~= numel(sourceIds)
                error('OceanNLP:Store:SizeMismatch', 'Chunks and source IDs must align in length.');
            end
            obj.Chunks    = reshape(chunks, 1, []); %Force 1 x auto
            obj.SourceIds = reshape(sourceIds, 1, []);
        end

        function text = getChunk(obj, idx)
            % GETCHUNK Retrieve chunk text by index.
            arguments
                obj
                idx double
            end
            text = obj.Chunks(idx);
        end

        function ids = getSource(obj, idx)
            % GETSOURCE Retrieve source document ID for given chunk index.
            arguments
                obj
                idx double
            end
            ids = obj.SourceIds(idx);
        end

        function n = numChunks(obj)
            % NUMCHUNKS Return total number of stored chunks.
            n = numel(obj.Chunks);
        end
    end
end

