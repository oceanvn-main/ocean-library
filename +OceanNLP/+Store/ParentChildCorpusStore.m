classdef ParentChildCorpusStore < OceanNLP.Store.ICorpusStore
    % PARENTCHILDCORPUSSTORE Hierarchical Parent-Child retrieval corpus store.
    %
    %   Implements Parent-Child / Auto-Merging retrieval architecture.
    %   Smaller "child" chunks are indexed for high-precision vector search,
    %   while larger "parent" documents/paragraphs are returned for LLM context.

    properties (SetAccess=private)
        Children  string = strings(1,0) % Small child chunks (used for embedding/search)
        ParentIds double = zeros(1,0)   % ID mapping from each child to its parent document
        Parents   string = strings(1,0) % Larger parent context strings
    end

    methods
        function fit(obj, children, parentIds)
            % FIT Store child chunks and their corresponding parent IDs.
            arguments
                obj
                children  string
                parentIds double
            end
            if numel(children) ~= numel(parentIds)
                error('OceanNLP:Store:SizeMismatch', 'Children and parent IDs must align in length.');
            end
            obj.Children  = children;
            obj.ParentIds = parentIds;
            obj.Parents   = children; % Default: parent text equals child text if not specified
        end

        function fitHierarchy(obj, parents, children, parentIds)
            % FITHIERARCHY Store full parent-child hierarchy with parent context strings.
            arguments
                obj
                parents   string
                children  string
                parentIds double
            end
            obj.fit(children, parentIds);
            obj.Parents = parents;
        end

        function [texts, ids] = getParents(obj, childIds)
            % GETPARENTS Retrieve unique parent context texts for retrieved child chunk IDs.
            arguments
                obj
                childIds double
            end
            ids   = unique(obj.ParentIds(childIds), 'stable');
            texts = obj.Parents(ids);
        end

        function text = getChunk(obj, idx)
            % GETCHUNK Retrieve child chunk text by index.
            arguments
                obj
                idx double
            end
            text = obj.Children(idx);
        end

        function ids = getSource(obj, idx)
            % GETSOURCE Retrieve parent ID for a child chunk index.
            arguments
                obj
                idx double
            end
            ids = obj.ParentIds(idx);
        end

        function n = numChunks(obj)
            % NUMCHUNKS Return total count of child chunks.
            n = numel(obj.Children);
        end
    end
end

