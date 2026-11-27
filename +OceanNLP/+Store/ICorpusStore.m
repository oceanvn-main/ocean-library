classdef (Abstract) ICorpusStore < handle
    % ICORPUSSTORE Abstract interface for document corpus storage in OceanNLP.
    %
    %   Defines the contract for storing, mapping, and retrieving text chunks
    %   and their underlying source document metadata for vector search and RAG.

    methods (Abstract)
        % FIT Populate store with text chunks and their source document IDs.
        fit(obj, chunks, sourceIds)

        % GETCHUNK Retrieve raw chunk text for the specified index/indices.
        text = getChunk(obj, idx)

        % GETSOURCE Retrieve source document ID for the specified chunk index.
        ids = getSource(obj, idx)
    end
end

