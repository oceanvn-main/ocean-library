classdef (Abstract) IChunker < handle
    % ICHUNKER Abstract base class for text chunking strategies in OceanNLP.
    %
    %   All chunker implementations inherit from IChunker and provide a uniform
    %   interface for splitting input text into segments (chunks).

    methods (Abstract)
        % CHUNK Split input text into an array of string chunks.
        %
        %   chunks = chunk(obj, text) takes a string input and returns a 1xN
        %   string array of segmented text chunks according to the strategy.
        chunks = chunk(obj, text)
    end
end

