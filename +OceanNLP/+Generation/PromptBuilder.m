classdef PromptBuilder < handle
    % PROMPTBUILDER  Constructs prompts by injecting retrieved context chunks into templates.
    %
    % PURPOSE:
    %   Assembles RAG prompts by combining context text chunks and a user question,
    %   enforcing maximum chunk count and total character length limits.
    %
    % PROPERTIES:
    %   Template      - String template containing {context} and {question} placeholders
    %   MaxChunks     - Maximum number of context chunks to include (default: 8)
    %   MaxCharacters - Maximum character budget for the context string (default: 12000)
    %
    % EXAMPLES:
    %   builder = OceanNLP.Generation.PromptBuilder('MaxChunks', 5, 'MaxCharacters', 4000);
    %   prompt = builder.build(["Chunk 1 text", "Chunk 2 text"], "What is RAG?");
    %
    % See also: OceanNLP.Generation.RAGPromptTemplate, OceanNLP.Generation.IGenerator

    properties (SetAccess = private)
        Template string
        MaxChunks (1,1) double
        MaxCharacters (1,1) double
    end

    methods
        function obj = PromptBuilder(options)
            arguments
                options.Template string = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                options.MaxChunks (1,1) double {mustBeInteger, mustBePositive} = 8
                options.MaxCharacters (1,1) double {mustBeInteger, mustBePositive} = 12000
            end
            obj.Template = options.Template;
            obj.MaxChunks = options.MaxChunks;
            obj.MaxCharacters = options.MaxCharacters;
        end

        function prompt = build(obj, chunks, question)
            arguments
                obj
                chunks string
                question string
            end
            selectedChunks = chunks(1:min(obj.MaxChunks, numel(chunks)));
            context = join(selectedChunks, "\n\n---\n\n");
            if strlength(context) > obj.MaxCharacters
                context = extractBefore(context, obj.MaxCharacters + 1);
            end
            prompt = replace(obj.Template, ["{context}", "{question}"], [context, question]);
        end
    end
end
