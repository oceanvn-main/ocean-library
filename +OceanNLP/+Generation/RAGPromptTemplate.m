classdef RAGPromptTemplate < handle
    % RAGPROMPTTEMPLATE  Custom prompt template renderer for RAG pipelines.
    %
    % PURPOSE:
    %   Renders prompt strings by substituting `{context}`, `{question}`, and
    %   optional `{instructions}` placeholder variables.
    %
    % PROPERTIES:
    %   Template - Custom template string containing placeholders
    %
    % EXAMPLES:
    %   tmpl = OceanNLP.Generation.RAGPromptTemplate("System: {instructions}\nContext: {context}\nQ: {question}");
    %   prompt = tmpl.render("Doc context", "User question", "Be concise");
    %
    % See also: OceanNLP.Generation.PromptBuilder, OceanNLP.Generation.IGenerator

    properties (SetAccess = private)
        Template string
    end

    methods
        function obj = RAGPromptTemplate(template)
            arguments
                template (1,1) string
            end
            obj.Template = template;
        end

        function prompt = render(obj, context, question, instructions)
            arguments
                obj
                context string
                question (1,1) string
                instructions (1,1) string = ""
            end
            joinedContext = join(context, "\n\n");
            prompt = replace(obj.Template, ...
                ["{context}", "{question}", "{instructions}"], ...
                [joinedContext, question, instructions]);
        end
    end
end
