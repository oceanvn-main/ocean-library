classdef (Abstract) IGenerator < handle
    % IGENERATOR  Abstract strategy interface for text generation models / LLMs.
    %
    % PURPOSE:
    %   Provides a common contract for generating natural language answers from prompts.
    %
    % METHODS:
    %   answer = generate(obj, prompt) - Generate an answer string given a prompt.
    %
    % See also: OceanNLP.Generation.OpenAIGenerator,
    %           OceanNLP.Generation.FunctionGenerator,
    %           OceanNLP.Generation.MatlabLLMGenerator

    methods (Abstract)
        % GENERATE  Generate response text from a structured or unstructured prompt string.
        %   answer - String containing the generated text response.
        answer = generate(obj, prompt)
    end
end

