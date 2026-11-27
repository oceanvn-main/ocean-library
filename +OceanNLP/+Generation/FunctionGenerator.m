classdef FunctionGenerator < OceanNLP.Generation.IGenerator
    % FUNCTIONGENERATOR  Adapter wrapping any custom function handle as an IGenerator.
    %
    % PURPOSE:
    %   Adapts a function handle `@(prompt) ...` into a pluggable OceanNLP generator.
    %   Useful for mock LLMs in unit tests or local custom generation models.
    %
    % PROPERTIES:
    %   GeneratorFunction - Underlying function handle
    %
    % EXAMPLES:
    %   gen = OceanNLP.Generation.FunctionGenerator(@(p) "Mock answer for: " + p);
    %   answer = gen.generate("What is MATLAB?");
    %
    % See also: OceanNLP.Generation.IGenerator, OceanNLP.Generation.OpenAIGenerator

    properties (SetAccess = private)
        GeneratorFunction function_handle
    end

    methods
        function obj = FunctionGenerator(fn)
            arguments
                fn (1,1) function_handle
            end
            obj.GeneratorFunction = fn;
        end

        function answer = generate(obj, prompt)
            arguments
                obj
                prompt string
            end
            answer = string(obj.GeneratorFunction(prompt));
        end
    end
end
