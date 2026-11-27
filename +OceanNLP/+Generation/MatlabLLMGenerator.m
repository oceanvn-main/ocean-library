classdef MatlabLLMGenerator < OceanNLP.Generation.IGenerator
    % MATLABLLMGENERATOR  Adapter for MathWorks official or third-party MATLAB LLM clients.
    %
    % PURPOSE:
    %   Adapts installed MATLAB LLM integration functions into an IGenerator strategy.
    %
    % PROPERTIES:
    %   ClientFunction - Function handle invoking the underlying MATLAB LLM client
    %
    % See also: OceanNLP.Generation.IGenerator, OceanNLP.Generation.OpenAIGenerator

    properties (SetAccess = private)
        ClientFunction function_handle
    end

    methods
        function obj = MatlabLLMGenerator(clientFunction)
            arguments
                clientFunction (1,1) function_handle
            end
            obj.ClientFunction = clientFunction;
        end

        function answer = generate(obj, prompt)
            arguments
                obj
                prompt (1,1) string
            end
            answer = string(obj.ClientFunction(prompt));
        end
    end
end
