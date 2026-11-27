classdef OpenAIGenerator < OceanNLP.Generation.IGenerator
    % OPENAIGENERATOR  OpenAI-compatible chat completion API generator.
    %
    % PURPOSE:
    %   Sends prompt requests to an OpenAI-compatible HTTP chat completion endpoint
    %   (e.g., OpenAI API, Azure OpenAI, vLLM, Ollama, LM Studio) using `webwrite`.
    %
    % PROPERTIES:
    %   Endpoint    - Full URL endpoint for chat completions
    %   Model       - Model name (e.g. 'gpt-4o', 'gpt-3.5-turbo')
    %   ApiKey      - API authorization token (reads from OPENAI_API_KEY environment variable by default)
    %   Temperature - Sampling temperature parameter (default: 0)
    %
    % EXAMPLES:
    %   gen = OceanNLP.Generation.OpenAIGenerator('Model', "gpt-4o", 'ApiKey', "sk-...");
    %   answer = gen.generate("Summarize climate change.");
    %
    % See also: OceanNLP.Generation.IGenerator, OceanNLP.Generation.FunctionGenerator

    properties (SetAccess = private)
        Endpoint string
        Model string
        ApiKey string
        Temperature double = 0
    end

    methods
        function obj = OpenAIGenerator(options)
            arguments
                options.Endpoint (1,1) string = "https://api.openai.com/v1/chat/completions"
                options.Model (1,1) string
                options.ApiKey (1,1) string = string(getenv('OPENAI_API_KEY'))
                options.Temperature (1,1) double {mustBeNonnegative} = 0
            end
            if strlength(options.ApiKey) == 0
                error('OceanNLP:OpenAIGenerator:MissingApiKey', ...
                    'Provide ApiKey option or set OPENAI_API_KEY environment variable.');
            end
            obj.Endpoint = options.Endpoint;
            obj.Model = options.Model;
            obj.ApiKey = options.ApiKey;
            obj.Temperature = options.Temperature;
        end

        function answer = generate(obj, prompt)
            arguments
                obj
                prompt (1,1) string
            end
            body = struct( ...
                'model', obj.Model, ...
                'temperature', obj.Temperature, ...
                'messages', {{struct('role', 'user', 'content', char(prompt))}} ...
            );
            opts = weboptions( ...
                'MediaType', 'application/json', ...
                'HeaderFields', {'Authorization', ['Bearer ' char(obj.ApiKey)]}, ...
                'Timeout', 120 ...
            );
            response = webwrite(char(obj.Endpoint), body, opts);
            answer = string(response.choices(1).message.content);
        end
    end
end
