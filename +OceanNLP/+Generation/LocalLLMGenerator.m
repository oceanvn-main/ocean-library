classdef LocalLLMGenerator < OceanNLP.Generation.IGenerator
    % LOCALLLMGENERATOR  Adapter for local LLMs (LM Studio, Ollama, vLLM, LocalAI).
    %
    % PURPOSE:
    %   Sends prompt requests to a local HTTP chat completion server running on
    %   localhost (e.g. LM Studio on port 1234, Ollama on port 11434).
    %
    % PROPERTIES:
    %   Endpoint    - Local server HTTP endpoint URL (default: "http://localhost:1234/v1/chat/completions")
    %   Model       - Local model identifier (default: "gemma-4-12b")
    %   Temperature - Sampling temperature parameter (default: 0)
    %   Timeout     - HTTP request timeout in seconds (default: 120)
    %
    % EXAMPLES:
    %   % Default LM Studio setup with Gemma 4 12B:
    %   gen = OceanNLP.Generation.LocalLLMGenerator('Model', "gemma-4-12b");
    %   answer = gen.generate("What is RAG in MATLAB?");
    %
    %   % Custom LM Studio / Ollama port configuration:
    %   gen = OceanNLP.Generation.LocalLLMGenerator(...
    %       'Endpoint', "http://localhost:11434/v1/chat/completions", ...
    %       'Model', "gemma-4-12b");
    %
    % See also: OceanNLP.Generation.IGenerator, OceanNLP.Generation.OpenAIGenerator

    properties (SetAccess = private)
        Endpoint string
        Model string
        Temperature double = 0
        Timeout double = 120
    end

    methods
        function obj = LocalLLMGenerator(options)
            arguments
                options.Endpoint (1,1) string = "http://localhost:1234/v1/chat/completions"
                options.Model (1,1) string = "gemma-4-12b"
                options.Temperature (1,1) double {mustBeNonnegative} = 0
                options.Timeout (1,1) double {mustBePositive} = 120
            end
            obj.Endpoint = options.Endpoint;
            obj.Model = options.Model;
            obj.Temperature = options.Temperature;
            obj.Timeout = options.Timeout;
        end

        function answer = generate(obj, prompt)
            arguments
                obj
                prompt (1,1) string
            end

            targetModel = obj.Model;
            % Dynamically query /v1/models to get exact loaded chat model name if available
            try
                modelsUrl = strrep(obj.Endpoint, "/chat/completions", "/models");
                optsRead = weboptions('Timeout', 5);
                resModels = webread(char(modelsUrl), optsRead);
                if isfield(resModels, 'data') && ~isempty(resModels.data)
                    for k = 1:numel(resModels.data)
                        candidate = string(resModels.data(k).id);
                        if ~contains(candidate, "embed")
                            targetModel = candidate;
                            break;
                        end
                    end
                end
            catch
                % Fall back to configured obj.Model if /v1/models is unreachable
            end

            body = struct( ...
                'model', targetModel, ...
                'temperature', obj.Temperature, ...
                'messages', {{struct('role', 'user', 'content', char(prompt))}} ...
            );

            opts = weboptions( ...
                'MediaType', 'application/json', ...
                'HeaderFields', {'Authorization', 'Bearer not-needed'}, ...
                'Timeout', obj.Timeout ...
            );

            try
                response = webwrite(char(obj.Endpoint), body, opts);
                answer = string(response.choices(1).message.content);
            catch ME
                error('OceanNLP:LocalLLMGenerator:ConnectionFailed', ...
                    'Failed to connect to local LLM server at %s. Ensure LM Studio / local server is running.\nDetails: %s', ...
                    obj.Endpoint, ME.message);
            end
        end
    end
end
