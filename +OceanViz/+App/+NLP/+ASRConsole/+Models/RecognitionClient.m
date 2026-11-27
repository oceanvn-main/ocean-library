classdef RecognitionClient < OceanViz.App.NLP.ASRConsole.Models.IRecognitionClient
    % RECOGNITIONCLIENT  IRecognitionClient over HTTP/JSON (webread / webwrite).
    %
    %   c = OceanViz.App.NLP.ASRConsole.Models.RecognitionClient('BaseUrl', "http://127.0.0.1:8009")
    %   The service is the Python app in Production SI/ASR (`asr serve` or `just up a10`);
    %   its interactive API documentation is at <BaseUrl>/docs.
    %   Any HTTP or connection failure is rethrown as
    %   OceanViz:App:NLP:ASRConsole:ServiceError with the method, path and cause.

    properties
        Timeout (1,1) double {mustBePositive} = 120   % s; the first request loads the model
    end

    methods
        function obj = RecognitionClient(opts)
            arguments
                opts.BaseUrl (1,1) string = "http://127.0.0.1:8009"
                opts.Timeout (1,1) double {mustBePositive} = 120
            end
            obj.BaseUrl = opts.BaseUrl;
            obj.Timeout = opts.Timeout;
        end

        function h = health(obj)
            h = obj.get("/v1/health", 5);
            h.models_loaded = reshape(string(h.models_loaded), 1, []);
        end

        function hw = hardware(obj)
            hw = obj.get("/v1/hardware", 30);
        end

        function m = models(obj)
            m = obj.normaliseModels(obj.get("/v1/models", 30));
        end

        function r = recognize(obj, x, fs, choices, threads)
            arguments
                obj
                x {mustBeNumeric}
                fs (1,1) double {mustBePositive}
                choices struct
                threads = []
            end
            body = struct('sample_rate', fs, 'encoding', 'pcm16', ...
                'payload', obj.encodePcm16(x), ...
                'models', {num2cell(choices(:)')});       % a cell -> JSON array even for one model
            if ~isempty(threads), body.threads = threads; end
            r = obj.post("/v1/recognize", body, obj.Timeout);
            r.results = obj.normaliseResults(r.results);
        end
    end

    methods (Access = private)
        function out = get(obj, path, timeout)
            try
                out = webread(obj.BaseUrl + path, weboptions('Timeout', timeout, 'ContentType', 'json'));
            catch err
                obj.fail("GET", path, err);
            end
        end

        function out = post(obj, path, body, timeout)
            opts = weboptions('MediaType', 'application/json', 'ContentType', 'json', 'Timeout', timeout);
            try
                out = webwrite(obj.BaseUrl + path, body, opts);
            catch err
                obj.fail("POST", path, err);
            end
        end

        function fail(obj, method, path, err)
            e = MException('OceanViz:App:NLP:ASRConsole:ServiceError', '%s %s%s failed: %s', ...
                method, obj.BaseUrl, path, err.message);
            e = e.addCause(err);
            throw(e);
        end
    end
end
