classdef (Abstract) IRecognitionClient < handle
    % IRECOGNITIONCLIENT  What the ASR Console needs from the local recognition service (A10).
    %
    %   The console never calls webread / webwrite directly: it talks to an IRecognitionClient
    %   so tests can use FakeRecognitionClient without the service.
    %
    %   Implement:
    %     h  = health(obj)     struct: ready, service_version, models_loaded (string)
    %     hw = hardware(obj)   struct: the service's `asr hw probe` facts + onnxruntime_providers
    %     m  = models(obj)     struct array (normalised, see normaliseModels): model, title,
    %                          backends (string row), default_backend, model_version,
    %                          model_name, available, reason
    %     r  = recognize(obj, x, fs, choices, threads)
    %                          x: S x 1 samples in [-1, 1]; fs Hz; choices: struct array with
    %                          fields model, backend; threads: [] or a count. r: struct with
    %                          duration_s, sample_rate_in, peak_dbfs and results (struct array:
    %                          model, backend, model_version, model_name, text, latency_ms,
    %                          rtf, error), normalised by normaliseResults.
    %   Implementations: RecognitionClient (HTTP/JSON), FakeRecognitionClient (tests).

    properties
        BaseUrl (1,1) string = "http://127.0.0.1:8009"
    end

    methods (Abstract)
        h = health(obj)
        hw = hardware(obj)
        m = models(obj)
        r = recognize(obj, x, fs, choices, threads)
    end

    methods (Static)
        function b64 = encodePcm16(x)
            % ENCODEPCM16  Samples in [-1, 1] -> base64 of little-endian int16 (the service's
            %   AudioChunk.ENCODING_PCM16). +1 clips to 32767 (PCM16 cannot represent +1).
            s = int16(max(-32768, min(32767, round(double(x(:)) * 32768))));
            bytes = typecast(s, 'uint8');
            [~, ~, endian] = computer;
            if endian == 'B'
                bytes = reshape(flipud(reshape(bytes, 2, [])), [], 1);
            end
            b64 = string(matlab.net.base64encode(bytes));
        end

        function m = normaliseModels(raw)
            % NORMALISEMODELS  jsondecode output (struct array or cell) -> uniform struct array.
            m = struct('model', {}, 'title', {}, 'backends', {}, 'default_backend', {}, ...
                'model_version', {}, 'model_name', {}, 'available', {}, 'reason', {});
            items = OceanViz.App.NLP.ASRConsole.Models.IRecognitionClient.asCell(raw);
            for k = 1:numel(items)
                s = items{k};
                m(k).model = string(s.model);
                m(k).title = string(s.title);
                m(k).backends = reshape(string(s.backends), 1, []);
                m(k).default_backend = string(s.default_backend);
                m(k).model_version = string(s.model_version);
                m(k).model_name = string(s.model_name);
                m(k).available = logical(s.available);
                m(k).reason = string(s.reason);
            end
        end

        function r = normaliseResults(raw)
            % NORMALISERESULTS  jsondecode results -> uniform struct array (null -> NaN / "").
            r = struct('model', {}, 'backend', {}, 'model_version', {}, 'model_name', {}, ...
                'text', {}, 'latency_ms', {}, 'rtf', {}, 'error', {});
            items = OceanViz.App.NLP.ASRConsole.Models.IRecognitionClient.asCell(raw);
            num = @(v) OceanViz.App.NLP.ASRConsole.Models.IRecognitionClient.numOrNaN(v);
            for k = 1:numel(items)
                s = items{k};
                r(k).model = string(s.model);
                r(k).backend = string(s.backend);
                r(k).model_version = string(s.model_version);
                r(k).model_name = string(s.model_name);
                r(k).text = string(s.text);
                r(k).latency_ms = num(s.latency_ms);
                r(k).rtf = num(s.rtf);
                r(k).error = string(s.error);
            end
        end
    end

    methods (Static, Access = protected)
        function c = asCell(raw)
            if isempty(raw)
                c = {};
            elseif iscell(raw)
                c = raw(:)';
            else
                c = num2cell(raw(:)');
            end
        end

        function v = numOrNaN(v)
            if isempty(v), v = NaN; else, v = double(v); end
        end
    end
end
