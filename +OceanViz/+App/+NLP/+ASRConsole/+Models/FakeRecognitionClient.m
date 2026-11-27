classdef FakeRecognitionClient < OceanViz.App.NLP.ASRConsole.Models.IRecognitionClient
    % FAKERECOGNITIONCLIENT  In-memory IRecognitionClient for ViewModel tests (no service).
    %
    %   c = OceanViz.App.NLP.ASRConsole.Models.FakeRecognitionClient()
    %   c.Online = false            every call errors as the real client does when the service is down
    %   c.Texts.champion = "HELLO"  transcript returned for a model (default "<model> TEXT")
    %   c.FailModels = "citrinet"   these models answer with an error in their row
    %   c.Calls                     log of requests: struct array (method, fs, samples, choices, threads)

    properties
        Online (1,1) logical = true
        Texts struct = struct()
        FailModels (1,:) string = strings(1, 0)
        ModelList struct
        Calls struct = struct('method', {}, 'fs', {}, 'samples', {}, 'choices', {}, 'threads', {})
    end

    methods
        function obj = FakeRecognitionClient()
            raw = {struct('model', 'champion', 'title', 'Champion (registry alias)', ...
                    'backends', {{'pytorch'; 'onnx_fp32'; 'onnx_int8'}}, 'default_backend', 'pytorch', ...
                    'model_version', '2', 'model_name', 'hubert_base_ctc_a3', 'available', true, 'reason', ''), ...
                struct('model', 'reference', 'title', 'Reference (registry alias)', ...
                    'backends', {{'pytorch'}}, 'default_backend', 'pytorch', ...
                    'model_version', '1', 'model_name', 'wav2vec2_base_960h', 'available', true, 'reason', ''), ...
                struct('model', 'citrinet', 'title', 'NVIDIA Citrinet-256 (NeMo)', ...
                    'backends', {{'nemo'}}, 'default_backend', 'nemo', 'model_version', '', ...
                    'model_name', '', 'available', false, 'reason', 'remote service not reachable')};
            obj.ModelList = obj.normaliseModels(raw);
        end

        function h = health(obj)
            obj.check("health");
            h = struct('ready', true, 'service_version', "fake", 'models_loaded', strings(1, 0));
        end

        function hw = hardware(obj)
            obj.check("hardware");
            hw = struct('hostname', 'fake-host', 'platform', 'Fake-OS', 'cpu_logical', 8, 'ram_gb', 16, ...
                'torch_version', '2.14.1+cpu', 'gpus', struct('name', {'Fake GPU'}, ...
                'torch_has_kernels', {false}, 'kernel_note', {'fake: no kernels'}), ...
                'onnxruntime_version', '1.31.0', 'onnxruntime_providers', {{'CPUExecutionProvider'}});
        end

        function m = models(obj)
            obj.check("models");
            m = obj.ModelList;
        end

        function r = recognize(obj, x, fs, choices, threads)
            obj.check("recognize");
            obj.Calls(end + 1) = struct('method', "recognize", 'fs', fs, 'samples', numel(x), ...
                'choices', choices, 'threads', threads);
            raw = cell(1, numel(choices));
            dur = numel(x) / fs;
            for k = 1:numel(choices)
                m = string(choices(k).model);
                row = struct('model', m, 'backend', string(choices(k).backend), 'model_version', "", ...
                    'model_name', "", 'text', "", 'latency_ms', 10 * k, 'rtf', 0.01 * k / dur, 'error', "");
                if any(obj.FailModels == m)
                    row.text = ""; row.latency_ms = []; row.rtf = []; row.error = "fake failure";
                elseif isfield(obj.Texts, m)
                    row.text = string(obj.Texts.(m));
                else
                    row.text = upper(m) + " TEXT";
                end
                raw{k} = row;
            end
            r = struct('duration_s', dur, 'sample_rate_in', fs, ...
                'peak_dbfs', 20 * log10(max(max(abs(x(:))), 1e-6)), 'results', obj.normaliseResults(raw));
        end
    end

    methods (Access = private)
        function check(obj, method)
            if ~obj.Online
                error('OceanViz:App:NLP:ASRConsole:ServiceError', '%s %s failed: fake service offline', ...
                    upper(method), obj.BaseUrl);
            end
        end
    end
end
