classdef RecordingStore < handle
    % RECORDINGSTORE  Saves a recording (16-bit WAV) and its transcripts (JSON) side by side.
    %
    %   s = OceanViz.App.NLP.ASRConsole.Models.RecordingStore(folder)
    %   wav = s.save(x, fs, results, meta)   results: struct array (IRecognitionClient);
    %                                        meta: struct of extra fields (source, device, ...)
    %   Files are named rec_yyyyMMdd_HHmmss[_n].wav/.json. Nothing is saved unless asked:
    %   recordings of the user's voice stay on this PC.

    properties (SetAccess = private)
        Folder (1,1) string
    end

    methods
        function obj = RecordingStore(folder)
            arguments
                folder (1,1) string
            end
            obj.Folder = folder;
        end

        function wav = save(obj, x, fs, results, meta)
            arguments
                obj
                x {mustBeNumeric, mustBeNonempty}
                fs (1,1) double {mustBePositive}
                results struct = struct([])
                meta struct = struct()
            end
            if ~isfolder(obj.Folder), mkdir(obj.Folder); end
            stem = "rec_" + string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
            base = fullfile(obj.Folder, stem);
            n = 1;
            while isfile(base + ".wav")
                n = n + 1;
                base = fullfile(obj.Folder, stem + "_" + n);
            end
            wav = base + ".wav";
            audiowrite(wav, max(-1, min(1, double(x(:)))), fs, 'BitsPerSample', 16);
            doc = meta;
            doc.wav = char(wav);
            doc.sample_rate = fs;
            doc.duration_s = numel(x) / fs;
            doc.saved_at = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
            doc.results = results;
            fid = fopen(base + ".json", 'w', 'n', 'UTF-8');
            cleaner = onCleanup(@() fclose(fid));
            fprintf(fid, '%s', jsonencode(doc, 'PrettyPrint', true));
        end
    end
end
