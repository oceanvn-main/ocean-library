classdef StageContext < handle
    % STAGECONTEXT  Everything one completed pipeline run hands to the stage plug-ins.
    %
    %   Built by Models.StudioSession.run from an Inference.PipelineResult; stages only
    %   READ it. Lazily computed extras (pitch / energy contours, edit alignment, LM score
    %   split) call library classes and are cached per segment.
    %
    %   ctx.Result           OceanNLP.Speech.Inference.PipelineResult (immutable)
    %   ctx.SegmentIndex     segment whose reader the stages show (1..NumSegments)
    %   ctx.reader()         SpeechProcessAdapter reader of that segment ([] if none)
    %   ctx.segmentWave()    Waveform the recogniser saw for that segment
    %   ctx.artifact(name)   value of the step called NAME ([] if absent/omitted)
    %   ctx.artifacts(pat)   struct array (Name, Value) of steps whose name matches the
    %                        regular expression PAT (case-insensitive), in run order
    %   ctx.has(cap)         model capability ("ctc", "attention", "ctc-rescore", "lm", "frontend")
    %   ctx.Viz              OceanViz.Core used for every plot

    properties (SetAccess = private)
        Result = []
        Recognizer = []
        ModelName (1,1) string = ""
        ModelRow table = table()
        Capabilities (1,:) string = strings(1, 0)
        LanguageModel = []
        Settings struct = struct()
        Reference (1,1) string = ""
        InputLabel (1,1) string = ""
        InputKind (1,1) string = ""
        DetectorInfo struct = struct()
        Detector = []
        Conditioner = []
        RunSeconds (1,1) double = NaN
        Notes (1,:) string = strings(1, 0)
    end

    properties
        SegmentIndex (1,1) double {mustBeInteger, mustBePositive} = 1
        Viz = []
    end

    properties (Access = private)
        Cache = struct()
    end

    methods
        function obj = StageContext(result, opts)
            arguments
                result
                opts.Recognizer = []
                opts.ModelName (1,1) string = ""
                opts.ModelRow table = table()
                opts.Capabilities (1,:) string = strings(1, 0)
                opts.LanguageModel = []
                opts.Settings struct = struct()
                opts.Reference (1,1) string = ""
                opts.InputLabel (1,1) string = ""
                opts.InputKind (1,1) string = ""
                opts.DetectorInfo struct = struct()
                opts.Detector = []
                opts.Conditioner = []
                opts.RunSeconds (1,1) double = NaN
                opts.Notes (1,:) string = strings(1, 0)
                opts.Viz = []
            end
            obj.Result = result;
            f = setdiff(fieldnames(opts), {'Viz'});
            for i = 1:numel(f), obj.(f{i}) = opts.(f{i}); end
            obj.Viz = opts.Viz;
        end

        function n = numSegments(obj)
            n = max(1, numel(obj.Result.Readers));
        end

        function tf = hasPieces(obj)
            % HASPIECES  false when nothing was transcribed (no speech detected).
            tf = ~isempty(obj.Result.Readers) && ~isempty(obj.Result.SegmentTranscripts);
        end

        function t0 = segmentStart(obj)
            % SEGMENTSTART  Start (s, on the conditioned audio) of the shown piece.
            t0 = 0;
            P = obj.Result.PieceTimes;
            if ~isempty(P), t0 = P(min(obj.SegmentIndex, size(P, 1)), 1); end
        end

        function r = reader(obj)
            r = [];
            R = obj.Result.Readers;
            if isempty(R), return; end
            k = min(obj.SegmentIndex, numel(R));
            if iscell(R), r = R{k}; else, r = R(k); end
        end

        function w = segmentWave(obj)
            W = obj.Result.SegmentWaveforms;
            if isempty(W)
                w = obj.Result.Conditioned;
            else
                k = min(obj.SegmentIndex, numel(W));
                if iscell(W), w = W{k}; else, w = W(k); end
            end
        end

        function tr = segmentTranscript(obj)
            T = obj.Result.SegmentTranscripts;
            if isempty(T)
                tr = obj.Result.Transcript;
            else
                k = min(obj.SegmentIndex, numel(T));
                if iscell(T), tr = T{k}; else, tr = T(k); end
            end
        end

        function tf = has(obj, cap)
            tf = any(obj.Capabilities == string(cap));
        end

        function v = artifact(obj, name)
            v = [];
            a = obj.artifacts("^" + regexptranslate('escape', char(name)) + "$");
            if ~isempty(a), v = a(1).Value; end
        end

        function a = artifacts(obj, pattern)
            a = struct('Name', {}, 'Stage', {}, 'Value', {}, 'Summary', {});
            r = obj.reader();
            if isempty(r), return; end
            steps = r.listSteps();
            for k = 1:numel(steps)
                if isempty(regexpi(char(steps(k).Name), char(pattern), 'once')), continue; end
                if isempty(steps(k).OutputIds), continue; end
                art = r.getArtifact(steps(k).OutputIds(1));
                if art.Status ~= "recorded", continue; end
                a(end+1) = struct('Name', string(steps(k).Name), 'Stage', string(steps(k).Stage), ...
                    'Value', art.Value, 'Summary', string(steps(k).Summary)); %#ok<AGROW>
            end
        end

        function s = stepList(obj)
            % STEPLIST  The reader's steps (Name, Stage, Summary) of the shown segment.
            s = struct('Name', {}, 'Stage', {}, 'Summary', {});
            r = obj.reader();
            if isempty(r), return; end
            st = r.listSteps();
            s = struct('Name', {st.Name}, 'Stage', {st.Stage}, 'Summary', {st.Summary});
        end

        function d = pitch(obj)
            % PITCH  PitchContour data (PitchContourStrategy contract) of the segment, on the
            %   model's frame grid (Features.PitchContour; same grid options as the features).
            key = sprintf('pitch%d', obj.SegmentIndex);
            if isfield(obj.Cache, key), d = obj.Cache.(key); return; end
            w = obj.segmentWave();
            args = obj.gridArgs();
            pc = OceanNLP.Speech.Features.PitchContour(w.SampleRate, args{:});
            d = OceanViz.Strategies.Speech.PitchContourStrategy.fromFeatureSequence( ...
                pc.forward(w), 'VoicingThreshold', pc.VoicingThreshold);
            obj.Cache.(key) = d;
        end

        function e = energy(obj)
            % ENERGY  Features.EnergyContour (ln frame energy) of the segment on the model grid.
            key = sprintf('energy%d', obj.SegmentIndex);
            if isfield(obj.Cache, key), e = obj.Cache.(key); return; end
            w = obj.segmentWave();
            args = obj.gridArgs();
            ec = OceanNLP.Speech.Features.EnergyContour(w.SampleRate, args{:});
            fs = ec.forward(w);
            e = struct('Values', fs.Values, 'FrameTimes', fs.FrameTimes, 'Units', string(fs.Units), ...
                'Floor', ec.Floor);
            obj.Cache.(key) = e;
        end

        function a = alignment(obj)
            % ALIGNMENT  Reference vs merged hypothesis on the vocabulary unit
            %   (Evaluation.EditAlignment.align); [] without a reference.
            a = [];
            if obj.Reference == "", return; end
            if isfield(obj.Cache, 'alignment'), a = obj.Cache.alignment; return; end
            hyp = obj.Result.Transcript.Text;
            if obj.Recognizer.Vocabulary.Unit == "char"
                unit = "character";
                r = string(num2cell(char(obj.Reference)));
                h = string(num2cell(char(hyp)));
            else
                unit = "word";
                r = split(obj.Reference, " ")'; h = split(hyp, " ")';
                r = r(r ~= ""); h = h(h ~= "");
            end
            r = reshape(r, 1, []); h = reshape(h, 1, []);
            al = OceanNLP.Speech.Evaluation.EditAlignment.align(r, h);
            a = struct('Alignment', al, 'Reference', r, 'Hypothesis', h, 'Unit', unit);
            obj.Cache.alignment = a;
        end

        function T = scoreSplit(obj, candidates)
            % SCORESPLIT  CTC beam score of each candidate split into its parts
            %   score = ln p_ctc + LMWeight ln p_lm + Bonus |l|  (CTCPrefixBeamDecoder):
            %   LM = LMWeight sum_i ln p_lm(l_i | l_<i) from the decoder's own LM scorer,
            %   Bonus = Bonus |l|, CTC = score - LM - Bonus. Empty table without an LM.
            T = table();
            s = obj.Settings;
            if isempty(obj.LanguageModel) || ~isfield(s, 'LMWeight') || isempty(candidates), return; end
            sc = obj.LanguageModel.scorer();
            n = numel(candidates);
            [lm, bonus, ctc, total] = deal(zeros(n, 1));
            txt = strings(n, 1);
            for i = 1:n
                ids = candidates(i).TokenIds;
                l = 0;
                for j = 1:numel(ids)
                    l = l + sc(ids(1:j-1), ids(j));
                end
                lm(i) = s.LMWeight * l;
                bonus(i) = s.Bonus * numel(ids);
                total(i) = candidates(i).Score;
                ctc(i) = total(i) - lm(i) - bonus(i);
                txt(i) = candidates(i).Text;
            end
            T = table(txt, total, ctc, lm, bonus, 'VariableNames', ["Text", "Score", "CTC", "LM", "Bonus"]);
        end

        function args = gridArgs(obj)
            % Frame-grid options of the model's front end (so contours share its frames).
            args = {};
            f = obj.artifact("Model features");
            if isempty(f) || ~isfield(f, 'Config'), return; end
            cfg = f.Config;
            g = [];
            if isfield(cfg, 'Grid'), g = cfg.Grid;
            elseif isfield(cfg, 'LogMel') && isfield(cfg.LogMel, 'Grid'), g = cfg.LogMel.Grid; end
            if isempty(g), return; end
            args = {'WindowLength', g.WindowLength, 'HopLength', g.HopLength, 'NFFT', g.NFFT, ...
                'Window', g.Window, 'Periodic', g.Periodic, 'Center', g.Center, 'PadMode', g.PadMode};
        end
    end
end
