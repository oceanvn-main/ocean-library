classdef Waveform < handle
    % WAVEFORM  Sampled audio with its sample rate and provenance (immutable).
    %
    %   w = OceanNLP.Speech.Core.Waveform(x, fs)
    %   w = OceanNLP.Speech.Core.Waveform(x, fs, 'Source', "recordings/0_theo_0.wav")
    %   w = OceanNLP.Speech.Core.Waveform.read(file, 'Root', corpusRoot)
    %   w = OceanNLP.Speech.Core.Waveform.read(file, 'Range', [t0 t1], 'Units', "seconds")
    %   s = w.segment([n1 n2])                       samples n1..n2 (1-based, inclusive)
    %   s = w.segment([t0 t1], 'Units', "seconds")   [t0, t1) relative to w's first sample
    %   c = OceanNLP.Speech.Core.Waveform.concat(w1, w2, ...)
    %   t = w.times()                                S x 1 sample times in seconds
    %
    %   A discrete-time signal x[n] = x_a(n / f_s), n = 0..S-1, stored as an
    %   S x C matrix (rows = samples, columns = channels). Duration S / f_s s.
    %
    %   CONTRACT (Planning/NLP/Speech_ASR_Phase0_Contracts.md, sec. 2 and 4):
    %     - samples are real, finite, numeric, non-empty; stored as double
    %     - a 1 x S row with S > 1 is rejected (ambiguous: one multichannel
    %       sample or S mono samples?) -- pass x(:) for mono
    %     - more channels than samples (C > S) is rejected: such a matrix is
    %       almost always a transposed C x S signal -- pass x.'
    %     - the sample rate is a positive finite scalar in Hz; changing it is
    %       only legal together with new samples (derive), never by relabelling
    %     - History records every operation that produced these samples
    %       (resampling, channel mixing, augmentation, concatenation)
    %     - SourceOffset is the time (s) of the first sample inside Source.
    %       read('Range') and segment set it; derive keeps it unless the caller
    %       passes 'SourceOffset' (e.g. a trim that moves the start). Seconds,
    %       not samples, so it survives resampling.
    %     - Source is stored relative to the corpus Root with '/' separators.
    %       A file outside Root keeps its absolute path and raises the warning
    %       'OceanNLP:Speech:Core:Waveform:OutsideRoot'.
    %
    %   PERFORMANCE NOTES
    %     - Validation is one allfinite pass (no S x C logical temporary) and is
    %       skipped for integer input, which cannot hold NaN/Inf.
    %     - read('Range') decodes only the requested samples (audioread range);
    %       use it for long recordings instead of read + segment.
    %     - toMono / segment of an already-mono / full-range waveform return
    %       the same (immutable) object; no copy, no History entry.
    %     - Storage is always double (8 bytes per sample): about 460 MB per hour
    %       of 16 kHz mono. Read long files in ranges.

    properties (SetAccess = immutable)
        Samples (:,:) double
        SampleRate (1,1) double
        Source (1,1) string = ""
        History (1,:) string = strings(1, 0)
        SourceOffset (1,1) double = 0
    end

    methods
        function obj = Waveform(samples, sampleRate, opts)
            arguments
                samples {mustBeNumeric}
                sampleRate (1,1) double
                opts.Source (1,1) string = ""
                opts.History (1,:) string = strings(1, 0)
                opts.SourceOffset (1,1) double = 0
            end
            if isempty(samples)
                error('OceanNLP:Speech:Core:Waveform:Empty', 'A waveform needs at least one sample.');
            end
            if ~isreal(samples) || (isfloat(samples) && ~allfinite(samples))
                error('OceanNLP:Speech:Core:Waveform:Nonfinite', 'Samples must be real and finite.');
            end
            [S, C] = size(samples);
            if ~ismatrix(samples) || (S == 1 && C > 1)
                error('OceanNLP:Speech:Core:Waveform:Orientation', ...
                    'Samples must be S x C (rows = samples). Got %s; pass x(:) for a mono signal.', mat2str(size(samples)));
            end
            if C > S
                error('OceanNLP:Speech:Core:Waveform:Orientation', ...
                    ['Samples are %d x %d: more channels than samples, which is almost always a ', ...
                    'transposed C x S signal. Pass x.'' (rows = samples).'], S, C);
            end
            if ~(isfinite(sampleRate) && sampleRate > 0)
                error('OceanNLP:Speech:Core:Waveform:SampleRate', 'Sample rate must be a positive finite number of Hz.');
            end
            if ~(isfinite(opts.SourceOffset) && opts.SourceOffset >= 0)
                error('OceanNLP:Speech:Core:Waveform:SourceOffset', 'SourceOffset must be a finite number of seconds >= 0.');
            end
            obj.Samples = double(samples);          % no copy when already double
            obj.SampleRate = sampleRate;
            obj.Source = opts.Source;
            obj.History = opts.History;
            obj.SourceOffset = opts.SourceOffset;
        end

        function n = numSamples(obj)
            n = size(obj.Samples, 1);
        end

        function c = numChannels(obj)
            c = size(obj.Samples, 2);
        end

        function d = duration(obj)
            % DURATION  S / f_s seconds.
            d = obj.numSamples() / obj.SampleRate;
        end

        function t = times(obj, opts)
            % TIMES  S x 1 sample times t_n = (n - 1) / f_s in seconds.
            %   'Reference' "waveform" (default, t_1 = 0) | "source" (adds SourceOffset).
            arguments
                obj
                opts.Reference (1,1) string {mustBeMember(opts.Reference, ["waveform", "source"])} = "waveform"
            end
            t = (0:obj.numSamples() - 1)' / obj.SampleRate;
            if opts.Reference == "source"
                t = t + obj.SourceOffset;
            end
        end

        function w = toMono(obj, method)
            % TOMONO  Mix down to one channel: "mean" averages channels, "first" keeps channel 1.
            arguments
                obj
                method (1,1) string {mustBeMember(method, ["mean", "first"])} = "mean"
            end
            if obj.numChannels() == 1
                w = obj;
                return;
            end
            if method == "mean"
                x = mean(obj.Samples, 2);
            else
                x = obj.Samples(:, 1);
            end
            w = obj.derive(x, obj.SampleRate, sprintf("mono %s of %d channels", method, obj.numChannels()));
        end

        function w = segment(obj, range, opts)
            % SEGMENT  Contiguous part of the waveform; SourceOffset and History follow.
            %   'Units' "samples" (default): RANGE = [n1 n2], 1-based, inclusive.
            %   'Units' "seconds": RANGE = [t0 t1), relative to this waveform's first
            %   sample, each end rounded to the nearest sample: n1 = round(t0 f_s) + 1,
            %   n2 = round(t1 f_s).
            arguments
                obj
                range (1,2) double
                opts.Units (1,1) string {mustBeMember(opts.Units, ["samples", "seconds"])} = "samples"
            end
            [n1, n2] = OceanNLP.Speech.Core.Waveform.toSampleRange(range, opts.Units, obj.SampleRate, obj.numSamples());
            if n1 == 1 && n2 == obj.numSamples()
                w = obj;
                return;
            end
            fs = obj.SampleRate;
            note = sprintf("segment samples %d-%d (%.4f-%.4f s)", n1, n2, (n1 - 1) / fs, n2 / fs);
            w = OceanNLP.Speech.Core.Waveform(obj.Samples(n1:n2, :), fs, 'Source', obj.Source, ...
                'History', [obj.History, note], 'SourceOffset', obj.SourceOffset + (n1 - 1) / fs);
        end

        function w = derive(obj, samples, sampleRate, note, opts)
            % DERIVE  New waveform produced from this one; appends NOTE to History.
            %   'SourceOffset' overrides the inherited offset (seconds) when the
            %   transform moves the first sample (trimming, cropping).
            arguments
                obj
                samples {mustBeNumeric}
                sampleRate (1,1) double
                note (1,1) string
                opts.SourceOffset (1,1) double = obj.SourceOffset
            end
            w = OceanNLP.Speech.Core.Waveform(samples, sampleRate, ...
                'Source', obj.Source, 'History', [obj.History, note], 'SourceOffset', opts.SourceOffset);
        end
    end

    methods (Static)
        function w = read(file, opts)
            % READ  Load an audio file with base-MATLAB audioread (double, S x C).
            %   'Root': corpus root; FILE may be relative to it. Source is stored
            %   relative to Root (no developer-machine absolute paths in provenance).
            %   'Range' [a b] with 'Units' "samples" (1-based, inclusive) or
            %   "seconds" ([t0, t1), rounded as in segment): decode only that part;
            %   SourceOffset = (a - 1) / f_s. Default: the whole file.
            arguments
                file (1,1) string
                opts.Root (1,1) string = ""
                opts.Range double = []
                opts.Units (1,1) string {mustBeMember(opts.Units, ["samples", "seconds"])} = "samples"
            end
            [path, source] = OceanNLP.Speech.Core.Waveform.resolvePath(file, opts.Root);
            if ~isfile(path)
                error('OceanNLP:Speech:Core:Waveform:FileNotFound', 'Audio file not found: %s', path);
            end
            offset = 0;
            if isempty(opts.Range)
                [x, fs] = audioread(path, 'double');
            else
                if ~isequal(size(opts.Range), [1 2])
                    error('OceanNLP:Speech:Core:Waveform:Range', 'Range must be a 1 x 2 vector [start stop].');
                end
                info = audioinfo(path);
                [n1, n2] = OceanNLP.Speech.Core.Waveform.toSampleRange(opts.Range, opts.Units, ...
                    info.SampleRate, info.TotalSamples);
                [x, fs] = audioread(path, [n1 n2], 'double');
                offset = (n1 - 1) / fs;
            end
            w = OceanNLP.Speech.Core.Waveform(x, fs, 'Source', source, 'SourceOffset', offset);
        end

        function w = concat(varargin)
            % CONCAT  Vertical (time) concatenation of waveforms with equal f_s and C.
            %   Source = component Sources joined with "+"; SourceOffset = the first
            %   component's; History = one entry naming the count, plus the
            %   component histories when any is non-empty.
            if nargin == 0 || ~all(cellfun(@(a) isa(a, 'OceanNLP.Speech.Core.Waveform') && isscalar(a), varargin))
                error('OceanNLP:Speech:Core:Waveform:Concat', 'concat takes one or more scalar Waveform objects.');
            end
            if nargin == 1
                w = varargin{1};
                return;
            end
            rates = cellfun(@(a) a.SampleRate, varargin);
            chans = cellfun(@(a) a.numChannels(), varargin);
            if any(rates ~= rates(1)) || any(chans ~= chans(1))
                error('OceanNLP:Speech:Core:Waveform:ConcatMismatch', ...
                    'Waveforms differ in sample rate (%s Hz) or channel count (%s).', mat2str(rates), mat2str(chans));
            end
            parts = cellfun(@(a) a.Samples, varargin, 'UniformOutput', false);
            note = sprintf("concatenated %d waveforms", nargin);
            hasHist = cellfun(@(a) ~isempty(a.History), varargin);
            if any(hasHist)
                hist = repmat("[]", 1, nargin);
                for k = find(hasHist)
                    hist(k) = "[" + join(varargin{k}.History, "; ") + "]";
                end
                note = note + " (component histories: " + join(hist, " + ") + ")";
            end
            sources = cellfun(@(a) a.Source, varargin);
            w = OceanNLP.Speech.Core.Waveform(vertcat(parts{:}), rates(1), 'Source', join(sources, "+"), ...
                'History', note, 'SourceOffset', varargin{1}.SourceOffset);
        end
    end

    methods (Static, Access = private)
        function [n1, n2] = toSampleRange(range, units, fs, S)
            % Convert a [start stop] range to 1-based inclusive sample indices and check bounds.
            if ~all(isfinite(range))
                error('OceanNLP:Speech:Core:Waveform:Range', 'Range must be finite.');
            end
            if units == "seconds"
                n1 = round(range(1) * fs) + 1;
                n2 = round(range(2) * fs);
            else
                if any(range ~= round(range))
                    error('OceanNLP:Speech:Core:Waveform:Range', 'Sample ranges must be integers.');
                end
                n1 = range(1);
                n2 = range(2);
            end
            if n1 < 1 || n2 > S || n1 > n2
                error('OceanNLP:Speech:Core:Waveform:Range', ...
                    'Range %s (%s) maps to samples %d-%d, outside 1-%d or empty.', mat2str(range), units, n1, n2, S);
            end
        end

        function [path, source] = resolvePath(file, root)
            % File path to open, and the provenance string (Root-relative, '/' separators).
            norm = @OceanNLP.Speech.Core.Waveform.normalizePath;
            if root == ""
                path = file;
                source = norm(file);
            elseif ~OceanNLP.Speech.Core.Waveform.isAbsolute(file)
                path = fullfile(root, file);
                source = regexprep(norm(file), '^(\./)+', '');
            else
                path = file;
                r = regexprep(norm(root), '/+$', '');
                p = norm(file);
                if ispc     % Windows paths are case-insensitive
                    inside = startsWith(lower(p), lower(r) + "/");
                else
                    inside = startsWith(p, r + "/");
                end
                if inside
                    source = extractAfter(p, strlength(r) + 1);
                else
                    warning('OceanNLP:Speech:Core:Waveform:OutsideRoot', ...
                        'File %s is outside Root %s; Source keeps the absolute path.', file, root);
                    source = p;
                end
            end
        end

        function p = normalizePath(p)
            % '\' -> '/', and repeated separators collapsed (a leading UNC '//' is kept).
            p = regexprep(replace(p, '\', '/'), '(?<=.)/{2,}', '/');
        end

        function tf = isAbsolute(p)
            tf = startsWith(p, ["/", "\"]) || ~isempty(regexp(p, '^[A-Za-z]:', 'once'));
        end
    end
end
