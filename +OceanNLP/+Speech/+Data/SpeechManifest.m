classdef SpeechManifest < handle
    % SPEECHMANIFEST  File-backed list of utterances with transcripts, speakers and splits.
    %
    %   m = OceanNLP.Speech.Data.SpeechManifest(entries, 'Root', root, 'Name', "FSDD")
    %   m = OceanNLP.Speech.Data.SpeechManifest.fromFSDD()          % or fromFSDD(root)
    %   m = OceanNLP.Speech.Data.SpeechManifest.load(file, 'Root', root)
    %   r = OceanNLP.Speech.Data.SpeechManifest.defaultRoot()
    %   s = m.canonicalFSDDSplit();   s = m.splitBySpeaker(train, val, test)
    %   m.validate(); t = m.audit(); m.save(file); p = m.resolve(i); n = m.height()
    %   s = m.subset(mask); m.assertDisjoint(other, 'By', "speaker"|"id")
    %
    %   ENTRIES (table, one row per utterance; Phase 0 contract sec. 7):
    %     Id          unique string
    %     AudioPath   path RELATIVE to Root ("/" separators); a composed utterance
    %                 (UtteranceComposer) lists its component files joined by "+"
    %     Transcript  raw transcript (normalization happens downstream)
    %     Language    BCP-47 tag ("en"), "" when unknown
    %     Speaker     speaker id;  Session: "" when the corpus records none (FSDD)
    %     Duration    seconds (from audioinfo: TotalSamples / SampleRate)
    %     SampleRate  Hz;  Split: "train" | "validation" | "test" | "" (unassigned)
    %     License     licence of the source audio
    %
    %   Root is never written to disk (save stores only root-relative paths); it is
    %   supplied when loading, defaulting to defaultRoot() = <repo>/Data/Datasets/Speech,
    %   derived from this file's location.
    %
    %   FSDD (Free Spoken Digit Dataset, CC BY-SA 4.0): recordings named
    %   <digit>_<speaker>_<take>.wav; Transcript = English word of <digit> (the
    %   dataset's own label). Canonical split (contract sec. 1): test = speaker
    %   "theo"; validation = takes 45..49 of the other speakers (speaker-DEPENDENT,
    %   for model selection only); train = the remaining takes of those speakers.
    %   With 50 takes x 10 digits x 4 speakers: train 1350, validation 150, test 500.

    properties (Constant)
        Columns = ["Id", "AudioPath", "Transcript", "Language", "Speaker", "Session", ...
            "Duration", "SampleRate", "Split", "License"]
        DigitWords = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        FSDDTestSpeaker = "theo"
        FSDDValidationTakes = 45:49
        FSDDLicenseHeader = "Attribution-ShareAlike 4.0 International"
    end

    properties (SetAccess = private)
        Entries table
        Root (1,1) string
        Name (1,1) string
    end

    methods
        function obj = SpeechManifest(entries, opts)
            arguments
                entries table
                opts.Root (1,1) string = OceanNLP.Speech.Data.SpeechManifest.defaultRoot()
                opts.Name (1,1) string = ""
            end
            cols = OceanNLP.Speech.Data.SpeechManifest.Columns;
            missingCols = setdiff(cols, string(entries.Properties.VariableNames), 'stable');
            if ~isempty(missingCols)
                error('OceanNLP:Speech:Data:SpeechManifest:Columns', 'Entries lack columns: %s', join(missingCols, ', '));
            end
            entries = entries(:, cols);
            for c = cols(~ismember(cols, ["Duration", "SampleRate"]))
                v = entries.(c);
                if isempty(v), v = strings(0, 1); end
                entries.(c) = reshape(string(v), [], 1);
                if any(ismissing(entries.(c)))
                    error('OceanNLP:Speech:Data:SpeechManifest:Missing', 'Column %s has <missing> values.', c);
                end
            end
            entries.Duration = reshape(double(entries.Duration), [], 1);
            entries.SampleRate = reshape(double(entries.SampleRate), [], 1);
            if any(~isfinite(entries.Duration) | entries.Duration < 0) || any(~isfinite(entries.SampleRate) | entries.SampleRate <= 0)
                error('OceanNLP:Speech:Data:SpeechManifest:Numeric', 'Duration must be finite >= 0 and SampleRate finite > 0.');
            end
            entries.AudioPath = replace(entries.AudioPath, "\", "/");
            comps = OceanNLP.Speech.Data.SpeechManifest.componentsOf(entries.AudioPath);
            bad = cellfun(@(c) any(c == "" | startsWith(c, "/") | startsWith(c, lettersPattern(1) + ":")), comps);
            if any(bad)
                error('OceanNLP:Speech:Data:SpeechManifest:AbsolutePath', ...
                    'AudioPath must be relative to Root and non-empty; offenders: %s', join(entries.Id(bad), ', '));
            end
            [~, first] = unique(entries.Id, 'stable');
            dup = setdiff(1:height(entries), first);
            if ~isempty(dup)
                error('OceanNLP:Speech:Data:SpeechManifest:DuplicateId', 'Duplicate ids: %s', join(unique(entries.Id(dup)), ', '));
            end
            obj.Entries = entries;
            obj.Root = opts.Root;
            obj.Name = opts.Name;
        end

        function n = height(obj)
            n = height(obj.Entries);
        end

        function p = resolve(obj, i)
            % RESOLVE  Absolute path(s) of entry i's audio (1 x k string; k > 1 when composed).
            arguments
                obj
                i (1,1) double {mustBeInteger, mustBePositive}
            end
            if i > obj.height()
                error('OceanNLP:Speech:Data:SpeechManifest:Index', 'Index %d exceeds %d entries.', i, obj.height());
            end
            c = OceanNLP.Speech.Data.SpeechManifest.componentsOf(obj.Entries.AudioPath(i));
            p = arrayfun(@(r) string(fullfile(obj.Root, r)), c{1});
        end

        function report = validate(obj, opts)
            % VALIDATE  Every file exists and audioinfo agrees with Duration/SampleRate.
            %   Errors with ':Invalid', listing EVERY offender; otherwise returns a
            %   report struct (NumEntries, NumFiles). Duration tolerance: 1e-9 s.
            arguments
                obj
                opts.Tolerance (1,1) double {mustBeNonnegative} = 1e-9
            end
            E = obj.Entries;
            problems = strings(0, 1);
            nFiles = 0;
            for i = 1:height(E)
                files = obj.resolve(i);
                dur = 0; rates = zeros(1, numel(files)); ok = true;
                for f = 1:numel(files)
                    nFiles = nFiles + 1;
                    if ~isfile(files(f))
                        problems(end+1) = sprintf("%s: missing file %s", E.Id(i), files(f)); %#ok<AGROW>
                        ok = false;
                        continue;
                    end
                    try
                        info = audioinfo(files(f));
                    catch ME
                        problems(end+1) = sprintf("%s: unreadable %s (%s)", E.Id(i), files(f), ME.message); %#ok<AGROW>
                        ok = false;
                        continue;
                    end
                    dur = dur + info.TotalSamples / info.SampleRate;
                    rates(f) = info.SampleRate;
                end
                if ~ok, continue; end
                if any(rates ~= E.SampleRate(i))
                    problems(end+1) = sprintf("%s: SampleRate %g but file(s) at %s Hz", E.Id(i), E.SampleRate(i), mat2str(rates)); %#ok<AGROW>
                end
                if abs(dur - E.Duration(i)) > opts.Tolerance
                    problems(end+1) = sprintf("%s: Duration %.9g s but audio is %.9g s", E.Id(i), E.Duration(i), dur); %#ok<AGROW>
                end
            end
            if ~isempty(problems)
                error('OceanNLP:Speech:Data:SpeechManifest:Invalid', ...
                    'Manifest "%s": %d problem(s):\n%s', obj.Name, numel(problems), join(problems, newline));
            end
            report = struct('NumEntries', height(E), 'NumFiles', nFiles);
        end

        function s = subset(obj, mask)
            % SUBSET  New manifest with the selected rows (logical mask or indices).
            arguments
                obj
                mask {mustBeVector}
            end
            s = OceanNLP.Speech.Data.SpeechManifest(obj.Entries(mask, :), 'Root', obj.Root, 'Name', obj.Name);
        end

        function s = bySplit(obj, split)
            % BYSPLIT  Rows whose Split equals SPLIT.
            arguments
                obj
                split (1,1) string
            end
            s = obj.subset(obj.Entries.Split == split);
        end

        function s = splitBySpeaker(obj, train, val, test)
            % SPLITBYSPEAKER  Speaker-disjoint split; speakers not listed are dropped.
            %   e.g. leave-one-speaker-out: splitBySpeaker(["a","b"], "c", "d").
            arguments
                obj
                train string
                val string
                test string
            end
            train = reshape(train, [], 1); val = reshape(val, [], 1); test = reshape(test, [], 1);
            all3 = [train; val; test];
            if numel(unique(all3)) ~= numel(all3)
                error('OceanNLP:Speech:Data:SpeechManifest:Overlap', 'A speaker appears in more than one split.');
            end
            unknown = setdiff(all3, obj.Entries.Speaker);
            if ~isempty(unknown)
                error('OceanNLP:Speech:Data:SpeechManifest:UnknownSpeaker', 'Unknown speaker(s): %s', join(unknown, ', '));
            end
            E = obj.Entries;
            E.Split(:) = "";
            E.Split(ismember(E.Speaker, train)) = "train";
            E.Split(ismember(E.Speaker, val)) = "validation";
            E.Split(ismember(E.Speaker, test)) = "test";
            E = E(E.Split ~= "", :);
            s = OceanNLP.Speech.Data.SpeechManifest(E, 'Root', obj.Root, 'Name', obj.Name);
        end

        function s = canonicalFSDDSplit(obj)
            % CANONICALFSDDSPLIT  Contract sec. 1 split (test speaker theo; val takes 45-49).
            tok = regexp(obj.Entries.Id, '^(\d)_([A-Za-z]+)_(\d+)$', 'tokens', 'once');
            if any(cellfun(@isempty, tok))
                error('OceanNLP:Speech:Data:SpeechManifest:NotFSDD', 'Ids are not FSDD <digit>_<speaker>_<take>.');
            end
            if ~ismember(OceanNLP.Speech.Data.SpeechManifest.FSDDTestSpeaker, obj.Entries.Speaker)
                error('OceanNLP:Speech:Data:SpeechManifest:UnknownSpeaker', 'Test speaker "%s" is absent.', OceanNLP.Speech.Data.SpeechManifest.FSDDTestSpeaker);
            end
            take = cellfun(@(t) str2double(t{3}), tok);
            E = obj.Entries;
            isTest = E.Speaker == OceanNLP.Speech.Data.SpeechManifest.FSDDTestSpeaker;
            E.Split(:) = "train";
            E.Split(~isTest & ismember(take, OceanNLP.Speech.Data.SpeechManifest.FSDDValidationTakes)) = "validation";
            E.Split(isTest) = "test";
            s = OceanNLP.Speech.Data.SpeechManifest(E, 'Root', obj.Root, 'Name', obj.Name);
        end

        function assertDisjoint(obj, other, opts)
            arguments
                obj
                other (1,1) OceanNLP.Speech.Data.SpeechManifest
                opts.By (1,1) string {mustBeMember(opts.By, ["speaker", "id"])} = "id"
            end
            col = "Id";
            if opts.By == "speaker", col = "Speaker"; end
            common = intersect(obj.Entries.(col), other.Entries.(col));
            if ~isempty(common)
                shown = common(1:min(20, end));
                error('OceanNLP:Speech:Data:SpeechManifest:Overlap', ...
                    '%d shared %s value(s) between "%s" and "%s": %s', numel(common), lower(col), ...
                    obj.Name, other.Name, join(shown, ', '));
            end
        end

        function t = audit(obj)
            % AUDIT  Per (Split, Speaker) counts and duration statistics, plus an ALL row.
            %   Columns: Split, Speaker, NumUtterances, NumWords, TotalDuration,
            %   MinDuration, MeanDuration, MedianDuration, MaxDuration (seconds),
            %   SampleRates (distinct, as text), Languages (distinct).
            E = obj.Entries;
            if isempty(E)
                error('OceanNLP:Speech:Data:SpeechManifest:Empty', 'Cannot audit an empty manifest.');
            end
            [g, split, speaker] = findgroups(E.Split, E.Speaker);
            t = OceanNLP.Speech.Data.SpeechManifest.stats(E, g, split, speaker);
            t = [t; OceanNLP.Speech.Data.SpeechManifest.stats(E, ones(height(E), 1), "ALL", "ALL")];
        end

        function save(obj, file)
            % SAVE  JSON with root-relative paths (Root itself is not stored).
            arguments
                obj
                file (1,1) string
            end
            s = struct('Format', "OceanSpeechManifest", 'Version', 1, 'Name', obj.Name, ...
                'Columns', OceanNLP.Speech.Data.SpeechManifest.Columns, ...
                'Entries', table2struct(obj.Entries));
            fid = fopen(file, 'w', 'n', 'UTF-8');
            if fid < 0
                error('OceanNLP:Speech:Data:SpeechManifest:Write', 'Cannot write %s.', file);
            end
            closer = onCleanup(@() fclose(fid));
            fwrite(fid, jsonencode(s, 'PrettyPrint', true), 'char');
        end
    end

    methods (Static)
        function r = defaultRoot()
            % DEFAULTROOT  <repo>/Data/Datasets/Speech, from this file's location.
            here = mfilename('fullpath');            % <repo>/+OceanNLP/+Speech/+Data/SpeechManifest
            repo = fileparts(fileparts(fileparts(fileparts(here))));
            r = string(fullfile(repo, 'Data', 'Datasets', 'Speech'));
        end

        function m = fromFSDD(root)
            % FROMFSDD  Manifest of <root>/FSDD/recordings/*.wav with audioinfo metadata.
            arguments
                root (1,1) string = OceanNLP.Speech.Data.SpeechManifest.defaultRoot()
            end
            recDir = fullfile(root, 'FSDD', 'recordings');
            if ~isfolder(recDir)
                error('OceanNLP:Speech:Data:SpeechManifest:DatasetNotFound', 'FSDD recordings not found at %s', recDir);
            end
            licFile = fullfile(root, 'FSDD', 'license.txt');
            if ~isfile(licFile)
                error('OceanNLP:Speech:Data:SpeechManifest:LicenseNotFound', 'FSDD license.txt not found at %s', licFile);
            end
            header = strip(string(readlines(licFile)));
            header = header(header ~= "");
            if isempty(header) || header(1) ~= OceanNLP.Speech.Data.SpeechManifest.FSDDLicenseHeader
                error('OceanNLP:Speech:Data:SpeechManifest:License', ...
                    'FSDD license.txt does not start with "%s".', OceanNLP.Speech.Data.SpeechManifest.FSDDLicenseHeader);
            end
            license = "CC BY-SA 4.0 (FSDD/license.txt)";

            d = dir(fullfile(recDir, '*.wav'));
            names = string({d.name})';
            tok = regexp(names, '^(\d)_([A-Za-z]+)_(\d+)\.wav$', 'tokens', 'once');
            bad = cellfun(@isempty, tok);
            if any(bad)
                error('OceanNLP:Speech:Data:SpeechManifest:FileName', ...
                    'Unexpected FSDD file name(s): %s', join(names(bad), ', '));
            end
            if isempty(names)
                error('OceanNLP:Speech:Data:SpeechManifest:DatasetNotFound', 'No .wav files in %s', recDir);
            end
            digit = cellfun(@(t) str2double(t{1}), tok);
            speaker = string(cellfun(@(t) t{2}, tok, 'UniformOutput', false));
            take = cellfun(@(t) str2double(t{3}), tok);
            [~, ord] = sortrows(table(speaker, digit, take));
            names = names(ord); digit = digit(ord); speaker = speaker(ord);

            n = numel(names);
            dur = zeros(n, 1); fs = zeros(n, 1);
            for k = 1:n
                info = audioinfo(fullfile(recDir, names(k)));
                dur(k) = info.TotalSamples / info.SampleRate;
                fs(k) = info.SampleRate;
            end
            Id = extractBefore(names, ".wav");
            AudioPath = "FSDD/recordings/" + names;
            Transcript = reshape(OceanNLP.Speech.Data.SpeechManifest.DigitWords(digit + 1), [], 1);
            Language = repmat("en", n, 1);
            Session = repmat("", n, 1);
            Split = repmat("", n, 1);
            License = repmat(license, n, 1);
            E = table(Id, AudioPath, Transcript, Language, speaker, Session, dur, fs, Split, License, ...
                'VariableNames', OceanNLP.Speech.Data.SpeechManifest.Columns);
            m = OceanNLP.Speech.Data.SpeechManifest(E, 'Root', root, 'Name', "FSDD");
        end

        function m = load(file, opts)
            arguments
                file (1,1) string
                opts.Root (1,1) string = OceanNLP.Speech.Data.SpeechManifest.defaultRoot()
            end
            if ~isfile(file)
                error('OceanNLP:Speech:Data:SpeechManifest:FileNotFound', 'Manifest file not found: %s', file);
            end
            s = jsondecode(fileread(file));
            if ~isfield(s, 'Format') || string(s.Format) ~= "OceanSpeechManifest"
                error('OceanNLP:Speech:Data:SpeechManifest:Format', '%s is not an Ocean speech manifest.', file);
            end
            cols = OceanNLP.Speech.Data.SpeechManifest.Columns;
            if isempty(s.Entries)
                E = table('Size', [0 numel(cols)], 'VariableTypes', ...
                    ["string", "string", "string", "string", "string", "string", "double", "double", "string", "string"], ...
                    'VariableNames', cols);
            else
                E = struct2table(s.Entries(:), 'AsArray', true);
                for c = cols(~ismember(cols, ["Duration", "SampleRate"]))
                    E.(c) = string(E.(c));
                end
            end
            m = OceanNLP.Speech.Data.SpeechManifest(E, 'Root', opts.Root, 'Name', string(s.Name));
        end
    end

    methods (Static, Hidden)
        function c = componentsOf(audioPaths)
            % 1 x n cell of 1 x k string arrays (composed paths split on "+").
            c = arrayfun(@(p) reshape(split(p, "+"), 1, []), reshape(audioPaths, 1, []), 'UniformOutput', false);
        end
    end

    methods (Static, Access = private)
        function t = stats(E, g, split, speaker)
            words = arrayfun(@(s) numel(OceanNLP.Speech.Evaluation.WordErrorRate.tokenize(s)), E.Transcript);
            d = E.Duration;
            f = @(fn) splitapply(fn, d, g);
            rates = splitapply(@(r) join(string(unique(r))', ","), E.SampleRate, g);
            langs = splitapply(@(l) join(unique(l)', ","), E.Language, g);
            t = table(string(split), string(speaker), splitapply(@numel, d, g), splitapply(@sum, words, g), ...
                f(@sum), f(@min), f(@mean), f(@median), f(@max), rates, langs, 'VariableNames', ...
                ["Split", "Speaker", "NumUtterances", "NumWords", "TotalDuration", "MinDuration", ...
                "MeanDuration", "MedianDuration", "MaxDuration", "SampleRates", "Languages"]);
        end
    end
end
