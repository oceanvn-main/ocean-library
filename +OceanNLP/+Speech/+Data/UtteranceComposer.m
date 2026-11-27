classdef UtteranceComposer < handle
    % UTTERANCECOMPOSER  Multi-word utterances by concatenating real recordings.
    %
    %   c = OceanNLP.Speech.Data.UtteranceComposer(manifest, 'MinWords', 2, 'MaxWords', 5, ...
    %         'Count', n, 'Seed', s)
    %   m = c.forward()                 derived SpeechManifest
    %   m = OceanNLP.Speech.Data.UtteranceComposer.compose(manifest, ...)   (one call)
    %
    %   For u = 1..Count, with a private RandStream('mt19937ar', Seed):
    %     a ~ U{1..H}                     anchor row (groups weighted by their size)
    %     k ~ U{MinWords..MaxWords}       capped at the size of a's group
    %     the other k-1 rows ~ uniformly without replacement from a's group, where
    %     group = same Speaker AND same Split (a composed utterance never mixes
    %     speakers or splits, so speaker-disjoint evaluation stays speaker-disjoint)
    %   Entry: AudioPath = component paths joined by "+" (provenance; AudioDataset
    %   concatenates them in order), Transcript = component transcripts joined by
    %   one space, Duration = sum, Id = "c<u>_" + component ids joined by "+".
    %   No audio is synthesized beyond this concatenation (contract sec. 1).
    %   A group smaller than MinWords raises ':GroupTooSmall' when drawn.

    properties (SetAccess = immutable)
        Manifest
        MinWords (1,1) double
        MaxWords (1,1) double
        Count (1,1) double
        Seed (1,1) double
    end

    methods
        function obj = UtteranceComposer(manifest, opts)
            arguments
                manifest (1,1) OceanNLP.Speech.Data.SpeechManifest
                opts.MinWords (1,1) double {mustBeInteger, mustBePositive} = 2
                opts.MaxWords (1,1) double {mustBeInteger, mustBePositive} = 5
                opts.Count (1,1) double {mustBeInteger, mustBeNonnegative} = manifest.height()
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            if opts.MinWords > opts.MaxWords
                error('OceanNLP:Speech:Data:UtteranceComposer:Range', 'MinWords must be <= MaxWords.');
            end
            if manifest.height() == 0
                error('OceanNLP:Speech:Data:UtteranceComposer:Empty', 'Cannot compose from an empty manifest.');
            end
            obj.Manifest = manifest;
            obj.MinWords = opts.MinWords;
            obj.MaxWords = opts.MaxWords;
            obj.Count = opts.Count;
            obj.Seed = opts.Seed;
        end

        function m = forward(obj)
            E = obj.Manifest.Entries;
            if any(contains(E.Id, "+"))
                error('OceanNLP:Speech:Data:UtteranceComposer:AlreadyComposed', 'Input entries must be single recordings.');
            end
            stream = RandStream('mt19937ar', 'Seed', obj.Seed);
            g = findgroups(E.Speaker, E.Split);
            U = obj.Count;
            Id = strings(U, 1); AudioPath = Id; Transcript = Id; Language = Id;
            Speaker = Id; Split = Id; License = Id; Session = Id;
            Duration = zeros(U, 1); SampleRate = zeros(U, 1);
            for u = 1:U
                a = randi(stream, height(E));
                members = find(g == g(a));
                if numel(members) < obj.MinWords
                    error('OceanNLP:Speech:Data:UtteranceComposer:GroupTooSmall', ...
                        'Speaker "%s" / split "%s" has %d recording(s) < MinWords = %d.', ...
                        E.Speaker(a), E.Split(a), numel(members), obj.MinWords);
                end
                k = randi(stream, [obj.MinWords, obj.MaxWords]);
                k = min(k, numel(members));
                others = members(members ~= a);
                pick = [a; others(randperm(stream, numel(others), k - 1))];
                rows = E(pick, :);
                if numel(unique(rows.SampleRate)) > 1 || numel(unique(rows.Language)) > 1
                    error('OceanNLP:Speech:Data:UtteranceComposer:Mismatch', ...
                        'Components of speaker "%s" differ in sample rate or language.', E.Speaker(a));
                end
                Id(u) = sprintf("c%d_", u) + join(rows.Id, "+");
                AudioPath(u) = join(rows.AudioPath, "+");
                Transcript(u) = join(rows.Transcript, " ");
                Language(u) = rows.Language(1);
                Speaker(u) = rows.Speaker(1);
                Split(u) = rows.Split(1);
                Session(u) = join(unique(rows.Session), "+");
                License(u) = join(unique(rows.License), "; ");
                Duration(u) = sum(rows.Duration);
                SampleRate(u) = rows.SampleRate(1);
            end
            T = table(Id, AudioPath, Transcript, Language, Speaker, Session, Duration, SampleRate, Split, License);
            m = OceanNLP.Speech.Data.SpeechManifest(T, 'Root', obj.Manifest.Root, ...
                'Name', obj.Manifest.Name + "-composed");
        end

        function cfg = config(obj)
            cfg = struct('Type', "UtteranceComposer", 'MinWords', obj.MinWords, 'MaxWords', obj.MaxWords, ...
                'Count', obj.Count, 'Seed', obj.Seed, 'Source', obj.Manifest.Name);
        end
    end

    methods (Static)
        function m = compose(manifest, varargin)
            % COMPOSE  UtteranceComposer(manifest, ...).forward() in one call.
            m = OceanNLP.Speech.Data.UtteranceComposer(manifest, varargin{:}).forward();
        end
    end
end
