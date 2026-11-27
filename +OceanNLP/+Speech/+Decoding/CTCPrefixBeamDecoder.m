classdef CTCPrefixBeamDecoder < handle
    % CTCPREFIXBEAMDECODER  Speech adapter: CTC logits -> Core.Transcript (greedy or prefix beam).
    %
    %   d  = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(vocab, 'Method', "beam", ...
    %          'BeamWidth', 8, 'Prune', -inf, 'NBest', 8, 'LanguageModel', [], 'LMWeight', 0, ...
    %          'Bonus', 0, 'ModelId', "")
    %   tr = d.decode(logits, lengths)     logits K x N x T (Tensor or array), lengths 1 x N
    %                                      -> 1 x N Core.Transcript
    %
    %   The algorithms are not re-implemented here:
    %     greedy  OceanDL.Recurrent.CTCDecoder.greedy  (best path, then B = merge repeats,
    %             drop blanks). Score = log prob of that best path,
    %             sum_t max_k log y_k(t) over the valid frames -- a path score, not
    %             log p(labelling | x). Candidates = that single hypothesis.
    %     beam    OceanDL.Recurrent.CTCBeamSearch.decode on log-softmax(logits) of each
    %             utterance's valid frames 1..T_n (Graves 2012 / Hannun et al. 2014
    %             prefix beam search, path masses of a labelling summed). Score = beam
    %             ranking score log p_ctc(l|x) + LMWeight log p_lm(l) + Bonus |l|;
    %             Candidates = the NBest best prefixes (Text, TokenIds, Score).
    %   Shallow fusion (off by default: LanguageModel = [] or LMWeight = 0): the
    %   LanguageModel must provide scorer() returning @(prefix, label) -> log p(label |
    %   prefix) over vocabulary ids (e.g. Decoding.TokenNGramLM fitted on TRAIN
    %   transcripts only).
    %
    %   Honesty (contract sec. 4): CTC decoding yields no word timing, so Segments is
    %   left empty; Language is left ""; Decoder ("ctc-greedy" | "ctc-beam") and every
    %   option are recorded in DecoderOptions.
    %   The vocabulary/model compatibility check (vocab.assertCompatible(K)) runs
    %   before any decoding.

    properties (SetAccess = immutable)
        Vocabulary
        Method (1,1) string
        BeamWidth (1,1) double
        Prune (1,1) double
        NBest (1,1) double
        LanguageModel
        LMWeight (1,1) double
        Bonus (1,1) double
        ModelId (1,1) string
    end

    methods
        function obj = CTCPrefixBeamDecoder(vocabulary, opts)
            arguments
                vocabulary (1,1) OceanNLP.Speech.Data.SpeechVocabulary
                opts.Method (1,1) string {mustBeMember(opts.Method, ["greedy", "beam"])} = "beam"
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
                opts.Prune (1,1) double = -inf
                opts.NBest double {mustBeScalarOrEmpty} = []
                opts.LanguageModel = []
                opts.LMWeight (1,1) double {mustBeFinite} = 0
                opts.Bonus (1,1) double {mustBeFinite} = 0
                opts.ModelId (1,1) string = ""
            end
            if ~isempty(opts.LanguageModel) && ~ismethod(opts.LanguageModel, 'scorer')
                error('OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:LanguageModel', ...
                    'LanguageModel must provide scorer() -> @(prefix, label) log-probability.');
            end
            if opts.Method == "greedy" && (opts.LMWeight ~= 0 || opts.Bonus ~= 0)
                error('OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:GreedyFusion', ...
                    'LM fusion / length bonus need Method "beam".');
            end
            if opts.LMWeight ~= 0 && isempty(opts.LanguageModel)
                error('OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:LanguageModel', ...
                    'LMWeight ~= 0 needs a LanguageModel.');
            end
            nb = opts.NBest;
            if isempty(nb), nb = opts.BeamWidth; end
            obj.Vocabulary = vocabulary;
            obj.Method = opts.Method;
            obj.BeamWidth = opts.BeamWidth;
            obj.Prune = opts.Prune;
            obj.NBest = min(nb, opts.BeamWidth);
            obj.LanguageModel = opts.LanguageModel;
            obj.LMWeight = opts.LMWeight;
            obj.Bonus = opts.Bonus;
            obj.ModelId = opts.ModelId;
        end

        function tr = decode(obj, logits, lengths)
            arguments
                obj
                logits
                lengths double = []
            end
            Z = double(gather(OceanDL.Autograd.Tensor.raw(logits)));
            [K, N, T] = size(Z, 1:3);
            obj.Vocabulary.assertCompatible(K);
            if isempty(lengths), lengths = T * ones(1, N); end
            lengths = reshape(lengths, 1, []);
            if numel(lengths) ~= N || any(lengths < 0 | lengths > T | lengths ~= round(lengths))
                error('OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:Lengths', ...
                    'lengths must be N = %d integers in [0, %d].', N, T);
            end
            % log-softmax over classes (stable)
            m = max(Z, [], 1);
            LP = Z - (m + log(sum(exp(Z - m), 1)));
            blank = obj.Vocabulary.Blank;
            dopts = obj.options();
            tr = OceanNLP.Speech.Core.Transcript.empty(1, 0);
            if obj.Method == "greedy"
                seqs = OceanDL.Recurrent.CTCDecoder.greedy(Z, blank, lengths);
                for n = 1:N
                    lp = reshape(LP(:, n, 1:lengths(n)), K, lengths(n));
                    score = sum(max(lp, [], 1));
                    ids = seqs{n};
                    txt = obj.Vocabulary.decode(ids);
                    cand = struct('Text', txt, 'TokenIds', ids, 'Score', score);
                    tr(n) = OceanNLP.Speech.Core.Transcript(txt, 'TokenIds', ids, 'Decoder', "ctc-greedy", ...
                        'DecoderOptions', dopts, 'ModelId', obj.ModelId, 'Score', score, 'Candidates', cand);
                end
                return;
            end
            args = {'Blank', blank, 'BeamWidth', obj.BeamWidth, 'Prune', obj.Prune};
            if ~isempty(obj.LanguageModel) && obj.LMWeight ~= 0
                args = [args, {'PrefixScorer', obj.LanguageModel.scorer(), 'Weight', obj.LMWeight}];
            end
            if obj.Bonus ~= 0
                args = [args, {'Bonus', obj.Bonus}];
            end
            for n = 1:N
                lp = reshape(LP(:, n, 1:lengths(n)), K, lengths(n));
                hyps = OceanDL.Recurrent.CTCBeamSearch.decode(lp, args{:});
                hyps = hyps(1:min(obj.NBest, numel(hyps)));
                cand = struct('Text', cell(1, numel(hyps)), 'TokenIds', cell(1, numel(hyps)), 'Score', cell(1, numel(hyps)));
                for h = 1:numel(hyps)
                    cand(h).Text = obj.Vocabulary.decode(hyps(h).Labels);
                    cand(h).TokenIds = hyps(h).Labels;
                    cand(h).Score = hyps(h).Score;
                end
                tr(n) = OceanNLP.Speech.Core.Transcript(cand(1).Text, 'TokenIds', cand(1).TokenIds, ...
                    'Decoder', "ctc-beam", 'DecoderOptions', dopts, 'ModelId', obj.ModelId, ...
                    'Score', cand(1).Score, 'Candidates', cand);
            end
        end

        function o = options(obj)
            % OPTIONS  Everything that determines the decoder output (recorded in Transcript).
            lm = struct();
            if ~isempty(obj.LanguageModel) && ismethod(obj.LanguageModel, 'config')
                lm = obj.LanguageModel.config();
            end
            o = struct('Method', obj.Method, 'Blank', obj.Vocabulary.Blank, 'BeamWidth', obj.BeamWidth, ...
                'Prune', obj.Prune, 'NBest', obj.NBest, 'LMWeight', obj.LMWeight, 'Bonus', obj.Bonus, ...
                'LanguageModel', lm);
            if obj.Method == "greedy"
                o = struct('Method', obj.Method, 'Blank', obj.Vocabulary.Blank);
            end
        end
    end
end
