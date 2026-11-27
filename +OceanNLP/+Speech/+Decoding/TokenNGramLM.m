classdef TokenNGramLM < handle
    % TOKENNGRAMLM  Additively smoothed token n-gram LM for CTC shallow fusion (off by default).
    %
    %   lm = OceanNLP.Speech.Decoding.TokenNGramLM.fit(trainTokenIds, vocab, 'Order', 3, 'Alpha', 0.1)
    %   s  = lm.scorer();          @(prefix, label) -> log p(label | last Order-1 labels)
    %   lp = lm.logProb(context, label);   cfg = lm.config()
    %
    %   FIT ON TRAINING TRANSCRIPTS ONLY (token-id rows from vocab.encode of the train
    %   split). Each sequence is padded on the left with Order-1 copies of <bos>
    %   (vocab.Bos), so the first labels are conditioned on the sentence start.
    %   ESTIMATE (Lidstone / add-alpha smoothing, Jurafsky & Martin ch. 3.5):
    %     p(w | h) = (c(h, w) + alpha) / (c(h) + alpha V)
    %   h = the Order-1 preceding tokens, c(.) training counts, V = number of
    %   non-special units (vocabulary ids 6..K) -- the labels a CTC decoder can emit.
    %   No end-of-sentence term is scored (prefix beam search scores prefixes only).
    %   This is a deliberately simple baseline LM; it is not a claim of the best LM.

    properties (SetAccess = private)
        Order (1,1) double
        Alpha (1,1) double
        Bos (1,1) double
        NumUnits (1,1) double
        NumClasses (1,1) double
        Counts             % containers.Map  key(h, w) -> count
        ContextCounts      % containers.Map  key(h)    -> count
        NumSequences (1,1) double = 0
    end

    methods (Static)
        function lm = fit(sequences, vocabulary, opts)
            arguments
                sequences cell
                vocabulary (1,1) OceanNLP.Speech.Data.SpeechVocabulary
                opts.Order (1,1) double {mustBeInteger, mustBePositive} = 3
                opts.Alpha (1,1) double {mustBePositive} = 0.1
            end
            lm = OceanNLP.Speech.Decoding.TokenNGramLM();
            lm.Order = opts.Order;
            lm.Alpha = opts.Alpha;
            lm.Bos = vocabulary.Bos;
            lm.NumClasses = vocabulary.numClasses();
            lm.NumUnits = vocabulary.numClasses() - 5;
            lm.Counts = containers.Map('KeyType', 'char', 'ValueType', 'double');
            lm.ContextCounts = containers.Map('KeyType', 'char', 'ValueType', 'double');
            n1 = opts.Order - 1;
            for i = 1:numel(sequences)
                s = double(sequences{i}(:).');
                if any(s < 6 | s > lm.NumClasses)
                    error('OceanNLP:Speech:Decoding:TokenNGramLM:Ids', ...
                        'Sequence %d holds special or out-of-range ids.', i);
                end
                padded = [repmat(lm.Bos, 1, n1), s];
                for t = n1 + 1:numel(padded)
                    h = padded(t - n1:t - 1);
                    lm.bump(lm.ContextCounts, lm.key(h));
                    lm.bump(lm.Counts, lm.key([h, padded(t)]));
                end
            end
            lm.NumSequences = numel(sequences);
        end
    end

    methods
        function lp = logProb(obj, context, label)
            % LOGPROB  log p(label | context), context = preceding labels (any length).
            n1 = obj.Order - 1;
            context = double(context(:).');
            ctx = [repmat(obj.Bos, 1, max(0, n1 - numel(context))), context];
            h = ctx(end - n1 + 1:end);
            if n1 == 0, h = zeros(1, 0); end
            c = obj.get(obj.Counts, obj.key([h, label]));
            ch = obj.get(obj.ContextCounts, obj.key(h));
            lp = log((c + obj.Alpha) / (ch + obj.Alpha * obj.NumUnits));
        end

        function f = scorer(obj)
            f = @(prefix, label) obj.logProb(prefix, label);
        end

        function cfg = config(obj)
            cfg = struct('Type', "TokenNGramLM", 'Order', obj.Order, 'Alpha', obj.Alpha, ...
                'NumUnits', obj.NumUnits, 'NumSequences', obj.NumSequences, ...
                'Smoothing', "add-alpha (Lidstone)");
        end
    end

    methods (Access = private)
        function obj = TokenNGramLM()
        end
    end

    methods (Static, Access = private)
        function k = key(ids)
            k = ['k', sprintf('%d,', ids)];
        end

        function bump(map, k)
            if isKey(map, k)
                map(k) = map(k) + 1;
            else
                map(k) = 1;
            end
        end

        function v = get(map, k)
            if isKey(map, k), v = map(k); else, v = 0; end
        end
    end
end
