classdef TfidfWeighting < OceanStats.Data.Weighting.IWeighting
    % TFIDFWEIGHTING  Smoothed TF-IDF term reweighting (sklearn-standard formula).
    %
    % THEORY:
    %   idf_t = log((1 + N) / (1 + df_t)) + 1     (smooth-idf, avoids /0, matches
    %                                               the scikit-learn / Lucene default —
    %                                               this is the DEFAULT IdfScheme;
    %                                               see below)
    %   tf    = counts, or log1p(counts) if SublinearTf=true
    %   W     = tf .* idf                          (broadcast over terms/columns)
    %
    %   Where:
    %     N = Number of documents (rows in counts matrix)
    %     V = Vocabulary size / Number of terms (columns in counts matrix)
    %     tf is an (N x V) matrix.
    %     idf is a (1 x V) row vector.
    %
    %   rows of W are L2-normalized by default via OceanStats.Data.Norm.L2Norm
    %
    %   STRATEGY PATTERNS:
    %   - IDF SCHEME (+Idf/IIdfScheme.m): defaults to SmoothIdf (shown above).
    %   - NORM SCHEME (+Norm/INormScheme.m): defaults to L2Norm.
    %
    %   Pass different schemes via the static create() method to swap formulas
    %   without touching this class's source:
    %     tfidf = OceanStats.Data.Weighting.TfidfWeighting.create('IdfScheme', 'classic', 'NormScheme', 'l1');
    %
    % USAGE:
    %   tfidf = OceanStats.Data.Weighting.TfidfWeighting.create('IdfScheme', 'smooth', 'NormScheme', 'l2');
    %   tfidf.fit(counts_train);
    %   Xtr = tfidf.forward(counts_train);
    %   Xte = tfidf.forward(counts_test);
    %
    % SPARSE-SAFETY NOTE (verified empirically, R2026a, per master plan §3.2):
    %   The plan flagged `tf .* obj.IDF` (sparse (N x V) times dense (1 x V) row
    %   vector) as unverified for sparsity preservation. Empirical check on
    %   R2026a with a >95%-sparse toy matrix confirmed issparse() stays true and
    %   nnz() is unchanged through the tf-scaling, the sublinear-tf spfun step,
    %   AND the row-L2-normalization division — no densification anywhere in
    %   this pipeline on this MATLAB version. No triplet-pattern rewrite was
    %   required. Re-verify with issparse() if this is ported to an older
    %   MATLAB release, per §3.2 item 1 (never assume).
    %
    % See also: OceanStats.Data.Weighting.IWeighting
    %           OceanStats.Data.Weighting.BM25Weighting
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to fit/forward.

    properties
        SublinearTf (1,1) logical = false
        NormScheme  (1,1) OceanStats.Data.Norm.INormScheme = OceanStats.Data.Norm.L2Norm() % Strategy for row-wise normalization
        IdfScheme   (1,1) OceanStats.Data.Weighting.Idf.IIdfScheme = OceanStats.Data.Weighting.Idf.SmoothIdf()   % Strategy — overridable via constructor
    end

    properties (SetAccess = private)
        IDF       (1,:) double = []    % (1 x V) fitted inverse document frequency
        NumDocs   (1,1) double = 0
        VocabSize (1,1) double = 0
    end

    methods (Static)
        function obj = create(options)
            % CREATE  Helper to instantiate TfidfWeighting using strings for strategies.
            %
            % USAGE:
            %   tfidf = OceanStats.Data.Weighting.TfidfWeighting.create('IdfScheme', 'classic', 'NormScheme', 'l1');
            arguments
                options.SublinearTf (1,1) logical = false
                options.NormScheme  (1,1) string  = "l2"
                options.IdfScheme   (1,1) string  = "smooth"
            end
            
            idfObj = OceanStats.Data.Weighting.Idf.IdfSchemeFactory.create(options.IdfScheme);
            normObj = OceanStats.Data.Norm.NormSchemeFactory.create(options.NormScheme);
            
            obj = OceanStats.Data.Weighting.TfidfWeighting( ...
                'SublinearTf', options.SublinearTf, ...
                'NormScheme', normObj, ...
                'IdfScheme', idfObj);
        end
    end

    methods
        function obj = TfidfWeighting(options)
            arguments
                options.SublinearTf (1,1) logical = false
                % Strategy Pattern: Row-wise normalization decoupled via INormScheme
                options.NormScheme  (1,1) OceanStats.Data.Norm.INormScheme = OceanStats.Data.Norm.L2Norm()
                % Strategy Pattern (see +Idf/IIdfScheme.m): the idf formula
                % itself is decoupled from this class. Default preserves the
                % exact previously-validated smooth-idf behavior; pass a
                % different OceanStats.Data.Weighting.Idf.IIdfScheme (e.g.
                % via IdfSchemeFactory.create('classic')) to swap formulas
                % WITHOUT modifying this class's source.
                options.IdfScheme (1,1) OceanStats.Data.Weighting.Idf.IIdfScheme = OceanStats.Data.Weighting.Idf.SmoothIdf()
            end
            obj.SublinearTf = options.SublinearTf;
            obj.NormScheme  = options.NormScheme;
            obj.IdfScheme   = options.IdfScheme;
        end

        function fit(obj, counts)
            %FIT compute inverse document frequency for each term in the given counts matrix.
            % FIT  Learn per-term smoothed IDF from a (N x V) count matrix.
            %   WARNING: this writes directly onto obj's own buffers (IDF,
            %   NumDocs, VocabSize) — it does not return a copy. Since obj is
            %   a handle, every reference to it (e.g. a TfidfVectorizer that
            %   holds this object in its own Weighting property) sees the
            %   new values immediately. Calling fit() a second time overwrites
            %   whatever was previously learned; if that happens on anything
            %   other than the original training corpus, it silently defeats
            %   the anti-leakage guarantee ("fit on train only, forward() on
            %   train and test") this class's design otherwise gives you.
            arguments
                obj
                counts (:,:) double   % accepts sparse or dense
                % counts matrix, each row is a document, each column is a term
            end
            if obj.isFitted()
                warning('OceanStats:Data:Weighting:TfidfWeighting:RefitOverwrite', ...
                    ['fit() called on an already-fitted TfidfWeighting instance. ' ...
                     'This overwrites the previously learned IDF in place — any ' ...
                     'other reference to this same object (e.g. a TfidfVectorizer ' ...
                     'holding it) will immediately see the new values. If the ' ...
                     'original fit was on training data, re-fitting on anything ' ...
                     'else (e.g. test data) introduces data leakage. Construct a ' ...
                     'new %s instance instead if you need independent fitted state.'], ...
                    class(obj));
            end
            df      = sum(counts > 0, 1);
            N       = size(counts, 1);
            obj.IDF = obj.IdfScheme.compute(df, N);
            obj.NumDocs   = N;
            obj.VocabSize = size(counts, 2);
        end

        function W = forward(obj, counts)
            % FORWARD  Apply fitted IDF (and optional sublinear-tf, row-norm).
            arguments
                obj
                counts (:,:) double
            end
            if ~obj.isFitted()
                error('OceanStats:Data:Weighting:TfidfWeighting:NotFitted', ...
                    'Call fit(counts) before forward().');
            end
            if size(counts, 2) ~= obj.VocabSize
                error('OceanStats:Data:Weighting:TfidfWeighting:VocabMismatch', ...
                    'forward() got %d columns, fit() saw %d.', size(counts,2), obj.VocabSize);
            end

            %TF Saturation (Dampening) 0 > 1 good. 99 -> 100. Not much
            if obj.SublinearTf 
                if issparse(counts)
                    %Only apply operation to the non-zero cells, leave other untouched
                    tf = spfun(@log1p, counts);   % log1p(0)=0 — safe on nonzeros only
                else
                    tf = log1p(counts);
                end
            else
                tf = counts;
            end

            % Verified sparse-safe empirically on R2026a — see class help.
            W = tf .* obj.IDF;   % broadcast (N x V) .* (1 x V)
            % Element-wise N x V .* N x V

            W = obj.NormScheme.normalize(W);
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.IDF);
        end

        function s = stateDict(obj)
            % STATEDICT  Snapshot fitted buffers + config for save/reload.
            %   IdfScheme is stored by NAME (via IdfSchemeFactory), not as a
            %   serialized object — loadStateDict() reconstructs it through
            %   the factory, so this round-trips correctly for any of the
            %   three built-in schemes.
            s = struct('IDF', obj.IDF, 'NumDocs', obj.NumDocs, ...
                        'VocabSize', obj.VocabSize, ...
                        'SublinearTf', obj.SublinearTf, ...
                        'NormSchemeName', obj.NormScheme.name(), ...
                        'IdfSchemeName', obj.IdfScheme.name());
        end

        function loadStateDict(obj, s)
            % LOADSTATEDICT  Restore buffers + config from stateDict() output.
            arguments
                obj
                s (1,1) struct
            end
            obj.IDF         = s.IDF;
            obj.NumDocs     = s.NumDocs;
            obj.VocabSize   = s.VocabSize;
            obj.SublinearTf = s.SublinearTf;
            if isfield(s, 'NormSchemeName')
                obj.NormScheme = OceanStats.Data.Norm.NormSchemeFactory.create(s.NormSchemeName);
            elseif isfield(s, 'Norm')
                % Backwards compatibility for older saved states
                obj.NormScheme = OceanStats.Data.Norm.NormSchemeFactory.create(s.Norm);
            end
            if isfield(s, 'IdfSchemeName')
                obj.IdfScheme = OceanStats.Data.Weighting.Idf.IdfSchemeFactory.create(s.IdfSchemeName);
            end
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1 (symbolic) — closed-form LaTeX, symbol
            % dictionary, assumptions, and reference backing fit/forward.
            % The L2-row-normalization line is appended only when
            % obj.Norm=="l2" (the default) — it is a real behavioral fact
            % about the formula actually applied, not an always-on note.
            meta = struct();
            % idf line comes from the plugged-in IdfScheme, not a hardcoded
            % formula — keeps this honest (R7) no matter which scheme is
            % active, instead of always claiming the smooth-idf formula.
            normLatex = obj.NormScheme.symbolicLatex();
            lines = [obj.IdfScheme.symbolicLatex(), ...
                     "w_{d,t} = \mathrm{tf}_{d,t} \cdot \mathrm{idf}_t"];
            if normLatex ~= "\hat{w}_{d,t} = w_{d,t}"
                lines = [lines, normLatex];
            end
            meta.SymbolicLatex = lines;
            meta.Symbols = containers.Map();
            meta.Symbols('N') = "total number of documents (fit-time)";
            meta.Symbols('df_t') = "number of documents containing term t";
            meta.Symbols('tf_{d,t}') = "raw (or log1p-scaled, if SublinearTf) count of term t in document d";
            meta.Symbols('idf_t') = "inverse document frequency of term t (formula set by IdfScheme)";
            meta.Symbols('w_{d,t}') = "TF-IDF weight";
            if normLatex ~= "\hat{w}_{d,t} = w_{d,t}"
                meta.Symbols('\hat{w}_{d,t}') = "row-normalized TF-IDF weight";
            end
            meta.Assumptions = {'counts nonnegative', ...
                                 'idf fit only on training corpus (anti-leakage)', ...
                                 'SublinearTf is configuration, not part of the closed form above', ...
                                 sprintf('idf formula is the "%s" IdfScheme strategy — swap it via the constructor', obj.IdfScheme.name()), ...
                                 sprintf('normalization is the "%s" NormScheme strategy', obj.NormScheme.name())};
            meta.IsSymbolic = true;
            meta.Reference = sprintf("idf: %s scheme (%s); weighting: TF x idf, scikit-learn TfidfTransformer convention", ...
                obj.IdfScheme.name(), class(obj.IdfScheme));
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator.
            spec = struct();
            spec.Inputs = struct('Name', "counts", 'Domain', "count-matrix", 'Dtype', "double", 'Shape', "N x V");
            spec.Outputs = struct('Name', "W", 'Domain', "weighted-matrix", 'Dtype', "double", 'Shape', "N x V");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "tf-idf");
            spec.ComposesFrom = {};
        end
    end
end
