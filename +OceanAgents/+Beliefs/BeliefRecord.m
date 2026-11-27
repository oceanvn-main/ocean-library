classdef BeliefRecord
    % BELIEFRECORD  What a mind reads about one (subject, context, quantity): the estimate, how uncertain it is, how
    % much evidence supports it and where it came from. A sourced estimate, never "the truth" about the subject.
    %
    %   Estimate      probability in [0,1] (after any declared conversion of an imported score)
    %   Variance      posterior variance for Beta beliefs; [] for fixed scores (undefined, not zero)
    %   Alpha, Beta   Beta parameters; [] for fixed scores
    %   EvidenceMass  number of applied eligible events (0 for fixed scores)
    %   SourceKind    "import" | "prior" | "direct"
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md section 3.1.

    properties (SetAccess = immutable)
        SubjectID string = ""
        Context string = ""
        Quantity string = "completion"
        Estimate double = NaN
        Variance double = []
        Alpha double = []
        Beta double = []
        EvidenceMass double = 0
        SourceKind string = "prior"
        SourceID string = ""
        EvidenceIDs string = strings(1, 0)
        RawScore double = []
        ScaleMin double = []
        ScaleMax double = []
        Conversion string = "none"
        UpdateRule string = "fixed"
        Version double = 0
    end

    properties (Constant)
        SourceKinds = ["import" "prior" "direct"]
        UpdateRules = ["fixed" "beta-bernoulli"]
    end

    methods
        function obj = BeliefRecord(options)
            arguments
                options.SubjectID (1,1) string
                options.Context (1,1) string
                options.Quantity (1,1) string = "completion"
                options.Estimate (1,1) double
                options.Variance double = []
                options.Alpha double = []
                options.Beta double = []
                options.EvidenceMass (1,1) double = 0
                options.SourceKind (1,1) string
                options.SourceID (1,1) string = ""
                options.EvidenceIDs string = strings(1, 0)
                options.RawScore double = []
                options.ScaleMin double = []
                options.ScaleMax double = []
                options.Conversion (1,1) string = "none"
                options.UpdateRule (1,1) string
                options.Version (1,1) double = 0
            end
            bad = @(msg) error('OceanAgents:Core:InvalidRecord', 'BeliefRecord: %s', msg);
            need = {'SubjectID', 'Context', 'Estimate', 'SourceKind', 'UpdateRule'};
            if ~all(isfield(options, need)), bad(['missing required field(s): ' strjoin(need(~isfield(options, need)), ', ')]); end
            if ~isfinite(options.Estimate) || options.Estimate < 0 || options.Estimate > 1
                bad('Estimate must be a finite probability in [0, 1].');
            end
            if ~any(options.SourceKind == obj.SourceKinds), bad('unknown SourceKind.'); end
            if ~any(options.UpdateRule == obj.UpdateRules), bad('unknown UpdateRule.'); end
            if ~isempty(options.Variance) && (~isscalar(options.Variance) || ~isfinite(options.Variance) || options.Variance < 0)
                bad('Variance must be empty or a finite non-negative scalar.');
            end
            if xor(isempty(options.Alpha), isempty(options.Beta)) || ...
                    (~isempty(options.Alpha) && ~(isscalar(options.Alpha) && isscalar(options.Beta) && ...
                    options.Alpha > 0 && options.Beta > 0 && isfinite(options.Alpha) && isfinite(options.Beta)))
                bad('Alpha and Beta must both be empty or both positive finite scalars.');
            end
            if options.EvidenceMass < 0 || options.EvidenceMass ~= fix(options.EvidenceMass)
                bad('EvidenceMass must be a non-negative integer.');
            end
            f = fieldnames(options);
            for k = 1:numel(f)
                obj.(f{k}) = options.(f{k});
            end
            obj.EvidenceIDs = reshape(options.EvidenceIDs, 1, []);
        end

        function sd = stdDev(obj)
            % STDDEV  sqrt(Variance), or NaN when the variance is undefined (fixed scores).
            if isempty(obj.Variance), sd = NaN; else, sd = sqrt(obj.Variance); end
        end
    end
end
