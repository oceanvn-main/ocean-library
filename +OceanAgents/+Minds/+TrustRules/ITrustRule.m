classdef (Abstract) ITrustRule < handle
    % ITRUSTRULE  How a trust mind turns beliefs into one choice among "delegate to subject j" and the outside option.
    % The one axis that varies between trust minds' decision rules (Strategy pattern).
    %
    %   [label, info] = rule.choose(candidates, stream)
    %   candidates  struct with 1xC fields (legal actions only, in label order):
    %                 Labels, Subjects (string), IsOutside (logical), Estimate, Variance (NaN = undefined),
    %                 Alpha, Beta (NaN when not a Beta belief); and Payoff = [Success Failure Outside] (1x3)
    %   label       the chosen external action label (one of candidates.Labels)
    %   info        struct: Scores (1xC; [] if not closed-form), Probabilities (1xC over candidates; [] if not
    %               closed-form - a rule never invents a likelihood)
    % A rule may draw only from the given stream. See Planning/System/Game Theory/A2_Beliefs_Contract.md section 4.1.

    methods (Abstract)
        d = describe(obj)                 % struct: Name, Parameters, UsesRandomness (logical),
                                          %   Requires ("variance" | "beta-parameters" | "")
        [label, info] = choose(obj, candidates, stream)
    end
end
