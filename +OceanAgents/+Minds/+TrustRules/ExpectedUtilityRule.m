classdef ExpectedUtilityRule < OceanAgents.Minds.TrustRules.ITrustRule
    % EXPECTEDUTILITYRULE  Delegate to the provider with the highest (optionally risk-adjusted) expected utility,
    % or take the outside option when that is better. Deterministic: no random draw, ties to the lowest label.
    %
    %   rule = OceanAgents.Minds.TrustRules.ExpectedUtilityRule('RiskAversion', 0);
    %   [label, info] = rule.choose(candidates, stream);    % stream is accepted but never used
    %
    % For a delegate candidate j with estimate p_j and variance Var_j, payoffs [S F O]:
    %   p~_j = clamp(p_j - lambda * sqrt(Var_j), 0, 1)       (lambda = RiskAversion >= 0)
    %   U_j  = p~_j * S + (1 - p~_j) * F
    % The outside option scores O. The chosen label maximises U; exact ties go to the lowest label.
    % info.Scores = U over the candidates; info.Probabilities is one-hot on the chosen candidate.
    % lambda > 0 needs a variance: a candidate with NaN variance (fixed score) -> OceanAgents:Minds:Incompatible.
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md sections 4.1, 7 and H7-H9.

    properties (SetAccess = immutable)
        RiskAversion double = 0     % lambda: how many posterior standard deviations to subtract from the estimate
    end

    methods
        function obj = ExpectedUtilityRule(options)
            arguments
                options.RiskAversion (1,1) double {mustBeFinite, mustBeNonnegative} = 0
            end
            obj.RiskAversion = options.RiskAversion;
        end

        function d = describe(obj)
            % DESCRIBE  Name, Parameters, UsesRandomness; Requires lists what the beliefs must provide.
            requires = strings(1, 0);
            if obj.RiskAversion > 0, requires = "variance"; end
            d = struct('Name', "ExpectedUtilityRule", 'Parameters', struct('RiskAversion', obj.RiskAversion), ...
                'UsesRandomness', false, 'Requires', requires);
        end

        function [label, info] = choose(obj, candidates, ~)
            % CHOOSE  Maximum (risk-adjusted) expected utility; the stream is never touched.
            checkCandidates_(candidates);
            S = candidates.Payoff(1);
            F = candidates.Payoff(2);
            O = candidates.Payoff(3);
            n = numel(candidates.Labels);
            scores = zeros(1, n);
            for j = 1:n
                if candidates.IsOutside(j)
                    scores(j) = O;
                    continue
                end
                p = candidates.Estimate(j);
                if obj.RiskAversion > 0
                    if isnan(candidates.Variance(j))
                        error('OceanAgents:Minds:Incompatible', ['RiskAversion %g needs a posterior variance, but ' ...
                            'subject %s has none (fixed score).'], obj.RiskAversion, candidates.Subjects(j));
                    end
                    p = p - obj.RiskAversion * sqrt(candidates.Variance(j));   % p~ = p - lambda * sigma
                end
                p = min(max(p, 0), 1);                                         % clamp to [0, 1]
                % U = p~ S + (1 - p~) F, evaluated as F + p~ (S - F): the same number algebraically, and the
                % form with one rounding step (0.8*10 + 0.2*(-20) gives 4.000000000000001; -20 + 0.8*30 gives 4)
                scores(j) = F + p * (S - F);
            end
            k = argmaxLowestLabel_(scores, candidates.Labels);
            label = candidates.Labels(k);
            probabilities = zeros(1, n);
            probabilities(k) = 1;
            info = struct('Scores', scores, 'Probabilities', probabilities);
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "ExpectedUtilityRule";
            spec.Inputs = struct('Name', "candidates", 'Domain', "Agent-TrustCandidates", 'Dtype', "struct", 'Shape', "1 x C");
            spec.Outputs = struct('Name', "label", 'Domain', "Agent-ActionLabel", 'Dtype', "double", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', sprintf('EU(lambda=%g)', obj.RiskAversion));
            spec.ComposesFrom = {};
        end
    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Pillar 1 from meta/ExpectedUtilityRule.meta.json (A2 contract section 7).
            here = fileparts(mfilename('fullpath'));
            meta = jsondecode(fileread(fullfile(here, 'meta', 'ExpectedUtilityRule.meta.json')));
            meta.SymbolicLatex = string(meta.equations.risk_adjusted_utility.latex);
            meta.Assumptions = meta.pedagogy.assumptions;
            meta.IsSymbolic = true;
            meta.Reference = string(meta.textbook.reference);
        end
    end
end
