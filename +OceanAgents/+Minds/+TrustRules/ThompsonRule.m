classdef ThompsonRule < OceanAgents.Minds.TrustRules.ITrustRule
    % THOMPSONRULE  Thompson sampling over Beta trust beliefs: draw one plausible success probability per provider
    % from its posterior, then act as if the draws were true. Uncertain providers are sometimes tried because their
    % draws are sometimes high, which is how the agent keeps learning about them.
    %
    %   rule = OceanAgents.Minds.TrustRules.ThompsonRule();
    %   [label, info] = rule.choose(candidates, stream);
    %
    % For each delegate candidate, in increasing label order:  x ~ Gamma(alpha), y ~ Gamma(beta), p = x/(x+y)
    % (so p ~ Beta(alpha, beta)), and U = p S + (1 - p) F. The outside option scores O. Maximum U wins, exact ties
    % to the lowest label. Gamma variates use Marsaglia-Tsang (2000) with randn(stream)/rand(stream) only (MATLAB's
    % randg has no stream argument; A2 contract section 10). The outside option draws nothing.
    % info.Scores = [] and info.Probabilities = []: the choice probability has no closed form, so no likelihood is
    % invented. The sampled utilities of the last call are kept in LastSampledScores for inspection.
    % Non-Beta beliefs (NaN Alpha/Beta) -> OceanAgents:Minds:Incompatible.
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md sections 4.1, 7 and H12.

    properties (SetAccess = private)
        LastSampledScores double = zeros(1, 0)   % U of every candidate in the last choose() (NaN never occurs)
    end

    methods
        function d = describe(~)
            d = struct('Name', "ThompsonRule", 'Parameters', struct(), 'UsesRandomness', true, ...
                'Requires', "beta-parameters");
        end

        function [label, info] = choose(obj, candidates, stream)
            % CHOOSE  One posterior draw per delegate candidate (label order), then the maximum sampled utility.
            arguments
                obj
                candidates
                stream (1,1) RandStream
            end
            checkCandidates_(candidates);
            S = candidates.Payoff(1);
            F = candidates.Payoff(2);
            O = candidates.Payoff(3);
            n = numel(candidates.Labels);
            delegates = reshape(find(~candidates.IsOutside), 1, []);
            bad = delegates(isnan(candidates.Alpha(delegates)) | isnan(candidates.Beta(delegates)));
            if ~isempty(bad)
                error('OceanAgents:Minds:Incompatible', 'ThompsonRule needs Beta beliefs; subject %s has none.', ...
                    candidates.Subjects(bad(1)));
            end
            scores = zeros(1, n);
            scores(candidates.IsOutside) = O;
            [~, order] = sort(candidates.Labels(delegates));
            for j = delegates(order)
                p = sampleBeta_(candidates.Alpha(j), candidates.Beta(j), stream);   % p ~ Beta(alpha, beta)
                scores(j) = F + p * (S - F);                % U = p S + (1 - p) F (same form as ExpectedUtilityRule)
            end
            k = argmaxLowestLabel_(scores, candidates.Labels);
            label = candidates.Labels(k);
            obj.LastSampledScores = scores;
            info = struct('Scores', [], 'Probabilities', []);
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "ThompsonRule";
            spec.Inputs = struct('Name', "candidates", 'Domain', "Agent-TrustCandidates", 'Dtype', "struct", 'Shape', "1 x C");
            spec.Outputs = struct('Name', "label", 'Domain', "Agent-ActionLabel", 'Dtype', "double", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "thompson");
            spec.ComposesFrom = {};
        end
    end

    methods (Static)
        function p = sampleBeta(alpha, beta, stream)
            % SAMPLEBETA  One Beta(alpha, beta) draw from stream (the exact sampler choose() uses; exposed so its
            % distribution and its stream consumption can be tested).
            arguments
                alpha (1,1) double {mustBePositive, mustBeFinite}
                beta (1,1) double {mustBePositive, mustBeFinite}
                stream (1,1) RandStream
            end
            p = sampleBeta_(alpha, beta, stream);
        end

        function meta = getMetadata()
            % GETMETADATA  Pillar 1 from meta/ThompsonRule.meta.json (A2 contract section 7).
            here = fileparts(mfilename('fullpath'));
            meta = jsondecode(fileread(fullfile(here, 'meta', 'ThompsonRule.meta.json')));
            meta.SymbolicLatex = string(meta.equations.thompson.latex);
            meta.Assumptions = meta.pedagogy.assumptions;
            meta.IsSymbolic = true;
            meta.Reference = string(meta.textbook.reference);
        end
    end
end
