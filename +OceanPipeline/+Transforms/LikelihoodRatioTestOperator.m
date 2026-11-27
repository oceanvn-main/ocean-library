classdef LikelihoodRatioTestOperator < OceanPipeline.Transforms.IDataTransform
    % LIKELIHOODRATIOTESTOPERATOR  Pipeline Operator for Likelihood Ratio Test.
    %
    % NAMESPACE: OceanPipeline.Transforms
    %
    % Implements the Decoupled Dual-Representation Pipeline Architecture Spec
    % (.agents/pipeline-architecture-spec.md).
    %
    % CONTRACT:
    %   Numeric Execution Layer (f: forward):
    %     Calculates test statistic 2*(L_unrestricted - L_restricted) and
    %     Chi-Square upper-tail p-value.
    %
    %   Mathematical Representation Layer (phi: expr_fn):
    %     Decoupled symbolic LaTeX derivation generator for KaTeX visualizers:
    %     L_LRT = 2 * [ L(theta_hat) - L(theta_tilde) ] ~ Chi2(m)
    %
    %   Honest Opacity (R7 Rule):
    %     If IsSymbolic = false, expr_fn() returns an un-symbolic tag.
    %
    % PROPERTIES:
    %   IsSymbolic (logical) — Toggle symbolic representation (default true)
    %   Alpha (double)      — Significance level threshold (default 0.05)
    %
    % USAGE:
    %   op = OceanPipeline.Transforms.LikelihoodRatioTestOperator('IsSymbolic', true);
    %   state = op.fit([0.0, -0.857, 1]); % [L_unrestricted, L_restricted, restrictions]
    %   out = op.apply([0.0, -0.857, 1], state);
    %   expr = op.expr_fn(state);
    %
    % Author: Ocean Library System
    % Date: 2026-09-09

    properties
        IsSymbolic (1,1) logical = true
        Alpha      (1,1) double  = 0.05
    end

    properties (SetAccess = private)
        TestEngine OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest
    end

    methods
        function obj = LikelihoodRatioTestOperator(opts)
            % LIKELIHOODRATIOTESTOPERATOR  Construct pipeline operator block.
            arguments
                opts.IsSymbolic (1,1) logical = true
                opts.Alpha      (1,1) double  = 0.05
            end
            obj.IsSymbolic = opts.IsSymbolic;
            obj.Alpha      = opts.Alpha;
            obj.TestEngine = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
        end

        % ---------------------------------------------------------------- %
        function prototype = getPrototype(obj)
            % GETPROTOTYPE  Return declarative prototype spec (PortSpec & metadata).
            prototype.id = "op_tsa_lrt";
            prototype.name = "Likelihood Ratio Test (LRT)";
            prototype.is_symbolic = obj.IsSymbolic;
            
            prototype.inputs = { ...
                struct('name', 'log_likelihood_unrestricted', 'domain', 'scalar', 'dtype', 'real'), ...
                struct('name', 'log_likelihood_restricted',   'domain', 'scalar', 'dtype', 'real'), ...
                struct('name', 'restrictions',                'domain', 'scalar', 'dtype', 'int')  ...
            };
            
            prototype.outputs = { ...
                struct('name', 'lrt_stat', 'domain', 'scalar', 'dtype', 'real'), ...
                struct('name', 'p_value',  'domain', 'scalar', 'dtype', 'real'), ...
                struct('name', 'reject',   'domain', 'scalar', 'dtype', 'logical') ...
            };
            
            prototype.assumptions = obj.TestEngine.assumptions();
        end

        % ---------------------------------------------------------------- %
        function state = fit(obj, data)
            % FIT  Learn/bind transform state from pipeline input matrix.
            %   data — (1 x 3) or (N x 3) matrix: [L_unrestricted, L_restricted, m]
            arguments
                obj
                data double
            end
            if size(data, 2) < 3
                error('LikelihoodRatioTestOperator:InvalidInput', ...
                    'Input data matrix must have at least 3 columns: [L_unrestricted, L_restricted, restrictions].');
            end

            lunrestricted = data(1, 1);
            lrestricted   = data(1, 2);
            m             = data(1, 3);

            testData = struct( ...
                'LogLikUnrestricted', lunrestricted, ...
                'LogLikRestricted', lrestricted, ...
                'Restrictions', m, ...
                'Alpha', obj.Alpha ...
            );

            result = obj.TestEngine.test(testData);

            state.L_unrestricted = lunrestricted;
            state.L_restricted   = lrestricted;
            state.Restrictions   = m;
            state.TestResult     = result;
            state.IsFitted       = true;
        end

        % ---------------------------------------------------------------- %
        function output = apply(obj, data, state)
            % APPLY  Run numeric execution (forward pass f).
            arguments
                obj
                data double
                state struct
            end
            
            % If new data is provided, evaluate dynamically; otherwise use state result
            if size(data, 1) > 0 && size(data, 2) >= 3
                lunrestricted = data(1, 1);
                lrestricted   = data(1, 2);
                m             = data(1, 3);

                testData = struct( ...
                    'LogLikUnrestricted', lunrestricted, ...
                    'LogLikRestricted', lrestricted, ...
                    'Restrictions', m, ...
                    'Alpha', obj.Alpha ...
                );

                res = obj.TestEngine.test(testData);
                output = [res.Statistic, res.PValue, double(res.Reject)];
            else
                res = state.TestResult;
                output = [res.Statistic, res.PValue, double(res.Reject)];
            end
        end

        % ---------------------------------------------------------------- %
        function expr = expr_fn(obj, state)
            % EXPR_FN  Mathematical Representation Layer (phi: symbolic expression).
            %   Returns struct containing symbolic KaTeX equations and derivation steps.
            arguments
                obj
                state struct = struct()
            end

            if ~obj.IsSymbolic
                % Honest Opacity (R7): Return non-symbolic tag
                expr.is_symbolic = false;
                expr.symbolic = "";
                expr.description = "Operator tagged non-symbolic (Honest Opacity R7)";
                return;
            end

            expr.is_symbolic = true;
            expr.symbolic = "2 \cdot \left[ \mathcal{L}(\hat{\theta}) - \mathcal{L}(\tilde{\theta}) \right] \sim \chi^2(m)";
            expr.null_hypothesis = "H_0: \mathbf{R} \boldsymbol{\theta} = \mathbf{r} \quad (m \text{ restrictions})";
            expr.alternative_hypothesis = "H_1: \boldsymbol{\theta} \in \mathbf{\Omega} \text{ (unrestricted)}";

            if isfield(state, 'TestResult') && ~isempty(state.TestResult)
                res = state.TestResult;
                expr.evaluated_formula = sprintf("2 \\cdot [%.4f - (%.4f)] = %.4f", ...
                    state.L_unrestricted, state.L_restricted, res.Statistic);
                expr.p_value_formula = sprintf("P(\\chi^2(%d) \\ge %.4f) = %.4f", ...
                    res.Df, res.Statistic, res.PValue);
                expr.decision = sprintf("Decision (alpha=%.2f): %s", ...
                    res.Alpha, ternary(res.Reject, "REJECT H0", "FAIL TO REJECT H0"));
            end
        end
    end
end

function out = ternary(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end
