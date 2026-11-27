classdef RuledSiLU < OceanDL.Autograd.Function
    % RULEDSILU  Y = A .* sigmoid(A) with a MODIFIED backward rule, used only
    %   for projecting hidden units back to pixels (see OceanDL.Autograd.BackpropRule):
    %     "deconvnet"  dL/dA = max(g, 0)                                  (Zeiler & Fergus 2014)
    %     "guided"     dL/dA = max(g, 0) .* silu'(A) .* (A > 0)           (Springenberg et al. 2015, SiLU form)
    %   silu'(A) = s + A s (1 - s),  s = sigmoid(A).
    %   The published rules are for ReLU; this SiLU form is our adaptation (not in the texts).

    properties (SetAccess = private)
        Rule (1,1) string
    end

    methods
        function obj = RuledSiLU(rule)
            obj.Rule = string(rule);
        end

        function Y = forward(obj, A)
            s = 1 ./ (1 + exp(-A));
            Y = A .* s;
            obj.Saved.A = A;
            obj.Saved.S = s;
        end

        function grads = backward(obj, g)
            gp = max(g, 0);
            switch obj.Rule
                case "deconvnet"
                    grads = {gp};
                case "guided"
                    A = obj.Saved.A; s = obj.Saved.S;
                    grads = {gp .* (s + A .* s .* (1 - s)) .* (A > 0)};
                otherwise
                    error('OceanDL:Autograd:RuledSiLU:Rule', 'Unknown backprop rule "%s".', obj.Rule);
            end
        end
    end
end
