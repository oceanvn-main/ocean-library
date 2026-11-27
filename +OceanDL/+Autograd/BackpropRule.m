classdef BackpropRule
    % BACKPROPRULE  Process-wide choice of the backward rule through activation
    %   functions, used to project a hidden unit's response back to pixels
    %   (Szeliski 2nd ed. Sec. 5.4, Fig. 5.47; Zeiler & Fergus 2014;
    %   Springenberg et al. 2015). Only the BACKWARD pass changes; forward
    %   values are identical. Same pattern as GradMode.
    %
    %   "standard"   the true gradient (saliency, Simonyan et al. 2014)
    %   "deconvnet"  Zeiler & Fergus: pass only POSITIVE gradients, ignore the
    %                forward activation:      dL/dx = max(g, 0)
    %   "guided"     guided backpropagation: positive gradients AND positive
    %                inputs:                  dL/dx = max(g, 0) .* f'(x) .* (x > 0)
    %   For ReLU, f'(x) = (x > 0) and these are the published rules. The rules
    %   are applied by F.silu (the activation of our conv VAEs); extending them
    %   to SiLU in this way is our adaptation -- not in the texts, which use ReLU.
    %
    %   OceanDL.Autograd.BackpropRule.run("guided", @() ...)   % rule active inside fn
    %   guard = OceanDL.Autograd.BackpropRule.set("deconvnet"); ... clear guard

    methods (Static)
        function r = current()
            r = OceanDL.Autograd.BackpropRule.state();
        end

        function guard = set(rule)
            arguments
                rule (1,1) string {mustBeMember(rule, ["standard", "deconvnet", "guided"])}
            end
            previous = OceanDL.Autograd.BackpropRule.state();
            OceanDL.Autograd.BackpropRule.state(rule);
            guard = onCleanup(@() OceanDL.Autograd.BackpropRule.state(previous));
        end

        function varargout = run(rule, fn)
            % RUN  Evaluate fn() (typically forward + backward) with the rule active.
            guard = OceanDL.Autograd.BackpropRule.set(rule); %#ok<NASGU>
            [varargout{1:nargout}] = fn();
        end

        function r = state(newValue)
            persistent rule
            if isempty(rule), rule = "standard"; end
            if nargin > 0, rule = string(newValue); end
            r = rule;
        end
    end
end
