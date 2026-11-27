classdef GradMode
    % GRADMODE  Global switch for graph recording (PyTorch torch.no_grad).
    %
    %   With grad mode OFF, ops still compute values but build no graph and
    %   save no buffers -- the right mode for inference and evaluation.
    %
    % USAGE:
    %   guard = OceanDL.Autograd.GradMode.disable();   % off until guard is cleared
    %   y = model.forward(x);                                      % no graph
    %   clear guard                                                 % previous mode restored
    %
    %   y = OceanDL.Autograd.GradMode.noGrad(@() model.forward(x));

    methods (Static)
        function tf = isEnabled()
            tf = OceanDL.Autograd.GradMode.state();
        end

        function guard = disable()
            % DISABLE  Turn recording off; returns an onCleanup guard that
            %   restores the previous mode when it goes out of scope.
            previous = OceanDL.Autograd.GradMode.state();
            OceanDL.Autograd.GradMode.state(false);
            guard = onCleanup(@() OceanDL.Autograd.GradMode.state(previous));
        end

        function varargout = noGrad(fn)
            % NOGRAD  Evaluate fn() with recording off.
            guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            [varargout{1:nargout}] = fn();
        end

        function tf = state(newValue)
            % STATE  Read (and optionally set) the process-wide flag.
            persistent enabled
            if isempty(enabled)
                enabled = true;
            end
            if nargin > 0
                enabled = logical(newValue);
            end
            tf = enabled;
        end
    end
end
