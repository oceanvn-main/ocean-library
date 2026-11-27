classdef (Abstract) IOptimizerStrategy < OceanStats.Optim.IOptimizer
    % IOPTIMIZERSTRATEGY  Abstract base class for 3-Pillar Tensor Optimizers.
    %
    %   Implements the mandatory 3-Pillar Component Contract O = (phi, f, D)
    %   from pipeline-architecture-spec.md:
    %     Pillar 1 (phi) : Closed-form LaTeX formulas, Symbol dict, Assumptions (getMetadata)
    %     Pillar 2 (f)   : High-performance numeric parameter update pass (step, get_state_dict)
    %     Pillar 3 (D)   : PortSpec signature, DiagramSymbol visual metadata (getPipelineSpec)
    %
    %   CONTRACT:
    %     [param_new, telemetry] = opt.step(param, grad)
    %     meta  = opt.getMetadata()
    %     spec  = opt.getPipelineSpec()
    %     state = opt.get_state_dict()
    %     opt.set_state_dict(state)
    %     opt.reset()
    %     s     = opt.describe()
    %
    %   See also: OceanStats.Optim.IOptimizer
    %             OceanStats.Optim.ITensorTarget
    %             OceanStats.Optim.VectorizedOptimizerEngine

    methods (Abstract)
        meta = getMetadata(obj)
        % GETMETADATA  Pillar 1: Returns SymbolicLatex, Symbols struct, and Assumptions.

        spec = getPipelineSpec(obj)
        % GETPIPELINESPEC  Pillar 3: Returns PortSpec input/output signatures and DiagramSymbol.

        state = get_state_dict(obj)
        % GET_STATE_DICT  Pillar 2: Export internal state buffers (StepCount, moments).

        set_state_dict(obj, state)
        % SET_STATE_DICT  Pillar 2: Restore internal state buffers from snapshot struct.
    end

    methods
        function validateGradients(~, grad)
            % VALIDATEGRADIENTS  Guard against NaN/Inf or empty gradients.
            if isempty(grad)
                error('OceanStats:Optim:EmptyGradient', ...
                    'Gradient array cannot be empty.');
            end
            if any(isnan(grad(:))) || any(isinf(grad(:)))
                error('OceanStats:Optim:NaNGradient', ...
                    'Gradient contains NaN or Inf values.');
            end
        end
    end
end
