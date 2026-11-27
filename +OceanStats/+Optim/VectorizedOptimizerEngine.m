classdef VectorizedOptimizerEngine < OceanStats.Optim.IOptimizerStrategy
    % VECTORIZEDOPTIMIZERENGINE  Unified 1D contiguous vector optimization engine.
    %
    %   Decouples optimization algorithms from domain models by executing
    %   high-performance 1D contiguous BLAS/GPU vector arithmetic across all
    %   trainable parameter arrays in a single step call.
    %
    %   Implements the mandatory 3-Pillar Contract O = (phi, f, D):
    %     Pillar 1 (phi) : Closed-form LaTeX formulas, Symbol legend, Assumptions
    %     Pillar 2 (f)   : Vectorized step execution & state_dict serialization
    %     Pillar 3 (D)   : PortSpec signature & OceanViz dark-theme canvas spec
    %
    %   USAGE (with ITensorTarget):
    %     engine = OceanStats.Optim.VectorizedOptimizerEngine(strategy);
    %     [target, telemetry] = engine.stepTarget(target, X_batch, y_batch);
    %
    %   USAGE (with 1D contiguous vectors):
    %     engine = OceanStats.Optim.VectorizedOptimizerEngine(strategy);
    %     [theta_new, telemetry] = engine.stepVector(theta_flat, grads_flat);
    %
    %   See also: OceanStats.Optim.IOptimizerStrategy
    %             OceanStats.Optim.ITensorTarget
    %             OceanStats.Optim.ParameterGroup

    properties
        Strategy   % Concrete algorithm strategy (AdamW, Nesterov, etc.)
        ParameterGroups  cell = {}                      % Cell array of ParameterGroup objects
        StepCount        (1,1) double = 0               % Iteration counter
        UseGPU           (1,1) logical = false          % GPU array acceleration flag
    end

    methods
        function obj = VectorizedOptimizerEngine(strategy, opts)
            % VECTORIZEDOPTIMIZERENGINE  Constructor.
            %
            %   engine = VectorizedOptimizerEngine(strategy)
            %   engine = VectorizedOptimizerEngine(strategy, 'ParameterGroups', groups)
            arguments
                strategy               OceanStats.Optim.IOptimizer
                opts.ParameterGroups   cell = {}
                opts.UseGPU            (1,1) logical = false
            end
            obj.Strategy        = strategy;
            obj.ParameterGroups = opts.ParameterGroups;
            obj.UseGPU          = opts.UseGPU || (canUseGPU() && false);
        end

        function [target, telemetry] = stepTarget(obj, target, X, y)
            % STEPTARGET  Execute step directly on an ITensorTarget instance.
            arguments
                obj     OceanStats.Optim.VectorizedOptimizerEngine
                target  OceanStats.Optim.ITensorTarget
                X       double
                y       double
            end

            theta = target.getParameters();
            grads = target.computeGradients(X, y);

            [theta_new, telemetry] = obj.stepVector(theta, grads);
            target.setParameters(theta_new);
        end

        function [theta_new, telemetry] = stepVector(obj, theta_flat, grads_flat)
            % STEPVECTOR  Execute optimization step on 1D contiguous arrays.
            arguments
                obj         OceanStats.Optim.VectorizedOptimizerEngine
                theta_flat  (:,1) double
                grads_flat  (:,1) double
            end

            % Input validation
            obj.validateGradients(grads_flat);
            if numel(theta_flat) ~= numel(grads_flat)
                error('OceanStats:Optim:DimensionMismatch', ...
                    'Parameter size (%d) does not match gradient size (%d).', ...
                    numel(theta_flat), numel(grads_flat));
            end

            obj.StepCount = obj.StepCount + 1;
            tStart = tic;

            % GPU Transfer if enabled
            if obj.UseGPU && canUseGPU() && ~isa(theta_flat, 'gpuArray')
                theta_flat = gpuArray(theta_flat);
                grads_flat = gpuArray(grads_flat);
            end

            % Execute underlying strategy step
            if isempty(obj.ParameterGroups)
                % Single contiguous parameter vector execution
                theta_new = obj.Strategy.step(theta_flat, grads_flat);
            else
                % Parameter Group multi-tensor execution
                theta_new = theta_flat;
                for g = 1:numel(obj.ParameterGroups)
                    grp = obj.ParameterGroups{g};
                    idx = grp.ParamIndices;

                    % Temporarily set strategy learning rate for group if supported
                    if isprop(obj.Strategy, 'LR')
                        origLR = obj.Strategy.LR;
                        obj.Strategy.LR = grp.getEffectiveLR();
                    end

                    theta_new(idx) = obj.Strategy.step(theta_flat(idx), grads_flat(idx));

                    % Restore original strategy LR
                    if isprop(obj.Strategy, 'LR')
                        obj.Strategy.LR = origLR;
                    end
                end
            end

            % Gather telemetry
            execTime = toc(tStart);
            gradNorm = norm(double(grads_flat));
            paramNorm = norm(double(theta_new));

            telemetry = struct( ...
                'Step', obj.StepCount, ...
                'GradNorm', gradNorm, ...
                'ParamNorm', paramNorm, ...
                'ExecutionTime', execTime, ...
                'Strategy', obj.Strategy.describe());
        end

        function param_new = step(obj, param, grad)
            % STEP  IOptimizer legacy interface wrapper.
            [param_new, ~] = obj.stepVector(param(:), grad(:));
            param_new = reshape(param_new, size(param));
        end

        function state = get_state_dict(obj)
            % GET_STATE_DICT  Pillar 2: Export engine & strategy state snapshot.
            state = struct( ...
                'EngineClass', class(obj), ...
                'StepCount', obj.StepCount, ...
                'StrategyClass', class(obj.Strategy), ...
                'StrategyState', []);

            if isa(obj.Strategy, 'OceanStats.Optim.IOptimizerStrategy')
                state.StrategyState = obj.Strategy.get_state_dict();
            end
        end

        function set_state_dict(obj, state)
            % SET_STATE_DICT  Pillar 2: Restore state snapshot.
            if ~isstruct(state) || ~isfield(state, 'StepCount')
                error('OceanStats:Optim:InvalidStateDict', ...
                    'Invalid state_dict structure provided.');
            end
            obj.StepCount = state.StepCount;
            if isa(obj.Strategy, 'OceanStats.Optim.IOptimizerStrategy') && ...
               isfield(state, 'StrategyState') && ~isempty(state.StrategyState)
                obj.Strategy.set_state_dict(state.StrategyState);
            end
        end

        function reset(obj)
            % RESET  Clear engine and strategy internal states.
            obj.StepCount = 0;
            obj.Strategy.reset();
        end

        function s = describe(obj)
            % DESCRIBE  Human-readable description.
            s = sprintf('VectorizedEngine[%s, Step=%d, GPU=%d]', ...
                obj.Strategy.describe(), obj.StepCount, obj.UseGPU);
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1: Metadata reflection protocol.
            if isa(obj.Strategy, 'OceanStats.Optim.IOptimizerStrategy')
                meta = obj.Strategy.getMetadata();
                meta.EngineName = 'VectorizedOptimizerEngine';
            else
                meta = struct();
                meta.EngineName = 'VectorizedOptimizerEngine';
                meta.SymbolicLatex = 'theta_{t} = theta_{t-1} - f(g_t)';
                meta.Symbols = struct( ...
                    'theta', 'Parameter vector', ...
                    'g', 'Loss gradient vector');
                meta.Assumptions = struct( ...
                    'GradientSmoothness', 'Lipschitz continuous gradient', ...
                    'DataRealValued', 'Real double precision floating point');
            end
        end

        function spec = getPipelineSpec(obj)
            % GETPIPELINESPEC  Pillar 3: Diagram and PortSpec signature.
            spec.Inputs = { ...
                struct('name', 'Target', 'type', 'ITensorTarget', 'domain', 'continuous'), ...
                struct('name', 'Gradients', 'type', 'double', 'domain', 'continuous')};
            spec.Outputs = { ...
                struct('name', 'UpdatedTarget', 'type', 'ITensorTarget', 'domain', 'continuous'), ...
                struct('name', 'Telemetry', 'type', 'struct', 'domain', 'discrete')};
            spec.Kind = 'NONVIRTUAL';
            spec.DiagramSymbol = struct( ...
                'Label', 'Vectorized Engine', ...
                'Shape', 'RectangularBlock', ...
                'ColorTheme', 'Dark');
        end
    end
end
