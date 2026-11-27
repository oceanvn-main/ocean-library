classdef MatlabOptimizationWrapper < handle
    % MATLABOPTIMIZATIONWRAPPER Wrapper around MATLAB's built-in fminunc solver.
    %
    % NAMESPACE: OceanAlgo.Optimization
    %
    % Provides a clean, strategy-pattern interface around fminunc with live
    % telemetry logging via OutputFcn callback. Supports Steepest Descent,
    % Newton-Raphson, DFP, and BFGS algorithms.
    %
    % Author: Ocean Library System
    % Date: 2026-09-03
    
    methods
        function [x_opt, f_opt, history] = solve(obj, fun, x0, options)
            % SOLVE Executes unconstrained optimization using fminunc.
            %
            % Args:
            %   fun (function_handle): Objective function handle @(x) -> scalar.
            %   x0 (double): Initial starting point vector [1 x D] or [D x 1].
            %   options (struct/arguments): Optional configuration parameters.
            %
            % Options fields:
            %   Method (string): "bfgs" (default), "dfp", "steepdesc", or "newton".
            %   IsMaximization (logical): true (default for likelihood) or false.
            %   MaxIter (double): Max iterations (default 200).
            %   TolFun (double): Function tolerance (default 1e-8).
            %   TolX (double): Step tolerance (default 1e-8).
            %   Display (string): "off" (default) or "iter".
            %
            % Returns:
            %   x_opt (double): Optimal parameter vector.
            %   f_opt (double): Optimal objective value.
            %   history (struct): Telemetry struct containing fields x, f, grad_norm, step_norm, iterations.
            
            arguments
                obj
                fun (1,1) function_handle
                x0 double
                options.Method (1,1) string = "bfgs"
                options.IsMaximization (1,1) logical = true
                options.MaxIter (1,1) double = 200
                options.TolFun (1,1) double = 1e-8
                options.TolX (1,1) double = 1e-8
                options.Display (1,1) string = "off"
            end
            
            x0 = x0(:)'; % Force row vector [1 x D]
            isMax = options.IsMaximization;
            
            % Setup objective function direction (fminunc minimizes)
            if isMax
                targetFun = @(x) -fun(x);
            else
                targetFun = fun;
            end
            
            % Telemetry accumulators
            x_hist = [];
            f_hist = [];
            grad_hist = [];
            step_hist = [];
            
            % OutputFcn callback
            function stop = telemetryOutputFcn(x, optimValues, state)
                stop = false;
                switch state
                    case {'init', 'iter'}
                        x_curr = x(:)';
                        x_hist = [x_hist; x_curr];
                        
                        f_val = optimValues.fval;
                        if isMax
                            f_val = -f_val;
                        end
                        f_hist = [f_hist; f_val];
                        
                        if isfield(optimValues, 'firstorderopt') && ~isempty(optimValues.firstorderopt)
                            grad_hist = [grad_hist; optimValues.firstorderopt];
                        else
                            grad_hist = [grad_hist; NaN];
                        end
                        
                        if isfield(optimValues, 'stepsize') && ~isempty(optimValues.stepsize)
                            step_hist = [step_hist; optimValues.stepsize];
                        else
                            step_hist = [step_hist; 0];
                        end
                    case 'done'
                        % Completed
                end
            end
            
            % Configure HessUpdate & Algorithm for fminunc
            methodLower = lower(options.Method);
            switch methodLower
                case "steepdesc"
                    optimOpts = optimoptions('fminunc', ...
                        'Algorithm', 'quasi-newton', ...
                        'HessUpdate', 'steepdesc', ...
                        'MaxIterations', options.MaxIter, ...
                        'OptimalityTolerance', options.TolFun, ...
                        'StepTolerance', options.TolX, ...
                        'Display', options.Display, ...
                        'OutputFcn', @telemetryOutputFcn);
                case "dfp"
                    optimOpts = optimoptions('fminunc', ...
                        'Algorithm', 'quasi-newton', ...
                        'HessUpdate', 'dfp', ...
                        'MaxIterations', options.MaxIter, ...
                        'OptimalityTolerance', options.TolFun, ...
                        'StepTolerance', options.TolX, ...
                        'Display', options.Display, ...
                        'OutputFcn', @telemetryOutputFcn);
                case "bfgs"
                    optimOpts = optimoptions('fminunc', ...
                        'Algorithm', 'quasi-newton', ...
                        'HessUpdate', 'bfgs', ...
                        'MaxIterations', options.MaxIter, ...
                        'OptimalityTolerance', options.TolFun, ...
                        'StepTolerance', options.TolX, ...
                        'Display', options.Display, ...
                        'OutputFcn', @telemetryOutputFcn);
                case "newton"
                    optimOpts = optimoptions('fminunc', ...
                        'Algorithm', 'quasi-newton', ...
                        'HessUpdate', 'bfgs', ... % Standard Newton proxy in fminunc when exact Hessian isn't supplied
                        'MaxIterations', options.MaxIter, ...
                        'OptimalityTolerance', options.TolFun, ...
                        'StepTolerance', options.TolX, ...
                        'Display', options.Display, ...
                        'OutputFcn', @telemetryOutputFcn);
                otherwise
                    error("MatlabOptimizationWrapper:UnknownMethod", ...
                        "Unknown method '%s'. Supported methods: 'steepdesc', 'dfp', 'bfgs', 'newton'.", options.Method);
            end
            
            % Execute optimization
            [x_opt_vec, f_opt_target, exitflag, output_raw] = fminunc(targetFun, x0, optimOpts);
            
            x_opt = x_opt_vec(:)';
            if isMax
                f_opt = -f_opt_target;
            else
                f_opt = f_opt_target;
            end
            
            % Package telemetry history
            history = struct();
            history.x = x_hist;
            history.f = f_hist;
            history.grad_norm = grad_hist;
            history.step_norm = step_hist;
            history.iterations = output_raw.iterations;
            history.funcCount = output_raw.funcCount;
            history.exitflag = exitflag;
            history.algorithm = output_raw.algorithm;
            history.method = options.Method;
        end
    end
end
