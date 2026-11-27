classdef IObjectiveApproximation < handle
    % IOBJECTIVEAPPROXIMATION Base class for domain-aware gain approximators.
    %
    % Implements the TEMPLATE METHOD pattern:
    %
    %   computeGain()               <-- concrete template (this class)
    %       |
    %       +-- extractComponents() <-- abstract (subclasses implement)
    %       |       Knows the domain (WLS state, gradients, curvature, etc.)
    %       |
    %       +-- Method.compute()    <-- delegated to pluggable IApproximationMethod
    %               Knows only math (gain formula, e.g. second-order, first-order)
    %
    % This separates TWO concerns:
    %   1. HOW to read gradient/curvature from a domain-specific solver state
    %      → Responsibility of the subclass (extractComponents)
    %   2. WHAT formula to apply given those components
    %      → Responsibility of the injected Method (IApproximationMethod)
    %
    % Subclasses only need to implement extractComponents(), leaving the
    % gain formula entirely pluggable via the Method property.
    %
    % Example:
    %   approx = OceanMath.Approximation.ScoreTestApproximation();
    %   approx.Method = OceanMath.Approximation.FirstOrderApproximation();
    %   gain = approx.computeGain(wls, h_cand, W, s);
    %
    % See also: OceanMath.Approximation.IApproximationMethod,
    %           OceanMath.Approximation.ScoreTestApproximation
    
    properties
        % Pluggable gain formula. Default: second-order (Score Test).
        % Swap with any OceanMath.Approximation.IApproximationMethod subclass.
        Method OceanMath.Approximation.IApproximationMethod = ...
            OceanMath.Approximation.SecondOrderApproximation()
    end
    
    methods
        function gain = computeGain(obj, varargin)
            % COMPUTEGAIN Template method: extract → delegate to Method.
            %
            % Calls extractComponents() to obtain domain-specific gradient
            % and curvature, then forwards them to obj.Method.compute().
            %
            % Any arguments passed here are forwarded to extractComponents().
            % Subclasses define the expected signature of extractComponents().
            [gradient, curvature] = obj.extractComponents(varargin{:});
            gain = obj.Method.compute(gradient, curvature);
        end
    end
    
    methods (Abstract)
        % EXTRACTCOMPONENTS Pull gradient and curvature from domain-specific state.
        %
        % Subclasses receive whatever arguments the calling selector passes
        % (e.g., a WLS state object, a candidate column, weight vector, scores)
        % and return:
        %   gradient  - (1 x K) directional derivative in candidate direction
        %   curvature - scalar  second-order term (variance / Schur complement)
        [gradient, curvature] = extractComponents(obj, varargin)
    end
end
