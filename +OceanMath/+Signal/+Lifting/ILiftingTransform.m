classdef (Abstract) ILiftingTransform < handle
    % ILIFTINGTRANSFORM  Interface for algebraic lifting transforms (e.g. wavelets).
    %
    %   All multi-resolution wavelet decompositions and other algebraic lifting
    %   schemes must implement this uniform interface.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = public)
        Description char = ''
    end
    
    methods (Abstract)
        % FORWARD Performs the forward transform on input data.
        %
        %   coeffs = forward(obj, data)
        coeffs = forward(obj, data)
        
        % INVERSE Performs the inverse transform (reconstruction) from coefficients.
        %
        %   data = inverse(obj, coeffs)
        data = inverse(obj, coeffs)
    end
end
