classdef (Abstract) INeighborhoodFilter < OceanMath.Image.Pixels.IPixelTransform
    % INEIGHBORHOODFILTER  Base class for spatial neighborhood filters.
    %
    %   This class implements the Template Method design pattern. It automates
    %   common neighborhood filtering tasks: calculating padding sizes, applying
    %   boundary padding using BoundaryPadder, triggering subclass filtering hooks,
    %   and restoring the output image to the input data type.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = public)
        Boundary      = 'replicate' % Padding boundary mode
        ConstantValue = 0.0         % Value used for constant padding
    end
    
    methods
        function obj = INeighborhoodFilter(options)
            arguments
                options.Boundary = 'replicate'
                options.ConstantValue = 0.0
            end
            obj.Boundary = options.Boundary;
            obj.ConstantValue = options.ConstantValue;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Template method orchestrating neighborhood filtering.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            % 1. Determine padding size from subclass
            pad_size = obj.getPadSize();
            
            % 2. Apply boundary padding using the modular BoundaryPadder
            I_padded = OceanMath.Image.Pixels.BoundaryPadder.pad(...
                I_in, pad_size, obj.Boundary, obj.ConstantValue);
            
            % 3. Invoke subclass filtering logic
            I_out_double = obj.applyFilter(I_padded, I_in);
            
            % 4. Restore original class type (uint8, single, double)
            if isa(I_in, 'uint8')
                I_out = uint8(max(0.0, min(255.0, I_out_double)));
            elseif isa(I_in, 'single')
                I_out = single(I_out_double);
            else
                I_out = I_out_double;
            end
        end
    end
    
    methods (Abstract, Access = protected)
        % GETPADSIZE Subclass hook to return [pad_rows, pad_cols] padding requirements.
        pad_size = getPadSize(obj)
        
        % APPLYFILTER Subclass hook performing the neighborhood filtering.
        I_out = applyFilter(obj, I_padded, I_in)
    end
end
