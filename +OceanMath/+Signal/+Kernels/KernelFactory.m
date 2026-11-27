classdef KernelFactory
    % KERNELFACTORY  Static factory and resolver for discrete filter kernels.
    %
    %   This class provides a centralized mechanism for instantiating and resolving
    %   filter kernels. It allows upstream callers to pass kernel instances,
    %   built-in string names (e.g. 'gaussian'), or raw numeric arrays.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    methods (Static)
        function kernel = create(name, varargin)
            % CREATE  Instantiates a built-in IFilterKernel by name.
            %
            %   kernel = KernelFactory.create('gaussian', 'Size', 5)
            %   kernel = KernelFactory.create('triangle', 3)
            %   kernel = KernelFactory.create('box', 5)
            
            import OceanMath.Signal.Kernels.*
            
            switch lower(name)
                case {'gaussian', 'gauss'}
                    kernel = GaussianKernel(varargin{:});
                case {'triangle', 'bilinear', 'tent'}
                    kernel = TriangleKernel(varargin{:});
                case 'box'
                    kernel = BoxKernel(varargin{:});
                case 'sobel'
                    kernel = SobelKernel(varargin{:});
                case {'second_derivative', 'laplacian_diff', 'corner'}
                    kernel = SecondDerivativeKernel(varargin{:});
                case {'sinc', 'windowedsinc'}
                    kernel = SincKernel(varargin{:});
                otherwise
                    error('KernelFactory:UnknownKernel', ...
                        'Unknown kernel type: %s. Supported: gaussian, triangle, box, sobel, second_derivative, sinc.', name);
            end
        end
        
        function kernel = resolve(input_val)
            % RESOLVE  Safely translates any input into an IFilterKernel instance.
            %
            %   - Existing IFilterKernel: Returns it unchanged.
            %   - Numeric array: Returns a CustomKernel adapter wrapping the array.
            %   - String/Char: Resolves to a default built-in kernel by name.
            
            if isa(input_val, 'OceanMath.Signal.Kernels.IFilterKernel')
                kernel = input_val;
            elseif isnumeric(input_val)
                kernel = OceanMath.Signal.Kernels.CustomKernel(input_val);
            elseif ischar(input_val) || isstring(input_val)
                kernel = OceanMath.Signal.Kernels.KernelFactory.create(char(input_val));
            else
                error('KernelFactory:InvalidKernelInput', ...
                    'Kernel must be an IFilterKernel, a numeric array, or a valid string identifier.');
            end
        end
    end
end
