classdef TransformPipeline < OceanMath.Image.Pixels.IPixelTransform
    % TRANSFORMPIPELINE  Sequentially chains multiple pixel transformations.
    %
    %   This class implements the Pipes and Filters design pattern, allowing
    %   the caller to chain multiple IPixelTransform operations and execute
    %   them as a single unified transformation.
    %
    %   Example:
    %       pipeline = TransformPipeline({ ...
    %           ColorSpaceConverter('RGB2Gray'), ...
    %           LinearOperator('Scale', 1.2, 'Bias', 10) ...
    %       });
    %       I_out = pipeline.apply(I_in);
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        Transforms (1,:) cell
    end
    
    methods
        function obj = TransformPipeline(transforms)
            % TRANSFORMPIPELINE Construct a pipeline from a cell array of transforms.
            arguments
                transforms (1,:) cell
            end
            
            % Verify all items are instances of IPixelTransform
            for i = 1:length(transforms)
                if ~isa(transforms{i}, 'OceanMath.Image.Pixels.IPixelTransform')
                    error('OceanMath:Vision:Pixels:InvalidTransformType', ...
                        'Element %d in transforms cell array must inherit from IPixelTransform.', i);
                end
            end
            
            obj.Transforms = transforms;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Sequentially execute each filter in the pipeline.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            I_out = I_in;
            for i = 1:length(obj.Transforms)
                I_out = obj.Transforms{i}.apply(I_out);
            end
        end
    end
end
