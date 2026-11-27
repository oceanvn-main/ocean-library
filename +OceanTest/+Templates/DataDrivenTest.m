classdef DataDrivenTest < OceanTest.BaseTestCase
    % DATADRIVENTEST Example of running 1 test logic on multiple inputs.
    %
    % FEATURES:
    %   - TestParameter: Automatically generates combinations of inputs.
    %   - Useful for testing a function against Edge Cases, Normal, and Error inputs.
    
    properties (TestParameter)
        % Define your inputs here.
        % The test method will run once for EACH parameter.
        Shape = {'Square', 'Circle', 'Triangle'};
        Size = {1, 10, 100};
    end
    
    methods (Test)
        function testArea(testCase, Shape, Size)
            % This method runs 3x3 = 9 times!
            % fprintf('Testing %s with size %d\n', Shape, Size);
            
            % Mock Logic
            if strcmp(Shape, 'Square')
                area = Size^2;
                testCase.verifyGreaterThan(area, 0);
            end
        end
    end
end
