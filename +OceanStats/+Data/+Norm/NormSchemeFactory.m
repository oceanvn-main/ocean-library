classdef NormSchemeFactory < handle
    % NORMSCHEMEFACTORY  Instantiate Norm schemes by name.
    
    methods (Static)
        function obj = create(schemeName)
            % CREATE  Return a new INormScheme instance based on a string name.
            arguments
                schemeName (1,1) string
            end
            
            switch lower(schemeName)
                case "l2"
                    obj = OceanStats.Data.Norm.L2Norm();
                case "l1"
                    obj = OceanStats.Data.Norm.L1Norm();
                case "none"
                    obj = OceanStats.Data.Norm.NoneNorm();
                otherwise
                    error('OceanStats:Data:Norm:NormSchemeFactory:UnknownScheme', ...
                          'Unknown Norm scheme: "%s"', schemeName);
            end
        end
    end
end
