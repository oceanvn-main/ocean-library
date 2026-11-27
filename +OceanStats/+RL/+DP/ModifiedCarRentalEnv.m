classdef ModifiedCarRentalEnv < OceanStats.RL.Env.Classic.ModifiedCarRentalEnv
    % MODIFIEDCARRENTALENV  DEPRECATED name kept for existing callers: Jack's Car Rental (Sutton & Barto Ex 4.2 / 4.7)
    % now lives, algorithm-free, in OceanStats.RL.Env.Classic.ModifiedCarRentalEnv. New code should use that class.
    methods
        function obj = ModifiedCarRentalEnv(varargin)
            obj@OceanStats.RL.Env.Classic.ModifiedCarRentalEnv(varargin{:});
        end
    end
end
