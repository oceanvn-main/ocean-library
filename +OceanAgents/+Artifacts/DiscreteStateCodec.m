classdef DiscreteStateCodec
    % DISCRETESTATECODEC  Maps a discrete observation in {Start, ..., Start+N-1} to a table row 1..N.
    % An observation outside that range is an error, never a silent fallback to row 1 (a fallback would make an
    % agent act on, and learn into, the wrong state without any visible symptom).
    %
    %   c = OceanAgents.Artifacts.DiscreteStateCodec(48);                 % states 1..48 -> rows 1..48
    %   c = OceanAgents.Artifacts.DiscreteStateCodec(5, 'Start', 0);      % states 0..4  -> rows 1..5
    %   c = OceanAgents.Artifacts.DiscreteStateCodec.fromSpace(env.ObservationSpace);
    %   row = c.encode(obs)      % obs - Start + 1   (OceanAgents:Artifacts:UnknownState when out of range)
    %   obs = c.decode(row)
    %
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 6.2.

    properties (SetAccess = immutable)
        N double = 1        % number of states (rows)
        Start double = 1    % observation value of row 1
    end

    methods
        function obj = DiscreteStateCodec(n, options)
            arguments
                n (1,1) double {mustBeInteger, mustBePositive}
                options.Start (1,1) double {mustBeInteger} = 1
            end
            obj.N = n;
            obj.Start = options.Start;
        end

        function row = encode(obj, obs)
            % ENCODE  Row of a scalar integer observation; UnknownState for anything else.
            if ~(isnumeric(obs) && isreal(obs) && isscalar(obs) && isfinite(obs) && obs == fix(obs) ...
                    && obs >= obj.Start && obs <= obj.Start + obj.N - 1)
                error('OceanAgents:Artifacts:UnknownState', 'Observation %s is not a state in %d..%d.', ...
                    mat2str(obs), obj.Start, obj.Start + obj.N - 1);
            end
            row = double(obs) - obj.Start + 1;
        end

        function obs = decode(obj, row)
            % DECODE  Observation value of a row (inverse of encode).
            if ~(isnumeric(row) && isscalar(row) && row == fix(row) && row >= 1 && row <= obj.N)
                error('OceanAgents:Artifacts:UnknownState', 'Row %s outside 1..%d.', mat2str(row), obj.N);
            end
            obs = double(row) + obj.Start - 1;
        end

        function tf = matchesSpace(obj, space)
            % MATCHESSPACE  True when space is a Discrete space with the same N and Start.
            tf = isa(space, 'OceanStats.RL.Env.Spaces.Discrete') && space.N == obj.N && space.Start == obj.Start;
        end
    end

    methods (Static)
        function obj = fromSpace(space)
            % FROMSPACE  Codec for an OceanStats.RL.Env.Spaces.Discrete observation space.
            arguments
                space (1,1) OceanStats.RL.Env.Spaces.Discrete
            end
            obj = OceanAgents.Artifacts.DiscreteStateCodec(space.N, 'Start', space.Start);
        end
    end
end
