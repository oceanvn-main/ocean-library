classdef CapabilitySet
    % CAPABILITYSET  Optional capabilities an environment (or composed world) actually provides. Anything not declared
    % true is unsupported, and a component that needs it must fail preflight instead of guessing.
    %
    %   caps = OceanAgents.Core.CapabilitySet('Enumerable', true);
    %   caps.require(["Enumerable" "ExactModel"])   % -> OceanAgents:Core:MissingCapability naming ExactModel
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.7.

    properties (SetAccess = immutable)
        Enumerable logical = false
        ExactModel logical = false
        Snapshot logical = false
        Communication logical = false
        Commitments logical = false
        PrivilegedState logical = false
        TurnBased logical = false
    end

    properties (Constant)
        Names = ["Enumerable" "ExactModel" "Snapshot" "Communication" "Commitments" "PrivilegedState" "TurnBased"]
    end

    methods
        function obj = CapabilitySet(options)
            arguments
                options.Enumerable = false
                options.ExactModel = false
                options.Snapshot = false
                options.Communication = false
                options.Commitments = false
                options.PrivilegedState = false
                options.TurnBased = false
            end
            for name = obj.Names
                obj.(char(name)) = asFlag_(options.(char(name)), char(name));
            end
        end

        function names = list(obj)
            % LIST  Names of the capabilities that are provided.
            has = arrayfun(@(n) obj.(char(n)), obj.Names);
            names = obj.Names(has);
        end

        function require(obj, names)
            % REQUIRE  Throw OceanAgents:Core:MissingCapability listing every requested capability that is absent.
            names = string(names);
            unknown = setdiff(names, obj.Names);
            if ~isempty(unknown)
                fail_('InvalidRecord', 'Unknown capability name(s): %s.', strjoin(unknown, ', '));
            end
            missing = names(~arrayfun(@(n) obj.(char(n)), names));
            if ~isempty(missing)
                fail_('MissingCapability', 'Required capability not provided: %s.', strjoin(missing, ', '));
            end
        end
    end
end
