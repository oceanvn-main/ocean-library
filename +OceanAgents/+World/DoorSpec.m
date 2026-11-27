classdef DoorSpec
    % DOORSPEC  Declaration of one door: a GUARD (the axioms newcomers must satisfy) and a CONTAINER (the components and
    % resources it alone writes, its rules, its projections). A door may hold sub-doors with the same contract.
    %
    %   D = OceanAgents.World.Guard.Dimension;  T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
    %   door = OceanAgents.World.DoorSpec('Name', "mover", 'Version', "1", ...
    %            'Owns', struct('Name', {"Position", "Velocity"}, 'Type', {T('Width', 2, 'Dimension', D.L()), ...
    %                                                                  T('Width', 2, 'Dimension', D.L() / D.T())}), ...
    %            'Membership', ["Position" "Velocity"], 'Rules', moveRule, 'Drive', "every-tick");
    %
    %   Drive  "every-tick" | "every-k" (with DriveK) | "on-event" (only when its rules have delivered events) | "never"
    %   Init   cmds = Init(stream): spawn Commands run at reset (through normal admission)
    %   Stream "keyed" (default) | "legacy-mt19937ar" (bit-for-bit reproduction of existing environments only)
    % Flattening order (depth-first, parent before children) is the door RANK used for ordering and tie-breaks.
    % Contract: W1_Kernel_Contract.md section 4.2.

    properties (SetAccess = immutable)
        Name string = ""
        Version string = "1"
        Owns struct = struct('Name', {}, 'Type', {})
        Resources struct = struct()
        Membership string = strings(1, 0)
        Rules = OceanAgents.World.RuleSpec.empty(1, 0)
        Axioms = OceanAgents.World.Guard.AxiomSpec.empty(1, 0)
        Admits string = strings(1, 0)
        Drive string = "every-tick"
        DriveK double = 1
        Init = []
        Stream string = "keyed"                 % "keyed" (StreamRegistry purpose door:<Name>) | "legacy-mt19937ar"
        Children = []                           % DoorSpec array (set in the constructor; a self-referencing default is unsafe)
    end

    properties (Constant)
        Drives = ["every-tick" "every-k" "on-event" "never"]
        Streams = ["keyed" "legacy-mt19937ar"]        % legacy: RandStream('mt19937ar','Seed',seed), only to reproduce
                                                      % existing environments bit for bit (W2_Grid_Doors_Contract.md 2)
    end

    methods
        function obj = DoorSpec(options)
            arguments
                options.Name
                options.Version = "1"
                options.Owns = struct('Name', {}, 'Type', {})
                options.Resources = struct()
                options.Membership = strings(1, 0)
                options.Rules = OceanAgents.World.RuleSpec.empty(1, 0)
                options.Axioms = OceanAgents.World.Guard.AxiomSpec.empty(1, 0)
                options.Admits = strings(1, 0)
                options.Drive = "every-tick"
                options.DriveK = 1
                options.Init = []
                options.Stream = "keyed"
                options.Children = []
            end
            bad = @(msg) error('OceanAgents:World:BadAssembly', 'DoorSpec: %s', msg);
            if ~isfield(options, 'Name'), bad('Name is required.'); end
            obj.Name = string(options.Name);
            if ~isscalar(obj.Name) || strlength(obj.Name) == 0, bad('Name must be a non-empty string.'); end
            obj.Version = string(options.Version);
            ow = options.Owns;
            if ~isstruct(ow) || (~isempty(ow) && ~all(isfield(ow, {'Name', 'Type'})))
                bad(sprintf('%s: Owns must be a struct array with fields Name and Type.', obj.Name));
            end
            for k = 1:numel(ow)
                nm = string(ow(k).Name);
                if ~isvarname(char(nm)), bad(sprintf('%s: component name "%s" is not a valid identifier.', obj.Name, nm)); end
                if ~isa(ow(k).Type, 'OceanAgents.World.Guard.ComponentType')
                    bad(sprintf('%s: Owns(%d).Type must be a Guard.ComponentType.', obj.Name, k));
                end
                ow(k).Name = nm;
            end
            obj.Owns = reshape(ow, 1, []);
            if ~isstruct(options.Resources) || ~isscalar(options.Resources)
                bad(sprintf('%s: Resources must be a scalar struct.', obj.Name));
            end
            OceanAgents.Core.assertValueData(options.Resources, obj.Name + ".Resources");
            obj.Resources = options.Resources;
            obj.Membership = reshape(string(options.Membership), 1, []);
            if isempty(obj.Membership), obj.Membership = strings(1, 0); end
            if ~isempty(options.Rules) && ~isa(options.Rules, 'OceanAgents.World.RuleSpec')
                bad(sprintf('%s: Rules must be RuleSpec objects.', obj.Name));
            end
            if isempty(options.Rules), obj.Rules = OceanAgents.World.RuleSpec.empty(1, 0);
            else, obj.Rules = reshape(options.Rules, 1, []); end
            if ~isempty(options.Axioms) && ~isa(options.Axioms, 'OceanAgents.World.Guard.AxiomSpec')
                bad(sprintf('%s: Axioms must be Guard.AxiomSpec objects.', obj.Name));
            end
            if isempty(options.Axioms), obj.Axioms = OceanAgents.World.Guard.AxiomSpec.empty(1, 0);
            else, obj.Axioms = reshape(options.Axioms, 1, []); end
            obj.Admits = reshape(string(options.Admits), 1, []);
            if isempty(obj.Admits), obj.Admits = strings(1, 0); end
            obj.Drive = string(options.Drive);
            if ~any(obj.Drive == obj.Drives), bad(sprintf('%s: unknown Drive "%s".', obj.Name, obj.Drive)); end
            k = options.DriveK;
            if ~(isnumeric(k) && isscalar(k) && k >= 1 && k == fix(k)), bad(sprintf('%s: DriveK must be a positive integer.', obj.Name)); end
            obj.DriveK = double(k);
            if ~isempty(options.Init) && ~isa(options.Init, 'function_handle')
                bad(sprintf('%s: Init must be a function handle.', obj.Name));
            end
            obj.Init = options.Init;
            obj.Stream = string(options.Stream);
            if ~isscalar(obj.Stream) || ~any(obj.Stream == obj.Streams)
                bad(sprintf('%s: Stream must be "keyed" or "legacy-mt19937ar".', obj.Name));
            end
            if ~isempty(options.Children) && ~isa(options.Children, 'OceanAgents.World.DoorSpec')
                bad(sprintf('%s: Children must be DoorSpec objects.', obj.Name));
            end
            if isempty(options.Children), obj.Children = OceanAgents.World.DoorSpec.empty(1, 0);
            else, obj.Children = reshape(options.Children, 1, []); end
        end

        function names = ownedNames(obj)
            % OWNEDNAMES  Components and resources this door alone writes.
            names = [string({obj.Owns.Name}), string(fieldnames(obj.Resources))'];
            if isempty(names), names = strings(1, 0); end
        end

        function [flat, depth] = flatten(obj)
            % FLATTEN  Depth-first list (parent before children) of this door and its sub-doors, with each door's depth
            % (1 = this door). The position in the list is the door rank.
            flat = obj;  depth = 1;
            for c = obj.Children
                [f, d] = c.flatten();
                flat = [flat, f]; %#ok<AGROW>
                depth = [depth, d + 1]; %#ok<AGROW>
            end
        end
    end
end
