classdef Agency
    % AGENCY  The door that owns agent identity (W2_Grid_Doors_Contract.md section 4). It owns one component,
    % AgentID (a string, hidden: agents never see raw IDs through views), and has no rules. Its membership
    % ["AgentID"] marks which entities are agents; AgentWorld builds its roster from the AgentID values of the alive
    % Agency members, in entity-ID order.
    %
    %   d = OceanAgents.World.Doors.Agency.spec();            % DoorSpec "agency"
    %   t = OceanAgents.World.Doors.Agency.idType();          % the ComponentType to offer AgentID with
    %   offer.AgentID = struct('Value', "alice", 'Type', t);  % in a spawn offer (any door may spawn an agent)

    properties (Constant)
        Name = "agency"
    end

    methods (Static)
        function d = spec()
            d = OceanAgents.World.DoorSpec('Name', OceanAgents.World.Doors.Agency.Name, 'Version', "1", ...
                'Owns', struct('Name', "AgentID", 'Type', OceanAgents.World.Doors.Agency.idType()), ...
                'Membership', "AgentID", 'Drive', "never");
        end

        function t = idType()
            % AgentID: one hidden string per entity.
            t = OceanAgents.World.Guard.ComponentType('Class', "string", 'Width', 1, 'Visibility', "hidden");
        end

        function meta = getMetadata()
            meta = struct('SymbolicLatex', strings(1, 0), 'Symbols', struct('Symbol', {}, 'Description', {}), ...
                'Assumptions', {{'owner of agent identity (hidden AgentID); membership marker; no rules'}}, ...
                'IsSymbolic', false, 'Reference', "W2_Grid_Doors_Contract.md section 4");
        end

        function spec = getPipelineSpec()
            spec = struct('ComponentName', "Agency", ...
                'Inputs', struct('Name', {}, 'Domain', {}, 'Dtype', {}, 'Shape', {}), ...
                'Outputs', struct('Name', "AgentID", 'Domain', "World-Component", 'Dtype', "string", 'Shape', "n x 1"), ...
                'Kind', "NONVIRTUAL", 'DiagramSymbol', struct('Type', "default", 'Label', "agency"), 'ComposesFrom', {{}});
        end
    end
end
