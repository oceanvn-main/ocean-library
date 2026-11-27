classdef Renamer
    % RENAMER  Consistently renames every NAME in a door set (components, resources, doors, event types) so a test can
    % prove that no guard verdict depends on the text of a name (K14: names are addresses, never evidence).
    %
    %   R = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer;
    %   map = R.mapFor(doors);                 % old name -> opaque new name ("n1", "n2", ...), deterministic
    %   doors2 = R.doors(doors, map);          % renamed DoorSpecs (entity predicates wrapped to see the old names)
    %   offer2 = R.renameFields(offer, map);         % renamed offer fields
    % Axiom IDs, rule names and phases are not renamed: axiom IDs ARE the verdict being compared.

    methods (Static)
        function map = mapFor(doors, events)
            % Every component, resource and door name (and the given event types) -> "z<k>_<random-free tag>".
            if nargin < 2, events = strings(1, 0); end
            names = strings(1, 0);
            flat = OceanAgents.World.Guard.GuardEngine.flattenDoors(doors);
            for d = flat
                names = [names, d.Name, d.ownedNames()]; %#ok<AGROW>
            end
            names = [unique(names, 'stable'), reshape(string(events), 1, [])];
            map = containers.Map('KeyType', 'char', 'ValueType', 'any');
            order = numel(names):-1:1;                           % reversed, so alphabetical order changes as well
            for k = 1:numel(names)
                map(char(names(k))) = sprintf("q%02d", order(k));
            end
        end

        function out = doors(in, map)
            out = OceanAgents.World.DoorSpec.empty(1, 0);
            for d = reshape(in, 1, [])
                out(end + 1) = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.door_(d, map); %#ok<AGROW>
            end
        end

        function s = renameFields(s, map)
            % Rename the fields of a struct (an offer or a values struct).
            out = struct();
            for f = string(fieldnames(s))'
                out.(char(OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.name_(f, map))) = s.(char(f));
            end
            s = out;
        end

        function inv = inverse(map)
            inv = containers.Map('KeyType', 'char', 'ValueType', 'any');
            k = keys(map);
            for j = 1:numel(k)
                inv(char(map(k{j}))) = string(k{j});
            end
        end
    end

    methods (Static, Access = private)
        function n = name_(n, map)
            n = string(n);
            for k = 1:numel(n)
                if isKey(map, char(n(k))), n(k) = map(char(n(k))); end
            end
        end

        function d2 = door_(d, map)
            R = 'OceanTest.OceanAgents.Fixtures.World.Guard.Renamer';
            nm = str2func([R '.name_']);
            owns = d.Owns;
            for k = 1:numel(owns), owns(k).Name = nm(owns(k).Name, map); end
            res = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.renameFields(d.Resources, map);
            rules = OceanAgents.World.RuleSpec.empty(1, 0);
            for r = d.Rules
                dc = r.DimChecks;
                for k = 1:numel(dc)
                    dc(k).Target = nm(dc(k).Target, map);
                    for j = 1:numel(dc(k).Terms)
                        dc(k).Terms{j} = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.factors_(dc(k).Terms{j}, map);
                    end
                end
                rules(end + 1) = OceanAgents.World.RuleSpec('Name', r.Name, 'Phase', r.Phase, 'Kind', r.Kind, ...
                    'Requires', nm(r.Requires, map), 'Reads', nm(r.Reads, map), 'Writes', nm(r.Writes, map), ...
                    'Consumes', nm(r.Consumes, map), 'Emits', nm(r.Emits, map), 'Commands', r.Commands, 'Reveals', nm(r.Reveals, map), ...
                    'Predicts', nm(r.Predicts, map), 'DueEvent', nm(r.DueEvent, map), 'DimChecks', dc, 'Fn', r.Fn); %#ok<AGROW>
            end
            axioms = OceanAgents.World.Guard.AxiomSpec.empty(1, 0);
            inv = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.inverse(map);
            for a = d.Axioms
                args = {'ID', a.ID, 'Scope', a.Scope, 'Kind', a.Kind, 'Mode', a.Mode, 'Probes', a.Probes, ...
                    'Description', a.Description};
                if a.isRange()
                    args = [args, {'Component', nm(a.Component, map), 'Range', a.Range}]; %#ok<AGROW>
                elseif ~isempty(a.Predicate)
                    p = a.Predicate;
                    if a.Scope == "entity"   % the predicate was written against the old addresses: translate back
                        p = @(v) p(OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.renameFields(v, inv));
                    end
                    args = [args, {'Reads', nm(a.Reads, map), 'Predicate', p}]; %#ok<AGROW>
                end
                axioms(end + 1) = OceanAgents.World.Guard.AxiomSpec(args{:}); %#ok<AGROW>
            end
            children = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.doors(d.Children, map);
            d2 = OceanAgents.World.DoorSpec('Name', nm(d.Name, map), 'Version', d.Version, 'Owns', owns, ...
                'Resources', res, 'Membership', nm(d.Membership, map), 'Rules', rules, 'Axioms', axioms, ...
                'Admits', d.Admits, 'Drive', d.Drive, 'DriveK', d.DriveK, 'Init', d.Init, 'Children', children);
        end

        function t = factors_(t, map)
            t = reshape(string(t), 1, []);
            for k = 1:numel(t)
                parts = split(t(k), "^");
                parts(1) = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer.name_(parts(1), map);
                t(k) = join(parts, "^");
            end
        end
    end
end
