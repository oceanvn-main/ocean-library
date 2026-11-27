function spec = pillarSpec_(obj, inputs, outputs, composesFrom)
% PILLARSPEC_  getPipelineSpec() struct for a runtime component. inputs/outputs: 1 x K string arrays of record
% domains (e.g. "Agent-ActorView"); each becomes a port named after the record type (A0 section 8).
arguments
    obj
    inputs string
    outputs string
    composesFrom cell = {}
end
name = string(regexprep(class(obj), '^.*\.', ''));
spec = struct();
spec.ComponentName = name;
spec.Inputs = ports_(inputs);
spec.Outputs = ports_(outputs);
spec.Kind = "NONVIRTUAL";
spec.DiagramSymbol = struct('Type', "default", 'Label', lower(name));
spec.ComposesFrom = composesFrom;
end

function p = ports_(domains)
p = struct('Name', {}, 'Domain', {}, 'Dtype', {}, 'Shape', {});
for k = 1:numel(domains)
    p(k) = struct('Name', erase(domains(k), "Agent-"), 'Domain', domains(k), 'Dtype', "record", 'Shape', "[1 1]");
end
end
