classdef (Sealed) BodySignature
    % BODYSIGNATURE  What a body accepts and can sense, and which components it carries. It is the body's answer to
    % the mind's detection probe (design D5a) and what a memory records about the body it was learned on, so a
    % changed body is noticed by comparing signatures (design section 12; AB1 section 6).
    %
    %   d = oldSig.diff(newSig);    d.Added / d.Removed (command labels), d.ComponentsAdded / d.ComponentsRemoved

    properties (SetAccess = immutable)
        Commands = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
        Senses = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
        Components string = strings(1, 0)          % "id:class" of each component
    end

    methods
        function obj = BodySignature(commands, senses, components)
            arguments
                commands = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
                senses = OceanAgents.Minds.Signals.SignalType.empty(1, 0)
                components string = strings(1, 0)
            end
            if ~isempty(commands) && ~all(arrayfun(@(t) t.Role == "command", commands))
                error('OceanAgents:Signals:BadType', 'BodySignature: every command type must have Role "command".');
            end
            obj.Commands = reshape(commands, 1, []);
            obj.Senses = reshape(senses, 1, []);
            obj.Components = reshape(components, 1, []);
            if isempty(obj.Components), obj.Components = strings(1, 0); end
        end

        function labels = commandLabels(obj)
            % COMMANDLABELS  Every label the body accepts as a command (union over label command types).
            labels = strings(1, 0);
            for t = obj.Commands
                if t.ValueKind == "label", labels = [labels, t.Labels]; end %#ok<AGROW>
            end
            labels = unique(labels, 'stable');
        end

        function tf = accepts(obj, offered)
            % ACCEPTS  True when some command type of the body accepts the offered command type.
            tf = any(arrayfun(@(t) t.accepts(offered), obj.Commands));
        end

        function d = diff(obj, other)
            % DIFF  What changes from this body to other.
            a = obj.commandLabels();  b = other.commandLabels();
            d = struct('Added', setdiff(b, a, 'stable'), 'Removed', setdiff(a, b, 'stable'), ...
                'Kept', intersect(a, b, 'stable'), ...
                'ComponentsAdded', setdiff(other.Components, obj.Components, 'stable'), ...
                'ComponentsRemoved', setdiff(obj.Components, other.Components, 'stable'));
        end

        function tf = isequalTo(obj, other)
            d = obj.diff(other);
            tf = isempty(d.Added) && isempty(d.Removed) && isempty(d.ComponentsAdded) && isempty(d.ComponentsRemoved);
        end
    end
end
