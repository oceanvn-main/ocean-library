classdef ComposablePipeline < handle
    %COMPOSABLEPIPELINE  Mutable named stage pipeline with contract checks.
    %   Stages (OceanNLP.Pipeline.Stage) can be added, replaced, removed,
    %   enabled, or reordered without changing the executor. Validation
    %   happens at three levels:
    %     1. add/replace: each stage is callable with one input, names are
    %        unique, and positions are in range.
    %     2. validate/analyze: walks the enabled stages in order and checks
    %        that every declared Requires field is available from the
    %        initial state or an earlier stage's Provides (spaCy
    %        analyze_pipes). run calls this before executing anything.
    %     3. run: stage errors name the failing stage; with CheckContracts,
    %        each stage's declared Provides fields are verified after it runs.
    %
    %   Stages without a contract are opaque: they are reported, never
    %   rejected, and a requirement that only an opaque stage could satisfy
    %   is a warning instead of an error.
    %
    %   Edits never check the chain by default, so a pipeline can pass
    %   through broken intermediate states on purpose (e.g. a two-step
    %   reorder). Set AutoValidate = true to get an
    %   OceanNLP:Pipeline:ChainBroken warning after any edit that leaves the
    %   chain broken. The edit still happens; run always refuses a broken
    %   chain regardless of this setting.
    %
    %   Passing a struct stage is deprecated; use Stage.fromFunction or
    %   Stage.fromMethod.
    properties (SetAccess = private)
        Stages (1,:) OceanNLP.Pipeline.Stage = OceanNLP.Pipeline.Stage.empty(1,0)
    end
    properties
        % Check the chain after every edit and warn if it is broken.
        AutoValidate (1,1) logical = false
        % State fields assumed present before the first stage: the default
        % for validate/analyze and the basis for AutoValidate.
        InitialFields (1,:) string = strings(1,0)
    end

    methods
        function obj = ComposablePipeline(stages, options)
            arguments
                stages = {}
                options.AutoValidate (1,1) logical = false
                options.InitialFields (1,:) string = strings(1,0)
            end
            obj.InitialFields = options.InitialFields;
            if iscell(stages)
                for i = 1:numel(stages), obj.add(stages{i}); end
            else
                for i = 1:numel(stages), obj.add(stages(i)); end
            end
            % Turned on after the initial stages are in, so building the
            % list does not warn at every intermediate step.
            obj.AutoValidate = options.AutoValidate;
            obj.autoCheck_("construction");
        end

        function add(obj, stage, options)
            arguments
                obj
                stage
                options.Position (1,1) double = inf
            end
            stage = obj.toStage_(stage);
            if any(obj.names() == stage.Name)
                error('OceanNLP:Pipeline:DuplicateName', 'A stage named "%s" already exists.', stage.Name);
            end
            n = numel(obj.Stages);
            if isinf(options.Position)
                obj.Stages(end+1) = stage;
            else
                position = options.Position;
                if position ~= fix(position) || position < 1 || position > n + 1
                    error('OceanNLP:Pipeline:BadPosition', ...
                        'Position must be an integer from 1 to %d; got %g.', n + 1, position);
                end
                obj.Stages = [obj.Stages(1:position-1), stage, obj.Stages(position:end)];
            end
            obj.autoCheck_(sprintf('add("%s")', stage.Name));
        end

        function replace(obj, name, stage)
            arguments
                obj
                name (1,1) string
                stage
            end
            i = obj.find_(name); %The old
            stage = obj.toStage_(stage);
            others = obj.names();
            others(i) = [];
            if any(others == stage.Name)
                error('OceanNLP:Pipeline:DuplicateName', 'A stage named "%s" already exists.', stage.Name);
            end
            obj.Stages(i) = stage;
            obj.autoCheck_(sprintf('replace("%s")', name));
        end

        function remove(obj, name)
            arguments
                obj
                name (1,1) string
            end
            obj.Stages(obj.find_(name)) = [];
            obj.autoCheck_(sprintf('remove("%s")', name));
        end

        function enable(obj, name, value)
            arguments
                obj
                name (1,1) string
                value (1,1) logical
            end
            i = obj.find_(name);
            obj.Stages(i) = obj.Stages(i).withEnabled(value);
            obj.autoCheck_(sprintf('enable("%s", %s)', name, mat2str(value)));
        end

        function move(obj, name, position)
            arguments
                obj
                name (1,1) string
                position (1,1) double {mustBeInteger, mustBePositive}
            end
            i = obj.find_(name);
            s = obj.Stages(i);
            obj.Stages(i) = [];
            position = min(position, numel(obj.Stages) + 1);
            obj.Stages = [obj.Stages(1:position-1), s, obj.Stages(position:end)];
            obj.autoCheck_(sprintf('move("%s", %d)', name, position));
        end

        function report = validate(obj, options)
            %VALIDATE Check the enabled stages' contracts without running them.
            %   report.IsValid is false only when an Error is found.
            %   report.Issues is a table: Severity, Position, Stage, Message.
            %   Severity is "error" (requirement nothing upstream provides),
            %   "warning" (requirement only an opaque stage might provide, or
            %   a field overwritten by a later stage) or "info" (stage has
            %   no contract).
            arguments
                obj
                options.InitialFields (1,:) string = obj.InitialFields
            end
            available = options.InitialFields;
            producer = repmat("initial state", size(available));
            opaqueSeen = false;
            severity = strings(0,1); position = zeros(0,1); stage = strings(0,1); message = strings(0,1);
            for i = 1:numel(obj.Stages)
                s = obj.Stages(i);
                if ~s.Enabled, continue, end
                if ~s.HasContract
                    severity(end+1,1) = "info"; position(end+1,1) = i; stage(end+1,1) = s.Name; %#ok<AGROW>
                    message(end+1,1) = "No Requires/Provides declared; stage is not checked."; %#ok<AGROW>
                    opaqueSeen = true;
                    continue
                end
                missing = setdiff(s.Requires, available, 'stable');
                for f = missing
                    if opaqueSeen
                        sev = "warning";
                        msg = sprintf('Requires "%s", which only an earlier unchecked stage could provide.', f);
                    else
                        sev = "error";
                        msg = sprintf('Requires "%s", but neither the initial state nor an earlier stage provides it.', f);
                    end
                    severity(end+1,1) = sev; position(end+1,1) = i; stage(end+1,1) = s.Name; message(end+1,1) = msg; %#ok<AGROW>
                end
                [overwritten, where] = intersect(available, s.Provides, 'stable');
                for k = 1:numel(overwritten)
                    severity(end+1,1) = "warning"; position(end+1,1) = i; stage(end+1,1) = s.Name; %#ok<AGROW>
                    message(end+1,1) = sprintf('Overwrites "%s" from %s.', overwritten(k), producer(where(k))); %#ok<AGROW>
                end
                added = setdiff(s.Provides, available, 'stable');
                available = [available, added]; %#ok<AGROW>
                producer = [producer, repmat("stage """ + s.Name + """", size(added))]; %#ok<AGROW>
            end
            report.IsValid = ~any(severity == "error");
            report.Issues = table(severity, position, stage, message, ...
                'VariableNames', {'Severity', 'Position', 'Stage', 'Message'});
            report.AvailableFields = available;
        end

        function summary = analyze(obj, options)
            %ANALYZE Readable per-stage view of what each stage reads and writes.
            arguments
                obj
                options.InitialFields (1,:) string = obj.InitialFields
            end
            report = obj.validate('InitialFields', options.InitialFields);
            n = numel(obj.Stages);
            Position = (1:n)';
            Name = strings(n,1); Enabled = false(n,1); Requires = strings(n,1); Provides = strings(n,1); Status = strings(n,1);
            for i = 1:n
                s = obj.Stages(i);
                Name(i) = s.Name; Enabled(i) = s.Enabled;
                Requires(i) = strjoin(s.Requires, ", "); Provides(i) = strjoin(s.Provides, ", ");
                sev = report.Issues.Severity(report.Issues.Position == i);
                if ~s.Enabled, Status(i) = "disabled";
                elseif any(sev == "error"), Status(i) = "error";
                elseif any(sev == "warning"), Status(i) = "warning";
                elseif ~s.HasContract, Status(i) = "unchecked";
                else, Status(i) = "ok";
                end
            end
            summary = table(Position, Name, Enabled, Requires, Provides, Status);
            if nargout == 0
                disp(summary);
                if ~isempty(report.Issues), disp(report.Issues); end
                clear summary
            end
        end

        function assertValid(obj, state)
            %ASSERTVALID Throw OceanNLP:Pipeline:InvalidPipeline when the
            %   stage contracts cannot be met from this initial state.
            arguments
                obj
                state
            end
            initial = strings(1,0);
            if isstruct(state), initial = string(fieldnames(state))'; end
            report = obj.validate('InitialFields', initial);
            if ~report.IsValid
                errors = report.Issues(report.Issues.Severity == "error", :);
                lines = compose('  stage %d "%s": %s', errors.Position, errors.Stage, errors.Message);
                error('OceanNLP:Pipeline:InvalidPipeline', ...
                    'Pipeline contracts are not satisfied:\n%s', strjoin(lines', newline));
            end
        end

        function state = run(obj, state, options)
            arguments
                obj
                state
                options.CheckContracts (1,1) logical = false
            end
            obj.assertValid(state);
            for i = 1:numel(obj.Stages)
                s = obj.Stages(i);
                if ~s.Enabled, continue, end
                try
                    state = s.apply(state);
                catch cause
                    id = cause.identifier;
                    if isempty(id), id = 'OceanNLP:Pipeline:StageFailed'; end
                    ME = MException(id, 'Stage %d "%s" failed: %s', i, s.Name, cause.message);
                    throw(addCause(ME, cause));
                end
                if options.CheckContracts && s.HasContract
                    absent = s.Provides;
                    if isstruct(state), absent = setdiff(s.Provides, string(fieldnames(state)), 'stable'); end
                    if ~isempty(absent)
                        error('OceanNLP:Pipeline:ContractViolated', ...
                            'Stage %d "%s" declared Provides "%s" but did not produce it.', ...
                            i, s.Name, strjoin(absent, '", "'));
                    end
                end
            end
        end

        function names = names(obj)
            names = [obj.Stages.Name];
            if isempty(names), names = strings(1,0); end
        end
    end

    methods (Access = private)
        function autoCheck_(obj, action)
            %AUTOCHECK_ Warn (never throw) when AutoValidate is on and the
            %   edit just made left the chain broken.
            if ~obj.AutoValidate, return, end
            report = obj.validate();
            if report.IsValid, return, end
            errors = report.Issues(report.Issues.Severity == "error", :);
            lines = compose('  stage %d "%s": %s', errors.Position, errors.Stage, errors.Message);
            warning('OceanNLP:Pipeline:ChainBroken', 'After %s:\n%s', action, strjoin(lines', newline));
        end

        function i = find_(obj, name)
            i = find(obj.names() == name, 1);
            if isempty(i)
                error('OceanNLP:Pipeline:MissingStage', 'Unknown stage "%s".', name);
            end
        end
    end

    methods (Static, Access = private)
        function stage = toStage_(stage)
            if isstruct(stage)
                warning('OceanNLP:Pipeline:StructStageDeprecated', ...
                    ['Struct stages are deprecated; use OceanNLP.Pipeline.Stage.fromFunction ' ...
                     'or Stage.fromMethod.']);
                stage = OceanNLP.Pipeline.Stage.fromStruct(stage);
            elseif ~isa(stage, 'OceanNLP.Pipeline.Stage') || ~isscalar(stage)
                error('OceanNLP:Pipeline:BadStage', ...
                    'Expected a scalar OceanNLP.Pipeline.Stage; got %s.', class(stage));
            end
        end
    end
end
