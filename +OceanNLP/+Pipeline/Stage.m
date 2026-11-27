classdef Stage
    %STAGE  Validated, self-describing step of a ComposablePipeline.
    %   A stage maps one state to the next state. It is invoked either as a
    %   function handle (Step) or as a method on a component (Component,
    %   Method). Construction checks that the stage is callable with one
    %   input, so a malformed stage fails when it is built, not when the
    %   pipeline runs (the scikit-learn / nn.Sequential convention).
    %
    %   Requires and Provides optionally declare which state fields the
    %   stage reads and writes (the spaCy requires/assigns and Kedro
    %   inputs/outputs convention). A stage that declares neither has no
    %   contract; ComposablePipeline.validate treats it as opaque.
    %
    %   Example:
    %       s = OceanNLP.Pipeline.Stage.fromFunction("prompt", @buildPrompt, ...
    %           'Requires', ["Query" "Chunks"], 'Provides', "Prompt");
    properties (SetAccess = private)
        Name string
        Step = []
        Component = []
        Method string = ""
        Requires (1,:) string = strings(1,0)
        Provides (1,:) string = strings(1,0)
        HasContract (1,1) logical = false
        Enabled (1,1) logical = true
    end

    methods
        function obj = Stage(name, options)
            arguments
                name (1,1) string
                options.Step
                options.Component
                options.Method (1,1) string
                options.Requires (1,:) string
                options.Provides (1,:) string
                options.Enabled (1,1) logical = true
            end
            obj.Name = OceanNLP.Pipeline.Stage.validateName_(name);
            hasStep = isfield(options, 'Step');
            hasMethod = isfield(options, 'Component') || isfield(options, 'Method');
            if hasStep == hasMethod
                error('OceanNLP:Pipeline:BadStage', ...
                    'Stage "%s" needs either Step, or Component and Method (not both).', name);
            end
            if hasStep
                obj.Step = OceanNLP.Pipeline.Stage.validateStep_(name, options.Step);
            else
                if ~isfield(options, 'Component') || ~isfield(options, 'Method')
                    error('OceanNLP:Pipeline:BadStage', ...
                        'Stage "%s" needs both Component and Method.', name);
                end
                OceanNLP.Pipeline.Stage.validateMethod_(name, options.Component, options.Method);
                obj.Component = options.Component;
                obj.Method = options.Method;
            end
            if isfield(options, 'Requires'), obj.Requires = unique(options.Requires, 'stable'); end
            if isfield(options, 'Provides'), obj.Provides = unique(options.Provides, 'stable'); end
            obj.HasContract = isfield(options, 'Requires') || isfield(options, 'Provides');
            obj.Enabled = options.Enabled;
        end

        function state = apply(obj, state)
            %APPLY Invoke the stage on one state. This is the single place
            %   where stages are called; ComposablePipeline.run and
            %   OceanNLP.Inspect.PipelineAdapter both go through it.
            if isempty(obj.Component)
                state = obj.Step(state);
            else
                state = obj.Component.(obj.Method)(state);
            end
        end

        function obj = withEnabled(obj, value)
            arguments
                obj
                value (1,1) logical
            end
            obj.Enabled = value;
        end
    end

    methods (Static)
        function obj = fromFunction(name, step, options)
            arguments
                name (1,1) string
                step
                options.Requires (1,:) string
                options.Provides (1,:) string
                options.Enabled (1,1) logical = true
            end
            args = namedArgs_(options);
            obj = OceanNLP.Pipeline.Stage(name, 'Step', step, args{:});
        end

        function obj = fromMethod(name, component, method, options)
            arguments
                name (1,1) string
                component
                method (1,1) string
                options.Requires (1,:) string
                options.Provides (1,:) string
                options.Enabled (1,1) logical = true
            end
            args = namedArgs_(options);
            obj = OceanNLP.Pipeline.Stage(name, 'Component', component, 'Method', method, args{:});
        end

        function obj = fromStruct(s)
            %FROMSTRUCT Convert a legacy struct stage (deprecated form).
            arguments
                s (1,1) struct
            end
            if ~isfield(s, 'Name')
                error('OceanNLP:Pipeline:BadStage', 'Stage struct needs a Name field.');
            end
            fields = intersect(fieldnames(s), ...
                {'Step', 'Component', 'Method', 'Requires', 'Provides', 'Enabled'}, 'stable');
            args = cell(1, 2 * numel(fields));
            for k = 1:numel(fields)
                args{2*k-1} = fields{k};
                args{2*k} = s.(fields{k});
            end
            obj = OceanNLP.Pipeline.Stage(s.Name, args{:});
        end
    end

    methods (Static, Access = private)
        function name = validateName_(name)
            if ~matches(name, regexpPattern("[A-Za-z][A-Za-z0-9_\-]*"))
                error('OceanNLP:Pipeline:BadStageName', ...
                    'Stage name "%s" must start with a letter and contain only letters, digits, "_" or "-".', name);
            end
        end

        function step = validateStep_(name, step)
            if ~isa(step, 'function_handle') || ~isscalar(step)
                error('OceanNLP:Pipeline:BadStage', ...
                    'Stage "%s": Step must be a scalar function handle.', name);
            end
            try
                n = nargin(step);
            catch
                return  % Function not resolvable yet; checked at call time.
            end
            info = functions(step);
            isAnonymous = strcmp(info.type, 'anonymous');
            % n < 0 means varargin; -1 and -2 still accept one input.
            % Named functions may have optional trailing inputs, so only
            % anonymous functions (all inputs required) are rejected for n > 1.
            if n == 0 || n < -2 || (isAnonymous && n > 1)
                error('OceanNLP:Pipeline:BadStage', ...
                    'Stage "%s": Step must accept exactly one input (the state); it declares %d.', name, n);
            end
        end

        function validateMethod_(name, component, method)
            if isstruct(component) && isscalar(component)
                ok = isfield(component, method) && isa(component.(method), 'function_handle');
            elseif isobject(component) && isscalar(component)
                ok = ismethod(component, method);
            else
                ok = false;
            end
            if ~ok
                error('OceanNLP:Pipeline:BadStage', ...
                    'Stage "%s": Component of class %s has no public method "%s".', ...
                    name, class(component), method);
            end
        end
    end
end

function args = namedArgs_(options)
    names = fieldnames(options);
    args = cell(1, 2 * numel(names));
    for k = 1:numel(names)
        args{2*k-1} = names{k};
        args{2*k} = options.(names{k});
    end
end
