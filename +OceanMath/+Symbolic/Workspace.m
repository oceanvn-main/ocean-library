classdef Workspace < handle
    % WORKSPACE Scalar symbolic algebra with named, contracted mathematical blocks.
    % Expressions are parsed using an explicit mathematical grammar. Source text
    % never enters eval, str2func, or a symbolic string-expression evaluator.
    % Symbol assumptions use session-unique internal names, preserving the user's
    % interactive symbolic workspace. Display names are restored only as text.

    properties (SetAccess = private, SetObservable)
        Symbols struct = struct('name', {}, 'domain', {})
        Definitions struct = struct('name', {}, 'parameters', {}, ...
            'parameterDomains', {}, 'expression', {}, 'outputDomain', {}, 'provenance', {})
        Notices string = strings(0, 1)
    end
    properties (Access = private)
        SymbolValues cell = cell(1, 0)
        Prefix (1,1) string
        SymbolCounter (1,1) double = 0
    end

    methods
        function obj = Workspace()
            arguments
            end
            if exist('sym', 'file') ~= 2
                error('OceanMath:Symbolic:ToolboxRequired', ...
                    'Symbolic Math Toolbox is required for this workspace.');
            end
            [~, token] = fileparts(tempname);
            obj.Prefix = "ows" + erase(string(token), "_");
        end

        function actualName = addSymbol(obj, name, domain)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
                name (1,1) string
                domain (1,1) string = "real"
            end
            obj.checkIdentifier(name);
            obj.checkDomain(domain);
            if numel(obj.Symbols) >= 50
                error('OceanMath:Symbolic:Limit', 'A workspace supports at most 50 symbols.');
            end
            actualName = obj.availableName(name);
            value = obj.newSymbol(domain);
            obj.Symbols(end+1) = struct('name', actualName, 'domain', domain);
            obj.SymbolValues{end+1} = value;
            obj.noteRename(name, actualName);
        end

        function actualName = define(obj, name, parameters, expression, parameterDomains, outputDomain, provenance)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
                name (1,1) string
                parameters string
                expression (1,1) string
                parameterDomains string = strings(1,0)
                outputDomain (1,1) string = "real"
                provenance (1,1) struct = struct()
            end
            obj.checkIdentifier(name);
            obj.checkDomain(outputDomain);
            parameters = reshape(parameters, 1, []);
            parameterDomains = reshape(parameterDomains, 1, []);
            if isempty(parameterDomains), parameterDomains = repmat("real", size(parameters)); end
            if numel(parameters) > 12 || numel(parameters) ~= numel(parameterDomains)
                error('OceanMath:Symbolic:Parameters', 'Provide at most 12 parameters and one domain per parameter.');
            end
            if numel(unique(parameters)) ~= numel(parameters)
                error('OceanMath:Symbolic:Parameters', 'Parameter names must be unique within a block.');
            end
            for k = 1:numel(parameters)
                obj.checkIdentifier(parameters(k));
                if any(parameters(k) == obj.reservedNames())
                    error('OceanMath:Symbolic:Parameters', 'Parameter names cannot use reserved mathematical names.');
                end
                obj.checkDomain(parameterDomains(k));
            end
            if numel(obj.Definitions) >= 30
                error('OceanMath:Symbolic:Limit', 'A workspace supports at most 30 mathematical blocks.');
            end
            actualName = obj.availableName(name);
            localValues = cell(1, numel(parameters));
            for k = 1:numel(parameters), localValues{k} = obj.newSymbol(parameterDomains(k)); end
            cleanup = onCleanup(@() obj.clearPrivateAssumptions(localValues)); %#ok<NASGU>
            [parsed, checks] = obj.parse(expression, parameters, localValues, strings(1,0));
            if parsed.equation
                error('OceanMath:Symbolic:ScalarOnly', 'A block must define a scalar expression, not an equation.');
            end
            check = obj.domainCheck(parsed.value, outputDomain, "Output of " + actualName);
            checks(end+1) = check;
            obj.Definitions(end+1) = struct('name', actualName, 'parameters', parameters, ...
                'parameterDomains', parameterDomains, 'expression', expression, ...
                'outputDomain', outputDomain, 'provenance', provenance);
            obj.noteRename(name, actualName);
            unknown = checks(string({checks.status}) == "unknown");
            for k = 1:numel(unknown)
                obj.Notices(end+1,1) = unknown(k).label + ": " + unknown(k).detail;
            end
        end

        function actualName = rename(obj, oldName, newName)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
                oldName (1,1) string
                newName (1,1) string
            end
            obj.checkIdentifier(newName);
            symbolIndex = find(string({obj.Symbols.name}) == oldName, 1);
            definitionIndex = find(string({obj.Definitions.name}) == oldName, 1);
            if isempty(symbolIndex) && isempty(definitionIndex)
                error('OceanMath:Symbolic:UnknownName', 'No symbol or block is named %s.', oldName);
            end
            if oldName == newName, actualName = oldName; return; end
            if ~isempty(symbolIndex)
                % Renaming a global into a formal parameter would capture its
                % formerly free references inside the affected block body.
                actualName = obj.availableName(newName, obj.formalNames());
            else
                actualName = obj.availableName(newName);
            end
            bundle = obj.exportBundle();
            if ~isempty(symbolIndex)
                bundle.symbols(symbolIndex).name = actualName;
                mapping = struct('oldName', oldName, 'newName', actualName, 'kind', "symbol");
            else
                bundle.definitions(definitionIndex).name = actualName;
                mapping = struct('oldName', oldName, 'newName', actualName, 'kind', "definition");
            end
            for k = 1:numel(bundle.definitions)
                bundle.definitions(k).expression = obj.rewriteNames(bundle.definitions(k).expression, ...
                    mapping, bundle.definitions(k).parameters);
            end
            trial = OceanMath.Symbolic.Workspace();
            trial.importBundle(bundle);
            obj.adopt(trial);
            obj.noteRename(newName, actualName);
            obj.Notices(end+1,1) = "Renamed " + oldName + " to " + actualName + "; dependent references were updated.";
        end

        function report = inspect(obj, expression)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
                expression (1,1) string
            end
            [parsed, checks] = obj.parse(expression, strings(1,0), cell(1,0), strings(1,0));
            report = obj.makeReport(expression, parsed, parsed.value, checks);
        end

        function report = calculate(obj, expression, operation, variable, substitutions)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
                expression (1,1) string
                operation (1,1) string
                variable (1,1) string = "x"
                substitutions (1,1) struct = struct()
            end
            allowed = ["simplify", "expand", "factor", "differentiate", "integrate", "solve", "substitute", "evaluate"];
            if ~any(operation == allowed)
                error('OceanMath:Symbolic:Operation', 'Choose simplify, expand, factor, differentiate, integrate, solve, substitute, or evaluate.');
            end
            [parsed, checks, constraints] = obj.parse(expression, strings(1,0), cell(1,0), strings(1,0));
            value = parsed.value;
            if parsed.equation && ~any(operation == ["solve", "simplify"])
                error('OceanMath:Symbolic:Equation', 'Equations currently support solve and simplify. Use a scalar expression for other operations.');
            end
            if any(operation == ["differentiate", "integrate", "solve"])
                index = find(string({obj.Symbols.name}) == variable, 1);
                if isempty(index)
                    error('OceanMath:Symbolic:UnknownName', 'Declare the operation variable %s first.', variable);
                end
                independent = obj.SymbolValues{index};
            end
            keys = string(fieldnames(substitutions));
            if ~isempty(keys) && ~any(operation == ["substitute", "evaluate"])
                error('OceanMath:Symbolic:Substitution', 'Use substitute or evaluate when supplying substitutions.');
            end
            if ~isempty(keys)
                checks = struct('label', {}, 'status', {}, 'detail', {});
                from = sym(zeros(1, numel(keys)));
                to = sym(zeros(1, numel(keys)));
                for k = 1:numel(keys)
                    index = find(string({obj.Symbols.name}) == keys(k), 1);
                    if isempty(index)
                        error('OceanMath:Symbolic:UnknownName', 'Unknown substitution symbol %s.', keys(k));
                    end
                    replacement = substitutions.(keys(k));
                    if ~(isstring(replacement) && isscalar(replacement)) && ~(ischar(replacement) && isrow(replacement))
                        error('OceanMath:Symbolic:Substitution', 'Substitution values must be mathematical expression text.');
                    end
                    [item, itemChecks] = obj.parse(string(replacement), strings(1,0), cell(1,0), strings(1,0));
                    if item.equation, error('OceanMath:Symbolic:Substitution', 'Substitution values must be scalar expressions.'); end
                    checks = [checks, itemChecks]; %#ok<AGROW>
                    checks(end+1) = obj.domainCheck(item.value, obj.Symbols(index).domain, "Substitution for " + keys(k));
                    from(k) = obj.SymbolValues{index}; to(k) = item.value;
                end
                % Temporary symbols make replacement simultaneous, including swaps.
                temporary = sym(zeros(size(from)));
                for k = 1:numel(from), temporary(k) = obj.newSymbol("complex"); end
                value = subs(subs(value, from, temporary), temporary, to);
                for k = 1:numel(constraints)
                    actual = subs(subs(constraints(k).value, from, temporary), temporary, to);
                    checks(end+1) = obj.domainCheck(actual, constraints(k).domain, ...
                        "After substitution: " + constraints(k).label);
                end
            end
            switch operation
                case "simplify", result = simplify(value, 'Steps', 30);
                case "expand"
                    if parsed.termBound > 2000
                        error('OceanMath:Symbolic:Limit', 'This expansion could exceed 2000 terms; simplify the expression first.');
                    end
                    result = expand(value);
                case "factor", result = factor(value);
                case "differentiate", result = diff(value, independent);
                case "integrate", result = int(value, independent);
                case "solve"
                    if parsed.equation, polynomial = lhs(value) - rhs(value); else, polynomial = value; end
                    try
                        degree = double(polynomialDegree(polynomial, independent));
                    catch
                        error('OceanMath:Symbolic:SolveLimit', 'Solve currently supports scalar polynomial equations of degree 1 through 4.');
                    end
                    if ~isscalar(degree) || ~isfinite(degree) || degree < 1 || degree > 4
                        error('OceanMath:Symbolic:SolveLimit', 'Solve currently supports scalar polynomial equations of degree 1 through 4.');
                    end
                    if parsed.equation
                        [result, solveParameters, conditions] = solve(value, independent, 'ReturnConditions', true);
                    else
                        [result, solveParameters, conditions] = solve(value == 0, independent, 'ReturnConditions', true);
                    end
                    checks(end+1) = struct('label', "Solution conditions", 'status', "unknown", ...
                        'detail', "Conditions: " + obj.displayText(conditions) + "; parameters: " + obj.displayText(solveParameters));
                case "substitute", result = value;
                case "evaluate", result = vpa(value, 16);
            end
            obj.checkFinite(result);
            if strlength(string(char(result))) > 30000
                error('OceanMath:Symbolic:Limit', 'The result exceeds the display limit; use a smaller expression.');
            end
            if operation == "factor"
                checks(end+1) = struct('label', "Factor representation", 'status', "passed", ...
                    'detail', "The result lists multiplicative factors returned by MATLAB factor.");
            end
            report = obj.makeReport(expression, parsed, result, checks);
        end

        function renames = importBundle(obj, bundle)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
                bundle (1,1) struct
            end
            if isfield(bundle, 'symbols') && isempty(bundle.symbols)
                bundle.symbols = struct('name', {}, 'domain', {});
            end
            if isfield(bundle, 'definitions') && isempty(bundle.definitions)
                bundle.definitions = struct('name', {}, 'parameters', {}, 'parameterDomains', {}, ...
                    'expression', {}, 'outputDomain', {}, 'provenance', {});
            end
            if ~isfield(bundle, 'symbols') || ~isfield(bundle, 'definitions') || ...
                    ~isstruct(bundle.symbols) || ~isstruct(bundle.definitions)
                error('OceanMath:Symbolic:Bundle', 'A bundle requires symbols and definitions struct arrays.');
            end
            if numel(bundle.symbols) + numel(obj.Symbols) > 50 || numel(bundle.definitions) + numel(obj.Definitions) > 30
                error('OceanMath:Symbolic:Limit', 'The merged workspace would exceed its symbol or block limit.');
            end
            % Work on a detached session. A failed import never mutates obj.
            trial = OceanMath.Symbolic.Workspace();
            for k = 1:numel(obj.Symbols), trial.addSymbol(obj.Symbols(k).name, obj.Symbols(k).domain); end
            for k = 1:numel(obj.Definitions), trial.addDefinitionRecord(obj.Definitions(k)); end
            trial.Notices = obj.Notices;
            formals = trial.formalNames();
            for k = 1:numel(bundle.definitions)
                if ~all(isfield(bundle.definitions(k), {'name','parameters','parameterDomains','expression','outputDomain','provenance'}))
                    error('OceanMath:Symbolic:Bundle', 'Every imported block must include its name, parameters, parameterDomains, expression, outputDomain, and provenance.');
                end
                formals = [formals, reshape(string(bundle.definitions(k).parameters),1,[])]; %#ok<AGROW>
            end
            formals = unique(formals);
            count = numel(bundle.symbols) + numel(bundle.definitions);
            renames = repmat(struct('oldName', "", 'newName', "", 'kind', ""), 1, count);
            planned = strings(1, count); oldNames = strings(1, count);
            cursor = 0;
            for k = 1:numel(bundle.symbols)
                if ~all(isfield(bundle.symbols(k), {'name','domain'}))
                    error('OceanMath:Symbolic:Bundle', 'Every imported symbol requires name and domain.');
                end
                cursor = cursor + 1;
                oldNames(cursor) = string(bundle.symbols(k).name);
                obj.checkIdentifier(oldNames(cursor));
                % An unchanged imported name preserves its original scope.
                % A collision suffix must not land on any formal parameter.
                forbidden = [planned(1:cursor-1), formals(formals ~= oldNames(cursor))];
                planned(cursor) = trial.availableName(oldNames(cursor), forbidden);
                renames(cursor) = struct('oldName', oldNames(cursor), 'newName', planned(cursor), 'kind', "symbol");
            end
            for k = 1:numel(bundle.definitions)
                if ~all(isfield(bundle.definitions(k), {'name','parameters','parameterDomains','expression','outputDomain','provenance'}))
                    error('OceanMath:Symbolic:Bundle', 'Every imported block must include its name, parameters, parameterDomains, expression, outputDomain, and provenance.');
                end
                cursor = cursor + 1;
                oldNames(cursor) = string(bundle.definitions(k).name);
                obj.checkIdentifier(oldNames(cursor));
                planned(cursor) = trial.availableName(oldNames(cursor), planned(1:cursor-1));
                renames(cursor) = struct('oldName', oldNames(cursor), 'newName', planned(cursor), 'kind', "definition");
            end
            if numel(unique(oldNames)) ~= numel(oldNames)
                error('OceanMath:Symbolic:Bundle', 'Imported symbol and block names must be unique.');
            end
            for k = 1:numel(bundle.symbols)
                trial.addSymbol(renames(k).newName, string(bundle.symbols(k).domain));
            end
            for k = 1:numel(bundle.definitions)
                item = bundle.definitions(k);
                item.name = renames(numel(bundle.symbols)+k).newName;
                item.expression = obj.rewriteNames(string(item.expression), renames, string(item.parameters));
                trial.addDefinitionRecord(item);
            end
            for k = 1:numel(renames), trial.noteRename(renames(k).oldName, renames(k).newName); end
            obj.adopt(trial);
        end

        function bundle = exportBundle(obj)
            arguments
                obj (1,1) OceanMath.Symbolic.Workspace
            end
            bundle = struct('schemaVersion', "1.0", 'kind', "ocean-symbolic-workspace", ...
                'scope', "scalar mathematical expressions", 'symbols', obj.Symbols, 'definitions', obj.Definitions);
        end
    end

    methods (Static)
        function names = reservedNames()
            arguments
            end
            names = ["pi", "e", "i", "exp", "sin", "cos", "log", "sqrt", "abs", "erf"];
        end
        function grammar = identifierGrammar()
            arguments
            end
            grammar = "A letter followed by up to 39 letters, digits, or underscores.";
        end
    end

    methods (Access = private)
        function value = newSymbol(obj, domain)
            obj.SymbolCounter = obj.SymbolCounter + 1;
            internal = obj.Prefix + "v" + string(obj.SymbolCounter);
            value = sym(char(internal));
            switch domain
                case "real", assume(value, 'real');
                case "positive", assume(value, 'positive');
                case "nonnegative", assume(value, 'real'); assumeAlso(value >= 0);
                case "integer", assume(value, 'integer');
                case "complex", assume(value, 'clear');
            end
        end

        function [node, checks, constraints] = parse(obj, expression, localNames, localValues, callStack)
            expression = strtrim(string(expression));
            if strlength(expression) == 0 || strlength(expression) > 2048
                error('OceanMath:Symbolic:Expression', 'An expression must contain 1 through 2048 characters.');
            end
            chars = char(expression); tokens = cell(1, 300); count = 0; offset = 1;
            pattern = '^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?';
            while offset <= numel(chars)
                if isspace(chars(offset)), offset = offset + 1; continue; end
                tail = chars(offset:end);
                lexeme = regexp(tail, pattern, 'match', 'once');
                if isempty(lexeme), lexeme = regexp(tail, '^[A-Za-z][A-Za-z0-9_]*', 'match', 'once'); end
                if isempty(lexeme)
                    if startsWith(tail, '**'), lexeme = '**';
                    elseif contains('+-*/^(),=', chars(offset)), lexeme = chars(offset);
                    else
                        error('OceanMath:Symbolic:Syntax', 'Unsupported character at position %d. Only scalar mathematical syntax is accepted.', offset);
                    end
                end
                count = count + 1;
                if count > 300, error('OceanMath:Symbolic:Limit', 'Use at most 300 expression tokens.'); end
                tokens{count} = lexeme; offset = offset + numel(lexeme);
            end
            tokens = tokens(1:count); cursor = 1; depth = 0;
            checks = struct('label', {}, 'status', {}, 'detail', {});
            constraints = struct('label', {}, 'domain', {}, 'value', {});
            node = readExpression(0);
            if peek() == "="
                cursor = cursor + 1; equationRight = readExpression(0);
                node.value = node.value == equationRight.value;
                node.latex = node.latex + " = " + equationRight.latex;
                node.equation = true;
            end
            if cursor <= count
                error('OceanMath:Symbolic:Syntax', 'Unexpected trailing syntax. Use explicit multiplication and one expression or equation.');
            end
            obj.checkFinite(node.value);

            function peeked = peek()
                if cursor <= count, peeked = string(tokens{cursor}); else, peeked = ""; end
            end
            function expect(expectedToken)
                if peek() ~= expectedToken, error('OceanMath:Symbolic:Syntax', 'Expected %s.', expectedToken); end
                cursor = cursor + 1;
            end
            function recordContract(contractValue, contractDomain, contractLabel)
                checks(end+1) = obj.domainCheck(contractValue, contractDomain, contractLabel);
                constraints(end+1) = struct('label', contractLabel, 'domain', contractDomain, 'value', contractValue);
            end
            function item = readExpression(minimum)
                depth = depth + 1;
                if depth > 40, error('OceanMath:Symbolic:Limit', 'Expression nesting is limited to 40 levels.'); end
                token = peek();
                if token == "", error('OceanMath:Symbolic:Syntax', 'The expression is incomplete.'); end
                cursor = cursor + 1;
                item = struct('value', sym(0), 'latex', "", 'equation', false, 'termBound', 1);
                if any(token == ["+", "-"])
                    child = readExpression(25); item = child;
                    if token == "-", item.value = -child.value; item.latex = "-\left(" + child.latex + "\right)"; end
                elseif token == "("
                    item = readExpression(0); expect(")"); item.latex = "\left(" + item.latex + "\right)";
                elseif ~isempty(regexp(char(token), pattern, 'once'))
                    item.value = obj.numericLiteral(token); item.latex = string(latex(item.value));
                elseif ~isempty(regexp(char(token), '^[A-Za-z][A-Za-z0-9_]*$', 'once'))
                    if peek() == "("
                        cursor = cursor + 1; args = cell(1,12); argc = 0;
                        if peek() ~= ")"
                            while true
                                argc = argc + 1;
                                if argc > 12, error('OceanMath:Symbolic:Arity', 'A call supports at most 12 arguments.'); end
                                args{argc} = readExpression(0);
                                if peek() ~= ",", break; end
                                cursor = cursor + 1;
                            end
                        end
                        expect(")"); args = args(1:argc);
                        values = cellfun(@(a) a.value, args, 'UniformOutput', false);
                        argLatex = cellfun(@(a) char(a.latex), args, 'UniformOutput', false);
                        item.latex = "\mathrm{" + obj.latexName(token) + "}\left(" + string(strjoin(argLatex, ', ')) + "\right)";
                        builtins = ["exp", "sin", "cos", "log", "sqrt", "abs", "erf"];
                        if any(token == builtins)
                            if argc ~= 1, error('OceanMath:Symbolic:Arity', '%s requires exactly one argument.', token); end
                            switch token
                                case "exp", item.value = exp(values{1});
                                case "sin", item.value = sin(values{1});
                                case "cos", item.value = cos(values{1});
                                case "log"
                                    recordContract(values{1}, "nonzero", "Logarithm argument");
                                    item.value = log(values{1});
                                case "sqrt", item.value = sqrt(values{1});
                                case "abs", item.value = abs(values{1});
                                case "erf", item.value = erf(values{1});
                            end
                        else
                            index = find(string({obj.Definitions.name}) == token, 1);
                            if isempty(index)
                                error('OceanMath:Symbolic:UnknownName', 'Unknown block %s. Define dependencies before their callers.', token);
                            end
                            if any(callStack == token) || numel(callStack) >= 30
                                error('OceanMath:Symbolic:Cycle', 'Cyclic or excessively deep block definitions are unsupported.');
                            end
                            definition = obj.Definitions(index);
                            if argc ~= numel(definition.parameters)
                                error('OceanMath:Symbolic:Arity', '%s requires %d arguments.', token, numel(definition.parameters));
                            end
                            for j = 1:argc
                                recordContract(values{j}, definition.parameterDomains(j), token + " argument " + definition.parameters(j));
                            end
                            [body, nestedChecks, nestedConstraints] = obj.parse(definition.expression, definition.parameters, values, [callStack, token]);
                            checks = [checks, nestedChecks];
                            constraints = [constraints, nestedConstraints];
                            recordContract(body.value, definition.outputDomain, "Output of " + token);
                            item.value = body.value; item.termBound = body.termBound;
                        end
                    else
                        localIndex = find(localNames == token, 1);
                        globalIndex = find(string({obj.Symbols.name}) == token, 1);
                        if ~isempty(localIndex), item.value = localValues{localIndex};
                        elseif ~isempty(globalIndex), item.value = obj.SymbolValues{globalIndex};
                        elseif token == "pi", item.value = sym(pi);
                        elseif token == "e", item.value = exp(sym(1));
                        elseif token == "i", item.value = sym(1i);
                        else, error('OceanMath:Symbolic:UnknownName', 'Undeclared symbol %s.', token);
                        end
                        if token == "pi", item.latex = "\pi"; else, item.latex = obj.latexName(token); end
                    end
                else
                    error('OceanMath:Symbolic:Syntax', 'Unexpected token %s.', token);
                end
                while true
                    operation = peek();
                    if any(operation == ["^", "**"]), precedence = 30;
                    elseif any(operation == ["*", "/"]), precedence = 20;
                    elseif any(operation == ["+", "-"]), precedence = 10;
                    else, precedence = -1;
                    end
                    if precedence < minimum, break; end
                    cursor = cursor + 1;
                    right = readExpression(precedence + double(~any(operation == ["^", "**"])));
                    left = item;
                    switch operation
                        case "+", item.value = left.value + right.value; item.latex = left.latex + " + " + right.latex; item.termBound = min(2001, left.termBound + right.termBound);
                        case "-", item.value = left.value - right.value; item.latex = left.latex + " - " + right.latex; item.termBound = min(2001, left.termBound + right.termBound);
                        case "*", item.value = left.value * right.value; item.latex = "\left(" + left.latex + "\right)\," + "\left(" + right.latex + "\right)"; item.termBound = min(2001, left.termBound * right.termBound);
                        case "/"
                            recordContract(right.value, "nonzero", "Division denominator");
                            item.value = left.value / right.value; item.latex = "\frac{" + left.latex + "}{" + right.latex + "}"; item.termBound = left.termBound;
                        otherwise
                            if isempty(symvar(right.value))
                                exponent = double(right.value);
                                if ~isreal(exponent) || ~isfinite(exponent) || abs(exponent) > 100
                                    error('OceanMath:Symbolic:Limit', 'Numeric exponents must be real and between -100 and 100.');
                                end
                                if exponent < 0, recordContract(left.value, "nonzero", "Negative power base"); end
                                if exponent >= 0 && exponent == floor(exponent)
                                    item.termBound = obj.powerTermBound(left.termBound, exponent);
                                else, item.termBound = 1;
                                end
                            else, item.termBound = 2001;
                            end
                            item.value = left.value ^ right.value; item.latex = "\left(" + left.latex + "\right)^{" + right.latex + "}";
                    end
                    obj.checkFinite(item.value);
                    if strlength(string(char(item.value))) > 10000
                        error('OceanMath:Symbolic:Limit', 'The expanded expression is too large; use smaller blocks.');
                    end
                end
                depth = depth - 1;
            end
        end

        function check = domainCheck(obj, value, domain, label)
            obj.checkFinite(value);
            if ~isscalar(value), error('OceanMath:Symbolic:ScalarOnly', 'Only scalar mathematical values are supported.'); end
            status = "unknown";
            if domain == "complex"
                status = "passed";
            else
                try
                    realCondition = imag(value) == 0;
                    switch domain
                        case "real", condition = realCondition;
                        case "positive", condition = realCondition & value > 0;
                        case "nonnegative", condition = realCondition & value >= 0;
                        case "integer", condition = realCondition & value == floor(value);
                        case "nonzero", condition = value ~= 0;
                    end
                    if isAlways(condition, 'Unknown', 'false'), status = "passed";
                    elseif isAlways(~condition, 'Unknown', 'false')
                        if domain == "nonzero"
                            error('OceanMath:Symbolic:Nonfinite', '%s must not be zero.', label);
                        end
                        error('OceanMath:Symbolic:Domain', '%s violates its %s domain contract.', label, domain);
                    end
                catch exception
                    if any(strcmp(exception.identifier, {'OceanMath:Symbolic:Domain', 'OceanMath:Symbolic:Nonfinite'})), rethrow(exception); end
                    % Numeric complex values can make ordered comparisons invalid.
                    if isempty(symvar(value))
                        numeric = double(value);
                        if ~isreal(numeric)
                            error('OceanMath:Symbolic:Domain', '%s violates its %s domain contract.', label, domain);
                        end
                    end
                end
            end
            if status == "passed", detail = "The value satisfies the declared " + domain + " domain under current assumptions.";
            else, detail = "The " + domain + " requirement could not be proved or disproved; supply stronger assumptions before treating it as established.";
            end
            check = struct('label', label, 'status', status, 'detail', detail);
        end

        function report = makeReport(obj, expression, parsed, result, checks)
            variables = symvar(result); free = strings(1, numel(variables));
            for k = 1:numel(variables), free(k) = obj.displayText(variables(k)); end
            report = struct('compact', expression, 'expanded', obj.displayText(parsed.value), ...
                'result', obj.displayText(result), 'compactLatex', parsed.latex, ...
                'expandedLatex', obj.displayLatex(parsed.value), 'resultLatex', obj.displayLatex(result), ...
                'checks', checks, 'freeSymbols', free, 'symbolicExpanded', parsed.value, 'symbolicResult', result);
        end
        function output = displayText(obj, value)
            output = string(char(value));
            for k = 1:numel(obj.SymbolValues)
                output = string(regexprep(char(output), ['\<', char(obj.SymbolValues{k}), '\>'], char(obj.Symbols(k).name)));
            end
        end
        function output = displayLatex(obj, value)
            output = string(latex(value));
            for k = 1:numel(obj.SymbolValues)
                output = replace(output, string(latex(obj.SymbolValues{k})), obj.latexName(obj.Symbols(k).name));
            end
        end
        function actual = availableName(obj, requested, additional)
            if nargin < 3, additional = strings(1,0); end
            occupied = [string({obj.Symbols.name}), string({obj.Definitions.name}), obj.reservedNames(), reshape(additional,1,[])];
            actual = requested; suffix = 2;
            while any(occupied == actual)
                ending = "_" + string(suffix);
                stem = extractBefore(requested, min(strlength(requested), 40-strlength(ending))+1);
                actual = stem + ending; suffix = suffix + 1;
            end
        end
        function names = formalNames(obj)
            names = strings(1,0);
            for k = 1:numel(obj.Definitions)
                names = [names, reshape(obj.Definitions(k).parameters,1,[])]; %#ok<AGROW>
            end
            names = unique(names);
        end
        function noteRename(obj, requested, actual)
            if requested ~= actual
                obj.Notices(end+1,1) = "Name " + requested + " was already used or reserved; preserved it and added " + actual + ".";
            end
        end
        function adopt(obj, other)
            obj.Symbols = other.Symbols; obj.Definitions = other.Definitions;
            obj.SymbolValues = other.SymbolValues; obj.Prefix = other.Prefix;
            obj.SymbolCounter = other.SymbolCounter; obj.Notices = other.Notices;
        end
        function addDefinitionRecord(obj, item)
            obj.define(string(item.name), string(item.parameters), string(item.expression), ...
                string(item.parameterDomains), string(item.outputDomain), item.provenance);
        end
    end

    methods (Static, Access = private)
        function checkIdentifier(name)
            if ~isscalar(name) || isempty(regexp(char(name), '^[A-Za-z][A-Za-z0-9_]{0,39}$', 'once'))
                error('OceanMath:Symbolic:Identifier', 'Use a letter followed by at most 39 letters, digits, or underscores.');
            end
        end
        function checkDomain(domain)
            if ~any(domain == ["real", "positive", "nonnegative", "integer", "complex"])
                error('OceanMath:Symbolic:Domain', 'Domain must be real, positive, nonnegative, integer, or complex.');
            end
        end
        function checkFinite(value)
            representation = string(char(value));
            if ~isempty(regexp(char(representation), '\<(Inf|NaN|undefined|infinity)\>', 'once'))
                error('OceanMath:Symbolic:Nonfinite', 'The expression has an undefined or nonfinite value; revise its domain or substitutions.');
            end
        end
        function value = numericLiteral(token)
            if strlength(token) > 80, error('OceanMath:Symbolic:Limit', 'Numeric literals are limited to 80 characters.'); end
            segments = regexp(char(token), '[eE]', 'split');
            mantissa = segments{1}; exponent = 0;
            if numel(segments) == 2, exponent = str2double(segments{2}); end
            if ~isfinite(exponent) || abs(exponent) > 100
                error('OceanMath:Symbolic:Limit', 'Numeric literal exponents are limited to -100 through 100.');
            end
            point = strfind(mantissa, '.'); decimals = 0;
            if ~isempty(point), decimals = numel(mantissa)-point; end
            digitsOnly = erase(mantissa, '.'); value = sym(0);
            for k = 1:numel(digitsOnly), value = value*10 + double(digitsOnly(k))-double('0'); end
            value = value * sym(10)^(exponent-decimals);
        end
        function text = latexName(name)
            text = replace(string(name), "_", "\_");
        end
        function result = powerTermBound(terms, exponent)
            result = 1;
            for k = 1:min(terms-1, exponent)
                result = result * (terms+exponent-1-(k-1))/k;
                if result > 2000, result = 2001; return; end
            end
        end
        function clearPrivateAssumptions(values)
            for k = 1:numel(values)
                try, assume(values{k}, 'clear'); catch, end
            end
        end
        function rewritten = rewriteNames(expression, mapping, locals)
            chars = char(expression);
            [starts, ends, names] = regexp(chars, '(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*', 'start', 'end', 'match');
            pieces = cell(1, 2*numel(names)+1); last = 1;
            for k = 1:numel(names)
                pieces{2*k-1} = chars(last:starts(k)-1);
                tail = strtrim(chars(ends(k)+1:end)); isCall = startsWith(tail, '(');
                name = string(names{k}); replacement = name;
                for j = 1:numel(mapping)
                    if mapping(j).oldName == name && ...
                            ((isCall && mapping(j).kind == "definition") || ...
                            (~isCall && mapping(j).kind == "symbol" && ~any(string(locals) == name)))
                        replacement = mapping(j).newName; break;
                    end
                end
                pieces{2*k} = char(replacement); last = ends(k)+1;
            end
            pieces{end} = chars(last:end); rewritten = string([pieces{:}]);
        end
    end
end
