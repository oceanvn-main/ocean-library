classdef SourceExtractor
    %SOURCEEXTRACTOR Read MATLAB/Python text and recover symbolic interfaces.
    % Imported files are never evaluated. Every recovered representation keeps
    % its source-line evidence and an explicit fidelity classification.

    methods (Static)
        function result = scanFile(filePath)
            arguments
                filePath (1,1) string
            end
            if ~isfile(filePath)
                error('OceanMath:Symbolic:SourceNotFound', 'Source file not found: %s', filePath);
            end
            [~,~,extension] = fileparts(filePath);
            language = OceanMath.Symbolic.SourceExtractor.languageFromExtension(string(extension));
            result = OceanMath.Symbolic.SourceExtractor.scanText(string(fileread(filePath)), language, filePath);
        end

        function result = scanText(sourceText, language, filePath)
            arguments
                sourceText (1,1) string
                language (1,1) string
                filePath (1,1) string = "<memory>"
            end
            language = lower(language);
            if ~any(language == ["matlab","python"])
                error('OceanMath:Symbolic:Language', 'Language must be matlab or python.');
            end
            if strlength(sourceText) > 2e6
                error('OceanMath:Symbolic:SourceLimit', 'Source text is limited to 2,000,000 characters.');
            end
            lines = splitlines(replace(sourceText, string(sprintf('\r\n')), newline));
            if language == "matlab"
                records = OceanMath.Symbolic.SourceExtractor.scanMatlab(lines, filePath);
            else
                records = OceanMath.Symbolic.SourceExtractor.scanPython(lines, filePath);
            end
            counts = struct('transparent', sum(string({records.classification}) == "transparent"), ...
                'opaque', sum(string({records.classification}) == "opaque"), ...
                'notExtractable', sum(string({records.classification}) == "not-extractable"));
            result = struct('schemaVersion', "1.0", 'file', filePath, 'language', language, ...
                'functions', records, 'counts', counts);
        end

        function validation = validateFormulaCandidate(record, candidateIndex, variableMap)
            arguments
                record (1,1) struct
                candidateIndex (1,1) double {mustBeInteger,mustBePositive}
                variableMap (1,1) struct = struct()
            end
            if ~isfield(record,'formulaCandidates') || candidateIndex>numel(record.formulaCandidates)
                error('OceanMath:Symbolic:FormulaCandidate','Formula candidate index is out of range.');
            end
            candidate=record.formulaCandidates(candidateIndex);
            allowed=[string(record.inputs),string(record.outputs),string(record.displayName)];
            keys=string(fieldnames(variableMap)); unresolved=strings(1,0);
            for k=1:numel(candidate.freeVariables)
                variable=string(candidate.freeVariables(k));
                if any(allowed==variable), continue; end
                keyIndex=find(keys==variable,1);
                if isempty(keyIndex) || ~any(allowed==string(variableMap.(keys(keyIndex))))
                    unresolved(end+1)=variable; %#ok<AGROW>
                end
            end
            validation=struct('validated',isempty(unresolved),'unresolvedVariables',unresolved, ...
                'candidate',candidate,'variableMap',variableMap, ...
                'status',"unknown",'detail',"");
            if isempty(unresolved)
                validation.status="passed";
                validation.detail="Every documented symbol is mapped to the source signature; the formula remains documentation-derived and display-only.";
            else
                validation.detail="Map these documented symbols before use: "+join(unresolved,", ")+".";
            end
        end
    end

    methods (Static, Access=private)
        function language = languageFromExtension(extension)
            if extension == ".m"
                language = "matlab";
            elseif extension == ".py"
                language = "python";
            else
                error('OceanMath:Symbolic:Language', 'Only .m and .py source files are supported.');
            end
        end

        function records = scanMatlab(lines, filePath)
            records = OceanMath.Symbolic.SourceExtractor.emptyRecords();
            starts = find(~cellfun('isempty', regexp(cellstr(lines), '^\s*function(?:\s|$)', 'once')));
            for k = 1:numel(starts)
                first = starts(k); last = numel(lines);
                if k < numel(starts), last = starts(k+1)-1; end
                header = regexp(char(lines(first)), '^\s*function\s+(\[[^\]]*\]|[A-Za-z]\w*)\s*=\s*([A-Za-z]\w*)\s*\(([^)]*)\)', 'tokens', 'once');
                if ~isempty(header)
                    outputs = OceanMath.Symbolic.SourceExtractor.csvNames(string(header{1}));
                    name = string(header{2}); inputs = OceanMath.Symbolic.SourceExtractor.csvNames(string(header{3}));
                else
                    header = regexp(char(lines(first)), '^\s*function\s+([A-Za-z]\w*)\s*\(([^)]*)\)', 'tokens', 'once');
                    if isempty(header), continue; end
                    outputs = strings(1,0); name = string(header{1});
                    inputs = OceanMath.Symbolic.SourceExtractor.csvNames(string(header{2}));
                end
                body = lines(first+1:last);
                records(end+1) = OceanMath.Symbolic.SourceExtractor.makeRecord( ...
                    name, inputs, outputs, body, first, last, "matlab", filePath); %#ok<AGROW>
            end
        end

        function records = scanPython(lines, filePath)
            records = OceanMath.Symbolic.SourceExtractor.emptyRecords();
            starts = find(~cellfun('isempty', regexp(cellstr(lines), '^\s*(?:async\s+)?def\s+[A-Za-z]\w*\s*\(', 'once')));
            for k = 1:numel(starts)
                first = starts(k); headerEnd=first;
                while headerEnd<min(numel(lines),first+20) && ...
                        isempty(regexp(char(strtrim(lines(headerEnd))),'\)\s*(?:->\s*[^:]*)?:\s*$','once'))
                    headerEnd=headerEnd+1;
                end
                headerLine = char(join(strip(lines(first:headerEnd))," "));
                header = regexp(headerLine, '^\s*(?:async\s+)?def\s+([A-Za-z]\w*)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:', 'tokens', 'once');
                if isempty(header), continue; end
                indent = length(headerLine)-length(strtrim(headerLine)); last = numel(lines);
                originalHeader=char(lines(first)); indent=length(originalHeader)-length(strtrim(originalHeader));
                for j = headerEnd+1:numel(lines)
                    text = char(lines(j)); current = length(text)-length(strtrim(text));
                    if ~isempty(strtrim(text)) && current <= indent, last = j-1; break; end
                end
                inputs = OceanMath.Symbolic.SourceExtractor.pythonNames(string(header{2}));
                body = lines(headerEnd+1:last);
                hasReturn=any(~cellfun('isempty',regexp(cellstr(strtrim(body)),'^return(?:\s|$)','once')));
                if hasReturn, outputs="result"; else, outputs=strings(1,0); end
                records(end+1) = OceanMath.Symbolic.SourceExtractor.makeRecord( ...
                    string(header{1}), inputs, outputs, body, first, last, "python", filePath); %#ok<AGROW>
            end
        end

        function record = makeRecord(name, inputs, outputs, body, first, last, language, filePath)
            [expression, literalExpression, sourceLine, support, rationale] = ...
                OceanMath.Symbolic.SourceExtractor.traceExpression(body, inputs, outputs, first, language);
            formulas = OceanMath.Symbolic.SourceExtractor.formulaCandidates(body, first, language);
            looksSideEffectful = any(contains(lower(body), ["plot(","fprintf(","disp(","print(","open(","write(","save(","ui"]),'all');
            if strlength(expression) > 0
                classification = "transparent"; confidence = 0.92;
            elseif isempty(outputs)
                classification = "not-extractable"; confidence = 0.9;
                if looksSideEffectful
                    rationale = "The function exposes side effects and no mathematical output.";
                else
                    rationale = "No returned mathematical output was found.";
                end
            else
                classification = "opaque"; confidence = 0.75;
                if strlength(rationale) == 0
                    rationale = "The signature is usable as an abstract operator, but its implementation is not safely reducible.";
                end
            end
            inputContracts = arrayfun(@(x) struct('name',x,'kind',"unknown",'domain',"unknown",'shape',"unknown"), inputs);
            outputContracts = arrayfun(@(x) struct('name',x,'kind',"unknown",'domain',"unknown",'shape',"unknown"), outputs);
            qualified = OceanMath.Symbolic.SourceExtractor.qualifiedName(filePath, name);
            if isempty(inputs), signature=name+"()"; else, signature=name+"("+join(inputs, ", ")+")"; end
            record = struct('id', qualified + "@L" + string(first), 'qualifiedName', qualified, ...
                'displayName', name, 'language', language, 'file', filePath, ...
                'sourceLines', [first last], 'signature', signature, ...
                'inputs', inputs, 'outputs', outputs, 'classification', classification, ...
                'expression', expression, 'rationale', rationale, 'confidence', confidence, ...
                'contracts', struct('inputs',inputContracts,'outputs',outputContracts), ...
                'computationalSupport', support, 'expressionSourceLine', sourceLine, ...
                'expressionOrigin',"normalized traced expression",'literalExpression',literalExpression, ...
                'formulaCandidates', formulas);
        end

        function [expression, literalExpression, sourceLine, support, rationale] = traceExpression(body, inputs, outputs, first, language)
            expression = ""; literalExpression=""; sourceLine = NaN; support = "abstract-only"; rationale = "";
            if numel(outputs) ~= 1
                rationale = "Multiple or missing outputs require an opaque interface."; return;
            end
            if any(~cellfun('isempty', regexp(cellstr(strtrim(body)), ...
                    '^(if|elseif|else|for|parfor|while|switch|case|try|catch|with|match)(?:\s|\(|$)', 'once')))
                rationale = "Control flow prevents an exact straight-line representation."; return;
            end
            names = strings(0,1); values = strings(0,1); target = outputs(1);
            for k = 1:numel(body)
                line = strtrim(body(k));
                if language == "matlab", line = string(regexprep(char(line), '%.*$', ''));
                else, line = string(regexprep(char(line), '#.*$', '')); end
                if strlength(line) == 0, continue; end
                if language == "python" && startsWith(line, "return ")
                    lhs = target; rhs = extractAfter(line, 7);
                else
                    token = regexp(char(line), '^([A-Za-z]\w*)\s*=\s*(.+?)\s*;?$', 'tokens', 'once');
                    if isempty(token), continue; end
                    lhs = string(token{1}); rhs = string(token{2});
                end
                if contains(rhs,"...") || startsWith(strtrim(rhs),"struct(") || ...
                        count(rhs,"(")~=count(rhs,")")
                    continue;
                end
                rhs = replace(rhs, [".*","./",".^","**","@"], ["*","/","^","^","*"]);
                if contains(rhs, ["[","]","{","}","'",'"',":","==","~="]), continue; end
                rhs = OceanMath.Symbolic.SourceExtractor.substituteLocals(rhs, names, values);
                if isempty(regexp(char(rhs), '^[A-Za-z0-9_+\-*/^()., \t]+$', 'once')), continue; end
                index = find(names == lhs, 1);
                if isempty(index), names(end+1)=lhs; values(end+1)=rhs; else, values(index)=rhs; end %#ok<AGROW>
                if lhs == target
                    expression = rhs; literalExpression = strtrim(body(k)); sourceLine = first+k;
                end
            end
            if strlength(expression) > 0
                identifiers = string(regexp(char(expression), '[A-Za-z][A-Za-z0-9_]*', 'match'));
                calls = string(regexp(char(expression), '[A-Za-z][A-Za-z0-9_]*(?=\s*\()', 'match'));
                builtins = ["exp","sin","cos","log","sqrt","abs","erf"];
                unknown = setdiff(unique(identifiers), [inputs, builtins, "pi", "e", reshape(calls,1,[])]);
                if isempty(unknown) && ~contains(join(body," "), [".*","./",".^","@","sum(","integral(","np.","numpy."])
                    support = "conditional-symbolic";
                else
                    support = "abstract-only";
                end
                rationale = "The output was traced through straight-line assignments.";
            end
        end

        function result = substituteLocals(expression, names, values)
            result = expression;
            for k = numel(names):-1:1
                result = string(regexprep(char(result), ['(?<![A-Za-z0-9_])' regexptranslate('escape',char(names(k))) '(?![A-Za-z0-9_]|\s*\()'], ['(' char(values(k)) ')']));
            end
        end

        function candidates = formulaCandidates(body, first, language)
            candidates = struct('label',{},'expression',{},'origin',{},'sourceLines',{}, ...
                'computationalSupport',{},'freeVariables',{},'validated',{});
            for k = 1:numel(body)
                raw = strtrim(body(k));
                isDoc = (language == "matlab" && startsWith(raw,"%")) || (language == "python" && startsWith(raw,"#"));
                text = regexprep(char(raw), '^\s*[%#]+\s*', '');
                equation = regexp(text, '([A-Za-z][A-Za-z0-9_]*(?:\([^=]*\))?\s*=\s*.+)', 'tokens', 'once');
                if isDoc && ~isempty(equation)
                    formula=string(equation{1}); variables=OceanMath.Symbolic.SourceExtractor.formulaVariables(formula);
                    candidates(end+1) = struct('label',"Documented formula",'expression',formula, ...
                        'origin',"documentation-derived",'sourceLines',[first+k first+k], ...
                        'computationalSupport',"display-only",'freeVariables',variables,'validated',false); %#ok<AGROW>
                end
            end
        end

        function names = csvNames(text)
            text = strip(erase(erase(text,"["),"]"));
            if strlength(text)==0, names=strings(1,0); return; end
            names = strip(split(text,","))'; names = names(strlength(names)>0);
        end

        function names = pythonNames(text)
            names = OceanMath.Symbolic.SourceExtractor.csvNames(text);
            for k=1:numel(names)
                names(k)=strip(extractBefore(names(k)+":",":"));
                names(k)=strip(extractBefore(names(k)+"=","="));
            end
            names = names(names~="self" & names~="cls");
        end

        function qualified = qualifiedName(filePath, name)
            [~,stem] = fileparts(filePath);
            stem = string(matlab.lang.makeValidName(stem));
            if strlength(stem)==0, stem="Source"; end
            qualified = stem + "." + name;
        end

        function records = emptyRecords()
            records = struct('id',{},'qualifiedName',{},'displayName',{},'language',{},'file',{}, ...
                'sourceLines',{},'signature',{},'inputs',{},'outputs',{},'classification',{}, ...
                'expression',{},'rationale',{},'confidence',{},'contracts',{}, ...
                'computationalSupport',{},'expressionSourceLine',{},'expressionOrigin',{}, ...
                'literalExpression',{},'formulaCandidates',{});
        end

        function variables=formulaVariables(formula)
            identifiers=string(regexp(char(formula),'[A-Za-z][A-Za-z0-9_]*','match'));
            builtins=["sum","log","log2","int","integral","exp","sin","cos","sqrt","abs"];
            variables=setdiff(unique(identifiers),builtins,'stable');
        end
    end
end
