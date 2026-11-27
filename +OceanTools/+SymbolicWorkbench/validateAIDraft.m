function draft = validateAIDraft(payload, sourceLineCount)
%VALIDATEAIDRAFT Validate a non-executable symbolic-workbench model draft.
%   This public validator performs no network or symbolic-engine work. It
%   validates the structured model response before a caller displays it or
%   passes it to a symbolic parser.

arguments
    payload (1,1) struct
    sourceLineCount (1,1) double {mustBeInteger, mustBePositive}
end

topLevelFields = {'title', 'summary', 'assumptions', 'limitations', 'candidates'};
validateFieldSet(payload, topLevelFields, 'OceanTools:SymbolicWorkbench:InvalidDraft');

draft = struct;
draft.title = requireText(payload, 'title');
draft.summary = requireText(payload, 'summary');
draft.assumptions = requireStringArray(payload, 'assumptions');
draft.limitations = requireStringArray(payload, 'limitations');

candidateFields = {'name', 'parameters', 'parameterDomains', 'expression', ...
    'outputDomain', 'explanation', 'sourceLines'};
emptyCandidate = struct( ...
    'name', "", ...
    'parameters', strings(1, 0), ...
    'parameterDomains', strings(1, 0), ...
    'expression', "", ...
    'outputDomain', "", ...
    'explanation', "", ...
    'sourceLines', zeros(1, 0));
if isempty(payload.candidates)
    draft.candidates = repmat(emptyCandidate, 1, 0);
    return;
end
if ~isstruct(payload.candidates)
    error('OceanTools:SymbolicWorkbench:InvalidCandidates', ...
        'Candidates must be a struct array or an empty array.');
end
draft.candidates = repmat(emptyCandidate, 1, numel(payload.candidates));

for candidateIndex = 1:numel(payload.candidates)
    candidate = payload.candidates(candidateIndex);
    validateFieldSet(candidate, candidateFields, 'OceanTools:SymbolicWorkbench:InvalidCandidate');

    parameters = requireStringArray(candidate, 'parameters');
    parameterDomains = requireStringArray(candidate, 'parameterDomains');
    if numel(parameters) ~= numel(parameterDomains)
        error('OceanTools:SymbolicWorkbench:ParameterDomainMismatch', ...
            'Each candidate parameter must have one parameter domain.');
    end
    validateDomains(parameterDomains);

    expression = requireText(candidate, 'expression');
    validateScalarExpression(expression);

    outputDomain = lower(requireText(candidate, 'outputDomain'));
    validateDomains(outputDomain);

    explanation = requireText(candidate, 'explanation');
    validateExplanation(explanation);
    sourceLines = validateSourceLines(candidate, sourceLineCount);
    draft.candidates(candidateIndex) = struct( ...
        'name', requireText(candidate, 'name'), ...
        'parameters', parameters, ...
        'parameterDomains', parameterDomains, ...
        'expression', expression, ...
        'outputDomain', outputDomain, ...
        'explanation', explanation, ...
        'sourceLines', sourceLines);
end
end

function validateFieldSet(value, requiredFields, errorId)
actualFields = fieldnames(value);
if ~all(isfield(value, requiredFields)) || numel(actualFields) ~= numel(requiredFields) || ...
        ~isempty(setdiff(actualFields, requiredFields))
    error(errorId, 'The draft does not match the required symbolic-draft schema.');
end
end

function value = requireText(container, fieldName)
value = string(container.(fieldName));
if ~isscalar(value) || ismissing(value) || strlength(strtrim(value)) == 0
    error('OceanTools:SymbolicWorkbench:InvalidText', ...
        'Field %s must be one nonempty text value.', fieldName);
end
value = strtrim(value);
end

function values = requireStringArray(container, fieldName)
rawValues = container.(fieldName);
if isempty(rawValues)
    values = strings(1, 0);
    return;
end

if ~(isstring(rawValues) || ischar(rawValues) || iscell(rawValues))
    error('OceanTools:SymbolicWorkbench:InvalidStringArray', ...
        'Field %s must be an array of text values.', fieldName);
end

values = reshape(string(rawValues), 1, []);
if any(ismissing(values)) || any(strlength(strtrim(values)) == 0)
    error('OceanTools:SymbolicWorkbench:InvalidStringArray', ...
        'Field %s cannot contain empty text values.', fieldName);
end
values = strtrim(values);
end

function validateDomains(domains)
allowedDomains = ["real", "positive", "nonnegative", "integer", "complex"];
if any(~ismember(lower(domains), allowedDomains))
    error('OceanTools:SymbolicWorkbench:InvalidDomain', ...
        'Domains must be real, positive, nonnegative, integer, or complex.');
end
end

function validateScalarExpression(expression)
if isempty(regexp(expression, '^[A-Za-z0-9_+*/^(). \t-]+$', 'once'))
    error('OceanTools:SymbolicWorkbench:InvalidExpression', ...
        'Expressions may contain only scalar symbolic grammar.');
end

functionTokens = regexp(expression, '([A-Za-z_][A-Za-z0-9_]*)\s*\(', 'tokens');
allowedFunctions = ["exp", "sin", "cos", "log", "sqrt", "abs", "erf"];
for tokenIndex = 1:numel(functionTokens)
    functionName = lower(string(functionTokens{tokenIndex}{1}));
    if ~ismember(functionName, allowedFunctions)
        error('OceanTools:SymbolicWorkbench:InvalidExpression', ...
            'Expressions may only call approved scalar symbolic functions.');
    end
end
end

function validateExplanation(explanation)
normalized = lower(explanation);
isSourceFaithful = contains(normalized, "source faithful:");
isHypothetical = contains(normalized, "hypothetical approximation:");
if ~isSourceFaithful && ~isHypothetical
    error('OceanTools:SymbolicWorkbench:InvalidExplanation', ...
        'Each explanation must label the candidate as source faithful or hypothetical.');
end
if isHypothetical && ~contains(normalized, "assumption")
    error('OceanTools:SymbolicWorkbench:InvalidExplanation', ...
        'A hypothetical approximation must state at least one assumption.');
end
end

function sourceLines = validateSourceLines(candidate, sourceLineCount)
sourceLines = candidate.sourceLines;
if ~isnumeric(sourceLines) || isempty(sourceLines) || ~isvector(sourceLines) || ...
        any(~isfinite(sourceLines)) || any(sourceLines ~= floor(sourceLines)) || ...
        any(sourceLines < 1) || any(sourceLines > sourceLineCount)
    error('OceanTools:SymbolicWorkbench:InvalidSourceLines', ...
        'Each candidate must cite one or more valid one-based source line numbers.');
end
sourceLines = reshape(double(sourceLines), 1, []);
end
