function fig = RAGPipelineDemoApp()
%RAGPIPELINEDEMOAPP Interactive UIHTML frontend for OceanNLP.RAG.RAGPipeline.

appFolder = fileparts(mfilename('fullpath'));
packageParent = fileparts(fileparts(fileparts(appFolder)));
currentPaths = strsplit(path, pathsep);
if ~ismember(packageParent, currentPaths)
    addpath(packageParent);
end

fig = uifigure( ...
    'Name', 'OceanNLP RAG Pipeline Demo', ...
    'Position', [100 100 1360 900], ...
    'Color', [0.96 0.97 0.98]);
rootGrid = uigridlayout(fig, [1 1]);
rootGrid.Padding = [0 0 0 0];
rootGrid.RowSpacing = 0;
html = uihtml(rootGrid);
html.HTMLEventReceivedFcn = @onUIEvent;
html.HTMLSource = fullfile(appFolder, 'index.html');

pipeline = [];
indexedDocumentCount = 0;
activeGenerationMode = "";

fig.CloseRequestFcn = @(~,~) delete(fig);

    function onUIEvent(src, event)
        eventName = "unknown";
        try
            eventName = string(event.HTMLEventName);
            eventData = event.HTMLEventData;
            switch eventName
                case "indexCorpus"
                    sendEventToHTMLSource(src, 'operationStarted', ...
                        struct('operation', 'index', 'message', 'Preparing corpus and building the index...'));
                    pipeline = [];
                    indexedDocumentCount = 0;
                    activeGenerationMode = "";
                    config = validateConfig_(eventData);
                    documents = parseDocuments_(getText_(eventData, 'corpus', 500000));
                    chunker = OceanNLP.Chunking.ChunkerFactory.create(config.chunker, ...
                        'WindowSize', config.windowSize, ...
                        'Overlap', config.overlap, ...
                        'MaxCharacters', config.maxCharacters, ...
                        'WindowRadius', config.windowRadius);
                    switch config.retriever
                        case "bm25"
                            ranker = OceanNLP.Retrieval.BM25Ranker();
                        case "semantic"
                            ranker = OceanNLP.Retrieval.SemanticRanker();
                    end
                    if config.generationMode == "demo"
                        generator = OceanNLP.Generation.FunctionGenerator( ...
                            @demoGenerate_);
                    else
                        generator = OceanNLP.Generation.LocalLLMGenerator( ...
                            'Endpoint', config.endpoint, 'Model', config.model);
                    end
                    candidate = OceanNLP.RAG.RAGPipeline( ...
                        'Chunker', chunker, ...
                        'Ranker', ranker, ...
                        'Generator', generator, ...
                        'TopK', config.topK);
                    candidate.index(documents);
                    pipeline = candidate;
                    indexedDocumentCount = numel(documents);
                    activeGenerationMode = config.generationMode;
                    sendEventToHTMLSource(src, 'indexResult', struct( ...
                        'documentCount', indexedDocumentCount, ...
                        'chunkCount', pipeline.Store.numChunks(), ...
                        'chunker', char(config.chunker), ...
                        'retriever', char(config.retriever), ...
                        'topK', config.topK));

                case "runQuery"
                    if isempty(pipeline) || ~pipeline.IsIndexed
                        error('OceanNLP:RAGDemo:NotIndexed', ...
                            'Index a corpus with the current settings before running a query.');
                    end
                    question = strtrim(getText_(eventData, 'question', 4000));
                    if strlength(question) == 0
                        error('OceanNLP:RAGDemo:EmptyQuestion', 'Enter a question before running the pipeline.');
                    end
                    sendEventToHTMLSource(src, 'operationStarted', ...
                        struct('operation', 'query', 'message', ...
                        'Running retrieval, prompt assembly, and generation in MATLAB...'));
                    result = pipeline.query(question);
                    payload = struct( ...
                        'query', char(result.Query), ...
                        'answer', char(result.Answer), ...
                        'chunkIds', reshape(double(result.ChunkIds), 1, []), ...
                        'scores', reshape(double(result.Scores), 1, []), ...
                        'chunks', {cellstr(result.Chunks(:))}, ...
                        'prompt', char(result.Prompt), ...
                        'documentCount', indexedDocumentCount, ...
                            'generationMode', char(activeGenerationMode));
                    sendEventToHTMLSource(src, 'queryResult', payload);

                case "configurationChanged"
                    pipeline = [];
                    indexedDocumentCount = 0;
                    activeGenerationMode = "";
                    sendEventToHTMLSource(src, 'indexInvalidated', ...
                        struct('message', 'Settings or corpus changed. Re-index to run a query.'));

                otherwise
                    warning('OceanNLP:RAGDemo:UnknownEvent', ...
                        'Ignoring unsupported UI event "%s".', eventName);
                    sendEventToHTMLSource(src, 'error', struct( ...
                        'event', char(eventName), ...
                        'message', 'The UI requested an unsupported operation.'));
            end
        catch ME
            fprintf(2, '[OceanNLP RAG UI] %s: %s\n', eventName, ME.message);
            try
                sendEventToHTMLSource(src, 'error', struct( ...
                    'event', char(eventName), ...
                    'identifier', ME.identifier, ...
                    'message', ME.message));
            catch bridgeError
                fprintf(2, '[OceanNLP RAG UI] Could not return the error to the UI: %s\n', ...
                    bridgeError.message);
            end
        end
    end
end

function config = validateConfig_(data)
required = {'corpus', 'chunker', 'windowSize', 'overlap', ...
    'maxCharacters', 'windowRadius', 'topK', 'endpoint', 'model'};
if ~isstruct(data) || ~isscalar(data)
    error('OceanNLP:RAGDemo:InvalidRequest', 'The UI request must be a scalar object.');
end
for k = 1:numel(required)
    if ~isfield(data, required{k})
        error('OceanNLP:RAGDemo:MissingField', 'The UI request is missing "%s".', required{k});
    end
end

config.corpus = getText_(data, 'corpus', 500000);
config.chunker = getText_(data, 'chunker', 32);
if ~ismember(config.chunker, ["sliding", "recursive", "sentence-window"])
    error('OceanNLP:RAGDemo:InvalidChunker', 'Choose a supported chunking strategy.');
end
config.retriever = "bm25";
if isfield(data, 'retriever')
    config.retriever = getText_(data, 'retriever', 32);
end
if ~ismember(config.retriever, ["bm25", "semantic"])
    error('OceanNLP:RAGDemo:InvalidRetriever', ...
        'Choose BM25 lexical or TF-IDF cosine retrieval.');
end
config.windowSize = 128;
config.overlap = 32;
config.maxCharacters = 1000;
config.windowRadius = 1;
switch config.chunker
    case "sliding"
        config.windowSize = getNumber_(data, 'windowSize', 1, 10000, true);
        config.overlap = getNumber_(data, 'overlap', 0, 9999, true);
        if config.overlap >= config.windowSize
            error('OceanNLP:RAGDemo:InvalidOverlap', ...
                'Sliding-window overlap must be smaller than its window size.');
        end
    case "recursive"
        config.maxCharacters = getNumber_(data, 'maxCharacters', 1, 100000, true);
    case "sentence-window"
        config.windowRadius = getNumber_(data, 'windowRadius', 0, 100, true);
end
config.topK = getNumber_(data, 'topK', 1, 20, true);
config.generationMode = "demo";
if isfield(data, 'generationMode')
    config.generationMode = getText_(data, 'generationMode', 16);
end
if ~ismember(config.generationMode, ["demo", "local"])
    error('OceanNLP:RAGDemo:InvalidGenerationMode', ...
        'Choose either demo or local generation.');
end
config.endpoint = getText_(data, 'endpoint', 2000);
config.model = getText_(data, 'model', 256);
if config.generationMode == "local"
    if strlength(config.endpoint) == 0 || ...
            isempty(regexp(char(config.endpoint), '^https?://[^/\s]+', 'once'))
        error('OceanNLP:RAGDemo:InvalidEndpoint', ...
            'The local generation endpoint must be an HTTP or HTTPS URL.');
    end
    if strlength(config.model) == 0
        error('OceanNLP:RAGDemo:InvalidModel', 'Enter a model identifier.');
    end
end
end

function value = getText_(data, fieldName, maxLength)
raw = data.(fieldName);
if ischar(raw) && (isrow(raw) || isempty(raw))
    value = string(raw);
elseif isstring(raw) && isscalar(raw)
    value = raw;
else
    error('OceanNLP:RAGDemo:InvalidText', 'Field "%s" must be a text value.', fieldName);
end
if strlength(value) > maxLength
    error('OceanNLP:RAGDemo:TextTooLong', 'Field "%s" exceeds the %d-character limit.', ...
        fieldName, maxLength);
end
end

function value = getNumber_(data, fieldName, lowerBound, upperBound, requireInteger)
raw = data.(fieldName);
if ~isnumeric(raw) || ~isscalar(raw)
    error('OceanNLP:RAGDemo:InvalidNumber', 'Field "%s" must be a numeric scalar.', fieldName);
end
value = double(raw);
if ~isfinite(value) || value < lowerBound || value > upperBound || ...
        (requireInteger && value ~= fix(value))
    error('OceanNLP:RAGDemo:NumberOutOfRange', ...
        'Field "%s" must be %s between %g and %g.', ...
        fieldName, ternary_(requireInteger, 'an integer', 'a number'), lowerBound, upperBound);
end
end

function documents = parseDocuments_(raw)
parts = regexp(char(raw), '(?m)^[ \t]*---[ \t]*\r?$', 'split');
documents = strip(string(parts));
documents(documents == "") = [];
if isempty(documents)
    error('OceanNLP:RAGDemo:EmptyCorpus', ...
        'Enter at least one document. Separate multiple documents with a line containing ---.');
end
end

function answer = demoGenerate_(prompt)
% Keep the mock explicit: the prompt is built, but no model is called.
arguments
    prompt (1,1) string
end
answer = "DEMO GENERATOR — no language model was called. " + ...
    "The real corpus was indexed, the selected retriever returned context, and the RAG prompt " + ...
    "was assembled successfully (" + string(strlength(prompt)) + ...
    " characters). Inspect the Retrieved context and Final prompt tabs.";
end

function value = ternary_(condition, whenTrue, whenFalse)
if condition
    value = whenTrue;
else
    value = whenFalse;
end
end
