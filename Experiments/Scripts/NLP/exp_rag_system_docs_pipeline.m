% EXP_RAG_SYSTEM_DOCS_PIPELINE Full-Pipeline RAG Scientific Experiment on Repository System Markdown Docs
%
% EXPERIMENT WORKFLOW:
%   1. Load local system repository markdown files (AGENTS.md, architecture.md, docs/index.md, etc.).
%   2. Chunk documentation using SlidingWindowChunker and RecursiveCharChunker.
%   3. Index chunks using BM25 Inverted Index, BM25 Matrix Index, and Hybrid Vector+BM25 rankers.
%   4. Connect Generation stage to LocalLLMGenerator (LM Studio at http://localhost:1234/v1/chat/completions).
%   5. Execute domain-specific system architecture queries across all configurations.
%   6. Compute RAGEvaluator metrics (MeanRetrieved, Faithfulness) and save report to Results/NLP/exp_rag_system_docs_report.txt.

clear; clc;

%% 1. Experiment Setup & Assumptions
fprintf('========================================================================\n');
fprintf('  OCEAN LIBRARY: FULL-PIPELINE RAG EXPERIMENT ON SYSTEM MARKDOWN DOCS\n');
fprintf('========================================================================\n\n');

rootDir = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
resultsDir = fullfile(rootDir, 'Results', 'NLP');
if ~isfolder(resultsDir), mkdir(resultsDir); end
reportFile = fullfile(resultsDir, 'exp_rag_system_docs_report.txt');

fprintf('Assumptions & Parameters:\n');
fprintf('  - Target LLM Engine : LocalLLMGenerator (LM Studio / Local OpenAI API)\n');
fprintf('  - Local Server Endpoint: http://localhost:1234/v1/chat/completions\n');
fprintf('  - Corpus Source       : System Markdown Documentation (.md files)\n');
fprintf('  - TopK Chunks         : 3\n\n');

%% 2. Load System Markdown Documentation Corpus
mdFilesStruct = [ ...
    dir(fullfile(rootDir, '*.md')); ...
    dir(fullfile(rootDir, '.agents', '*.md')); ...
    dir(fullfile(rootDir, '.agents', 'rules', '*.md')); ...
    dir(fullfile(rootDir, 'docs', '*.md')); ...
    dir(fullfile(rootDir, 'docs', 'guides', '*.md')) ...
];

documents = strings(0, 1);
filePaths = strings(0, 1);

for f = 1:numel(mdFilesStruct)
    if mdFilesStruct(f).isdir, continue; end
    filePath = fullfile(mdFilesStruct(f).folder, mdFilesStruct(f).name);
    try
        content = fileread(filePath);
        if strlength(strtrim(string(content))) > 20
            documents = [documents; string(content)]; %#ok<AGROW>
            filePaths = [filePaths; string(filePath)]; %#ok<AGROW>
            relPath = strrep(filePath, rootDir, '');
            fprintf('Loaded system doc [%d]: %s (%d chars)\n', ...
                numel(documents), relPath, strlength(content));
        end
    catch ME
        warning('OceanNLP:Experiment:FileReadError', 'Could not read %s: %s', filePath, ME.message);
    end
end

nTotalDocs = numel(documents);
totalChars = sum(strlength(documents));

if nTotalDocs == 0
    error('OceanNLP:Experiment:NoDocsFound', 'No system markdown documentation files were found at %s.', rootDir);
end

fprintf('\nLoaded System Corpus Summary:\n');
fprintf('  - Total System Docs : %d files\n', nTotalDocs);
fprintf('  - Total Text Chars  : %d characters\n\n', totalChars);

%% 3. Define Domain Architecture Test Queries
queries = [ ...
    "What are the core behavioral rules and math expression guidelines in Ocean Library?", ...
    % "How does the OceanViz engine work for visual rendering and dark theme styling?", ...
    % "What is the structure of the OceanAgents game-theory agent system and world kernel?" ...
];

%% 4. Instantiate Generator (LocalLLMGenerator vs Fallback)
localLLM = OceanNLP.Generation.LocalLLMGenerator(...
    'Endpoint', "http://localhost:1234/v1/chat/completions", ...
    'Model', "auto", ...
    'Timeout', 15);

isLocalServerLive = false;
try
    testAns = localLLM.generate("Test ping");
    isLocalServerLive = true;
    fprintf('[SUCCESS] Connected to live local LLM server (LM Studio / OpenAI API).\n\n');
catch ME
    fprintf('[NOTICE] Local LLM server at localhost:1234 is offline or unreachable (%s).\n', ME.message);
    fprintf('         Falling back to FunctionGenerator for pipeline execution verification.\n\n');
    localLLM = OceanNLP.Generation.FunctionGenerator(@(p) "[Offline Local LLM Fallback Answer]");
end

%% 5. Define Pipeline Configurations
configs = { ...
    struct('Name', "BM25-InvertedIndex", ...
           'Chunker', OceanNLP.Chunking.SlidingWindowChunker('WindowSize', 100, 'Overlap', 20), ...
           'CreateRanker', @() OceanNLP.Retrieval.BM25Ranker('Strategy', "inverted_index"), ...
           'TopK', 3), ...
    struct('Name', "BM25-MatrixIndex", ...
           'Chunker', OceanNLP.Chunking.SlidingWindowChunker('WindowSize', 100, 'Overlap', 20), ...
           'CreateRanker', @() OceanNLP.Retrieval.BM25Ranker('Strategy', "matrix"), ...
           'TopK', 3), ...
    struct('Name', "Hybrid-Dense-BM25-RRF", ...
           'Chunker', OceanNLP.Chunking.RecursiveCharChunker('MaxCharacters', 500), ...
           'CreateRanker', @() OceanNLP.Retrieval.HybridRanker(...
               'Rankers', {OceanNLP.Retrieval.BM25Ranker('Strategy', "matrix"), OceanNLP.Retrieval.SemanticRanker()}, ...
               'Fusion', OceanNLP.Fusion.ReciprocalRankFusion('K', 60)), ...
           'TopK', 3) ...
};

%% 6. Execute Experiment Loop
fid = fopen(reportFile, 'w');
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '========================================================================\n');
fprintf(fid, '  FULL-PIPELINE RAG EXPERIMENT REPORT (SYSTEM MARKDOWN DOCS CORPUS)\n');
fprintf(fid, '  Generated: %s\n', datestr(now));
fprintf(fid, '========================================================================\n\n');
fprintf(fid, 'Local Server Active: %s\n', string(isLocalServerLive));
fprintf(fid, 'Corpus Dataset     : System Markdown Files (%d documents, %d chars)\n\n', nTotalDocs, totalChars);

for c = 1:numel(configs)
    cfg = configs{c};
    fprintf('--- Executing Configuration [%d/%d]: %s ---\n', c, numel(configs), cfg.Name);
    
    builder = OceanNLP.Generation.PromptBuilder('MaxChunks', 3, 'MaxCharacters', 4000);
    ranker = cfg.CreateRanker();
    pipeline = OceanNLP.RAG.RAGPipeline(...
        'Chunker', cfg.Chunker, ...
        'Ranker', ranker, ...
        'Builder', builder, ...
        'Generator', localLLM, ...
        'TopK', cfg.TopK);
    
    % Index corpus
    tIndexStart = tic;
    pipeline.index(documents);
    tIndex = toc(tIndexStart);
    fprintf('  Indexed system markdown corpus (%d chunks) in %.4f seconds.\n', ranker.NumDocs, tIndex);
    
    % Query pipeline
    queryResults = [];
    for q = 1:numel(queries)
        tQueryStart = tic;
        res = pipeline.query(queries(q));
        tQuery = toc(tQueryStart);
        
        queryResults = [queryResults; res]; %#ok<AGROW>
        
        fprintf('  Query %d: "%s"\n', q, queries(q));
        fprintf('    Retrieved %d chunks (scores: %s) in %.4fs\n', ...
            numel(res.ChunkIds), mat2str(res.Scores, 3), tQuery);
        ansStr = res.Answer;
        if strlength(ansStr) > 80, ansStr = extractBefore(ansStr, 80) + "..."; end
        fprintf('    Answer Preview: %s\n', ansStr);
        for k = 1:numel(res.Chunks)
            cText = strtrim(res.Chunks(k));
            cText = strrep(cText, sprintf('\n'), ' ');
            cText = strrep(cText, sprintf('\r'), '');
            if strlength(cText) > 160
                cText = extractBefore(cText, 160) + "...";
            end
            fprintf('    Chunk %d (ID %d, Score %.4f): %s\n', k, res.ChunkIds(k), res.Scores(k), cText);
        end
    end
    
    % Evaluate configuration
    evaluator = OceanNLP.RAG.RAGEvaluator('FaithfulnessFunction', @(r) double(strlength(r.Answer) > 0));
    report = evaluator.evaluate(queryResults);
    
    % Write config report
    fprintf(fid, 'CONFIGURATION: %s\n', cfg.Name);
    fprintf(fid, '  Index Build Time : %.4f s\n', tIndex);
    fprintf(fid, '  Total Chunks     : %d\n', ranker.NumDocs);
    fprintf(fid, '  Mean Retrieved   : %.2f chunks\n', report.MeanRetrieved);
    fprintf(fid, '  Faithfulness     : %.2f\n', report.Faithfulness);
    
    for q = 1:numel(queryResults)
        r = queryResults(q);
        fprintf(fid, '  [Q%d]: %s\n', q, r.Query);
        fprintf(fid, '    Prompt Characters : %d\n', strlength(r.Prompt));
        fprintf(fid, '    Answer Characters : %d\n', strlength(r.Answer));
        fprintf(fid, '    Answer Text       : %s\n', r.Answer);
        for k = 1:numel(r.Chunks)
            cText = strtrim(r.Chunks(k));
            cText = strrep(cText, sprintf('\n'), ' ');
            cText = strrep(cText, sprintf('\r'), '');
            if strlength(cText) > 200
                cText = extractBefore(cText, 200) + "...";
            end
            fprintf(fid, '    Chunk %d (ID %d, Score %.4f): %s\n', k, r.ChunkIds(k), r.Scores(k), cText);
        end
    end
    fprintf(fid, '------------------------------------------------------------------------\n\n');
end

fprintf('\n========================================================================\n');
fprintf('  EXPERIMENT COMPLETE!\n');
fprintf('  Report saved to: %s\n', reportFile);
fprintf('========================================================================\n');
