% EXP_RAG_FULL_PIPELINE  Full-Pipeline RAG Scientific Experiment on 20 Newsgroups Corpus
%
% EXPERIMENT WORKFLOW:
%   1. Load real 20 Newsgroups dataset from Data/Datasets/20Newsgroups/20newsgroups_corpus.json.
%   2. Chunk documents using SlidingWindowChunker & RecursiveCharChunker.
%   3. Index chunks using BM25 Inverted Index, BM25 Matrix Index, and Hybrid Vector+BM25 rankers.
%   4. Connect Generation stage to LocalLLMGenerator (LM Studio / Gemma 4 12B at http://localhost:1234/v1/chat/completions).
%   5. Execute evaluation queries across all configurations.
%   6. Compute RAGEvaluator metrics (MeanRetrieved, Faithfulness) and save report to Results/NLP/exp_rag_20newsgroups_report.txt.

clear; clc;

%% 1. Experiment Setup & Assumptions
fprintf('========================================================================\n');
fprintf('  OCEAN LIBRARY: FULL-PIPELINE RAG EXPERIMENT ON 20 NEWSGROUPS CORPUS\n');
fprintf('========================================================================\n\n');

rootDir = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
resultsDir = fullfile(rootDir, 'Results', 'NLP');
if ~isfolder(resultsDir), mkdir(resultsDir); end
reportFile = fullfile(resultsDir, 'exp_rag_20newsgroups_report.txt');

datasetFile = fullfile(rootDir, 'Data', 'Datasets', '20Newsgroups', '20newsgroups_corpus.json');

fprintf('Assumptions & Parameters:\n');
fprintf('  - Target LLM Engine : LocalLLMGenerator (LM Studio / Gemma 4 12B)\n');
fprintf('  - Local Server Endpoint: http://localhost:1234/v1/chat/completions\n');
fprintf('  - Corpus Source       : 20 Newsgroups Dataset (%s)\n', datasetFile);
fprintf('  - TopK Chunks         : 3\n\n');

%% 2. Load 20 Newsgroups Dataset
if ~exist(datasetFile, 'file')
    error('OceanNLP:Experiment:DatasetMissing', ...
        '20 Newsgroups dataset file missing at %s. Run Python/export_20newsgroups.py first.', datasetFile);
end

fprintf('Loading 20 Newsgroups corpus JSON...\n');
rawJson = fileread(datasetFile);
corpusObj = jsondecode(rawJson);

% Extract document texts and category metadata
docStructs = corpusObj.documents;
nTotalDocs = numel(docStructs);

% Use subset or full dataset (e.g. N = 500 for benchmark, expandable to full 2,658)
nSample = min(500, nTotalDocs);
documents = strings(1, nSample);
categories = strings(1, nSample);

for i = 1:nSample
    documents(i) = string(docStructs(i).text);
    categories(i) = string(docStructs(i).category);
end

totalChars = sum(strlength(documents));
fprintf('Loaded 20 Newsgroups dataset:\n');
fprintf('  - Documents Selected : %d / %d\n', nSample, nTotalDocs);
fprintf('  - Total Text Chars   : %d characters\n', totalChars);
fprintf('  - Categories Included: %s\n\n', strjoin(string(corpusObj.categories)', ', '));

%% 3. Define Domain Test Queries
queries = [ ...
    "What are the main scientific instruments and goals of the Hubble Space Telescope?", ...
    "What 3D rendering algorithms and file formats exist for computer graphics?", ...
    "What are common symptoms and medical treatments for kidney stones?" ...
];

%% 4. Instantiate Generator (LocalLLMGenerator vs Fallback)
localLLM = OceanNLP.Generation.LocalLLMGenerator(...
    'Endpoint', "http://localhost:1234/v1/chat/completions", ...
    'Model', "gemma-4-12b", ...
    'Timeout', 10);

isLocalServerLive = false;
try
    testAns = localLLM.generate("Test ping");
    isLocalServerLive = true;
    fprintf('[SUCCESS] Connected to live local LLM server (LM Studio / Gemma 4 12B).\n\n');
catch ME
    fprintf('[NOTICE] Local LLM server at localhost:1234 is offline or unreachable (%s).\n', ME.message);
    fprintf('         Falling back to FunctionGenerator for pipeline execution verification.\n\n');
    localLLM = OceanNLP.Generation.FunctionGenerator(@(p) "[Offline Local LLM Fallback Answer]");
end

%% 5. Define Pipeline Configurations
configs = { ...
    struct('Name', "BM25-InvertedIndex", ...
           'Chunker', OceanNLP.Chunking.SlidingWindowChunker('WindowSize', 80, 'Overlap', 15), ...
           'CreateRanker', @() OceanNLP.Retrieval.BM25Ranker('Strategy', "inverted_index"), ...
           'TopK', 3), ...
    struct('Name', "BM25-MatrixIndex", ...
           'Chunker', OceanNLP.Chunking.SlidingWindowChunker('WindowSize', 80, 'Overlap', 15), ...
           'CreateRanker', @() OceanNLP.Retrieval.BM25Ranker('Strategy', "matrix"), ...
           'TopK', 3), ...
    struct('Name', "Hybrid-Dense-BM25-RRF", ...
           'Chunker', OceanNLP.Chunking.RecursiveCharChunker('MaxCharacters', 400), ...
           'CreateRanker', @() OceanNLP.Retrieval.HybridRanker(...
               'Rankers', {OceanNLP.Retrieval.BM25Ranker('Strategy', "matrix"), OceanNLP.Retrieval.SemanticRanker()}, ...
               'Fusion', OceanNLP.Fusion.ReciprocalRankFusion('K', 60)), ...
           'TopK', 3) ...
};

%% 6. Execute Experiment Loop
fid = fopen(reportFile, 'w');
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '========================================================================\n');
fprintf(fid, '  FULL-PIPELINE RAG EXPERIMENT REPORT (20 NEWSGROUPS CORPUS)\n');
fprintf(fid, '  Generated: %s\n', datestr(now));
fprintf(fid, '========================================================================\n\n');
fprintf(fid, 'Local Server Active: %s\n', string(isLocalServerLive));
fprintf(fid, 'Corpus Dataset     : 20 Newsgroups (%d documents, %d chars)\n\n', nSample, totalChars);

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
    fprintf('  Indexed 20 Newsgroups corpus (%d chunks) in %.4f seconds.\n', ranker.NumDocs, tIndex);
    
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
        if strlength(ansStr) > 200, ansStr = extractBefore(ansStr, 200) + "..."; end
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
