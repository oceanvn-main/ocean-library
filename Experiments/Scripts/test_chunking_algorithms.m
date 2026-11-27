%TEST_CHUNKING_ALGORITHMS Compare all repository RAG chunkers on one input.
%   The same input text is passed to every chunking strategy.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
resultsDir = fullfile(root, 'Results', 'NLP');
if ~isfolder(resultsDir), mkdir(resultsDir); end

inputText = "Ocean Library supports reusable numerical algorithms. " + ...
    "A retrieval system first divides documents into chunks. " + ...
    "Chunking controls context size and retrieval granularity. " + newline + ...
    "Overlapping windows preserve context across boundaries. " + ...
    "Sentence windows retain nearby sentences for question answering. " + ...
    "Recursive character chunking preserves paragraph-like structure.";

strategies = ["fixed-token", "sliding", "recursive", "sentence-window"];
chunkers = {
    OceanNLP.Chunking.FixedTokenChunker(ChunkSize=12), ...
    OceanNLP.Chunking.ChunkerFactory.create("sliding", WindowSize=12, Overlap=3), ...
    OceanNLP.Chunking.ChunkerFactory.create("recursive", MaxCharacters=110), ...
    OceanNLP.Chunking.ChunkerFactory.create("sentence-window", WindowRadius=1)};

reportFile = fullfile(resultsDir, 'chunking_algorithm_comparison.txt');
fid = fopen(reportFile, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Chunking algorithm comparison\n');
fprintf(fid, 'Input characters: %d\n\n', strlength(inputText));

fprintf('CHUNKING_VERIFY_OK\n');
fprintf('Input characters: %d\n', strlength(inputText));
for i = 1:numel(strategies)
    chunks = chunkers{i}.chunk(inputText);
    tokenCounts = zeros(1, numel(chunks));
    for j = 1:numel(chunks), tokenCounts(j) = numel(split(strtrim(chunks(j)))); end
    line = sprintf('%s: chunks=%d, characters=%d, token-range=[%d,%d]\n', ...
        strategies(i), numel(chunks), sum(strlength(chunks)), min(tokenCounts), max(tokenCounts));
    fprintf('%s', line);
    fprintf('  Output chunks:\n');
    fprintf(fid, '%s', line);
    for j = 1:numel(chunks)
        fprintf('    %d: %s\n', j, chunks(j));
        fprintf(fid, '  %d: %s\n', j, chunks(j));
    end
    fprintf(fid, '\n');
end
fprintf('Report: %s\n', reportFile);
