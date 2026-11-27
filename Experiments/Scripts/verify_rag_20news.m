%VERIFY_RAG_20NEWS Load the repository's text corpus and run an end-to-end RAG query.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataRoot = fullfile(root, '+OceanTest', '+OceanNLP', 'System', 'Validation', 'validation', '20news_raw_data');
files = dir(fullfile(dataRoot, '**', '*'));
files = files(~[files.isdir]);
assert(~isempty(files), 'No 20 Newsgroups text files found.');
n = min(20, numel(files));
documents = strings(1, n);
for i = 1:n
    documents(i) = string(fileread(fullfile(files(i).folder, files(i).name)));
end

chunker = OceanNLP.Chunking.FixedTokenChunker(ChunkSize=80);
ranker = OceanNLP.Retrieval.BM25Ranker(Strategy="inverted_index");
generator = OceanNLP.Generation.FunctionGenerator(@(~) "verified");
pipeline = OceanNLP.RAG.RAGPipeline(Chunker=chunker, Ranker=ranker, Generator=generator, TopK=3);
pipeline.index(documents);
result = pipeline.query("What is discussed in these documents?");

fprintf('RAG_VERIFY_OK\n');
fprintf('Documents=%d Chunks=%d Retrieved=%d\n', n, ranker.NumDocs, numel(result.ChunkIds));
fprintf('Answer=%s\n', result.Answer);
fprintf('PromptChars=%d\n', strlength(result.Prompt));
