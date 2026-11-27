# OceanNLP Recipes & Code Snippets

Copy-pasteable MATLAB recipes for text analysis, custom tokenization, feature extraction, and weighting pipelines in `OceanNLP`.

---

## 1. Quick English Preset Analyzer

```matlab
% Create an English analyzer with stemming and stop words
analyzer = OceanNLP.Text.Analyzer.preset("english");
tokens = analyzer.forward("The fast running foxes jumped over the lazy dogs!");
disp(tokens);
% Result: ["fast", "run", "fox", "jump", "lazi", "dog"]
```

---

## 2. Count Vectorizer with Custom Analyzer

```matlab
% Define raw corpus
corpus = [
    "HTML <b>formatting</b> should be stripped automatically!"
    "Stripping <i>HTML</i> tags improves token clean-up."
];

% Configure custom analyzer with HTML character filter
analyzer = OceanNLP.Text.Analyzer( ...
    'CharFilters', { OceanNLP.Text.CharFilter.HtmlStripFilter() }, ...
    'Tokenizer', OceanNLP.Tokenize.RegexTokenizer(), ...
    'TokenFilters', { OceanNLP.Text.Filter.LowercaseFilter() });

% Fit vectorizer and extract sparse count matrix
vectorizer = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', analyzer);
vectorizer.fit(corpus);
X_counts = vectorizer.forward(corpus);

fprintf('Corpus size: %d docs x %d vocabulary terms\n', size(X_counts, 1), size(X_counts, 2));
```

---

## 3. End-to-End TF-IDF & Cosine Similarity

```matlab
% Corpus
corpus = [
    "artificial intelligence and machine learning"
    "machine learning models and statistical analysis"
    "deep learning neural networks for computer vision"
];

% Process with English preset
analyzer = OceanNLP.Text.Analyzer.preset("english");
vectorizer = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', analyzer);
vectorizer.fit(corpus);
X = vectorizer.forward(corpus);

% Compute L2-normalized TF-IDF
tfidf = OceanStats.Data.Weighting.TfidfWeighting.create('Norm', "l2");
tfidf.fit(X);
W_tfidf = full(tfidf.forward(X));

% Compute document-to-document Cosine Similarity
cos_sim = OceanStats.Similarity.CosineSimilarity();
S = cos_sim.pairwise(W_tfidf, W_tfidf);
disp("Pairwise Cosine Similarity:");
disp(S);
```

---

## 4. Lucene/Elasticsearch-style BM25 Search Ranking

```matlab
corpus = [
    "the quick brown fox jumps over the lazy dog"
    "quick brown foxes are fast and smart animals"
    "lazy dogs sleep all day in the warm sun"
];

% Tokenize & Vectorize
vectorizer = OceanNLP.FeatureExtraction.CountVectorizer();
vectorizer.fit(corpus);
X = vectorizer.forward(corpus);

% Compute BM25 Weights
bm25 = OceanStats.Data.Weighting.BM25Weighting('K1', 1.2, 'B', 0.75);
bm25.fit(X);
W_bm25 = full(bm25.forward(X));

% Score a Search Query
query = "quick fox";
q_counts = full(vectorizer.forward(query))';  % (V x 1)
scores = W_bm25 * q_counts;                   % (N x 1)

[~, ranked_idx] = sort(scores, 'descend');
fprintf('Top ranked document index: %d\n', ranked_idx(1));
```
