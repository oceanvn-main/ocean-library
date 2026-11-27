# OceanNLP Text Processing & Analysis Architecture Guide

## Overview

`OceanNLP.Text` provides a modular text normalization, token filtering, stemming, and analysis pipeline for `OceanNLP`, modeled after the Lucene/Elasticsearch and Hugging Face Tokenizer architectures.

The text processing workflow follows a 4-stage pipeline orchestrated by `OceanNLP.Text.Analyzer`:

Input Text -> ICharFilter -> ITokenizer -> ITokenFilter / IStemmer -> Final Tokens

All components implement the `OceanNLP.Text.TextStrategy` base class and adhere to the 3-Pillar contract (`getMetadata()`, `getPipelineSpec()`).

### Core Components

- **`ICharFilter`**: Character-level cleanup before tokenization (e.g. `HtmlStripFilter`, `UnicodeNormalizeFilter`, `AccentFolderFilter`).
- **`IPreTokenizer`**: Structural split boundaries before subword/regex tokenization.
- **`ITokenFilter`**: Token-level stream transformations (e.g. `LowercaseFilter`, `StopwordFilter`, `LengthFilter`, `StemFilter`).
- **`IStemmer`**: Morphological word-stemming algorithms (e.g. `PorterStemmer`).
- **`Analyzer`**: Pipeline orchestrator combining character filters, tokenizers, and token filters into a single `ITokenizer`-compatible object.

## Usage

### Basic Usage

Use default presets for common language tasks:

```matlab
% English preset: regex tokenizer + lowercase + stop words + Porter stemmer
analyzer = OceanNLP.Text.Analyzer.preset("english");

% Inject analyzer into CountVectorizer
vectorizer = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', analyzer);

% Fit and transform corpus
corpus = ["The <b>running</b> cats ran quickly!", "Cats and dogs were running."];
vectorizer.fit(corpus);
X_counts = vectorizer.forward(corpus);
```

### Custom Pipeline Construction

Construct custom pipelines by assembling strategy objects:

```matlab
analyzer = OceanNLP.Text.Analyzer( ...
    'CharFilters', { ...
        OceanNLP.Text.CharFilter.HtmlStripFilter(), ...
        OceanNLP.Text.CharFilter.UnicodeNormalizeFilter('Form', "NFC") ...
    }, ...
    'Tokenizer', OceanNLP.Tokenize.RegexTokenizer('Pattern', "[\w']+"), ...
    'TokenFilters', { ...
        OceanNLP.Text.Filter.LowercaseFilter(), ...
        OceanNLP.Text.Filter.StopwordFilter('Language', "english"), ...
        OceanNLP.Text.Filter.LengthFilter('MinLength', 2, 'MaxLength', 25), ...
        OceanNLP.Text.Filter.StemFilter('Stemmer', OceanNLP.Text.Stem.PorterStemmer()) ...
    });
```

### Spec-Driven Pipeline Construction

Use spec structures resolved by `OceanNLP.Text.Factory`:

```matlab
spec = struct( ...
    'Tokenizer', "regex", ...
    'TokenFilters', ["lowercase", "stopword", "stem:porter"]);

analyzer = OceanNLP.Text.Analyzer.fromSpec(spec);
```

## Examples

### Complete End-to-End Search & Retrieval Pipeline

```matlab
% 1. Setup Document Corpus
corpus = [
    "Machine learning algorithms process large datasets efficiently."
    "Deep neural networks require substantial computing power."
    "Statistical learning models analyze structured dataset tables."
];

% 2. Create Analysis Pipeline
analyzer = OceanNLP.Text.Analyzer.preset("english");

% 3. Extract Count Features
vectorizer = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', analyzer);
vectorizer.fit(corpus);
X = vectorizer.forward(corpus);

% 4. Apply BM25 Weighting
bm25 = OceanStats.Data.Weighting.BM25Weighting('K1', 1.2, 'B', 0.75);
bm25.fit(X);
W_bm25 = bm25.forward(X);

% 5. Query Scoring
query = "learning algorithms datasets";
q_tokens = analyzer.forward(query);
q_ids = vectorizer.Vocabulary.encode(q_tokens);
q_vec = zeros(vectorizer.Vocabulary.VocabSize, 1);
for id = q_ids, if id > 1, q_vec(id) = q_vec(id) + 1; end; end

scores = W_bm25 * q_vec;
[sorted_scores, indices] = sort(scores, 'descend');
fprintf('Top matching doc index: %d (Score: %.4f)\n', indices(1), sorted_scores(1));
```

## See Also

- `OceanNLP.FeatureExtraction.CountVectorizer`
- `OceanNLP.Vocab.Vocabulary`
- `OceanNLP.Tokenize.ITokenizer`
- `OceanStats.Data.Weighting.BM25Weighting`
- `OceanStats.Data.Weighting.TfidfWeighting`
