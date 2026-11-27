function model = cachedFit(store, key, constructorFn, X, y)
    % CACHEDFIT  Train-once, load-on-rerun model caching.
    %
    %   model = OceanTools.cachedFit(store, key, @() LDA(), X_train, y_train)
    %
    % If the model has been trained before (same key), it is loaded from
    % the FeatureStore.  Otherwise, the constructor is called, the model
    % is fitted on (X, y), and saved under that key for next time.
    %
    % The key string encodes everything that makes a run unique:
    %   dataset, seed, model name, hyperparameters.
    % If you change ANY of those, use a different key.
    %
    % Inputs:
    %   store          - OceanData.DataArch.FeatureStore (root = model cache dir)
    %   key            - Unique string, e.g. 'vowel/seed42/LDA'
    %   constructorFn  - Function handle that returns a fresh model, e.g. @() LDA()
    %   X              - Training feature matrix   (N × p)
    %   y              - Training label vector      (N × 1)
    %
    % Output:
    %   model - The trained (or cached) model object.
    %
    % Example:
    %   store = OceanData.DataArch.FeatureStore('rootDir', './cache/models/');
    %   key   = sprintf('vowel/seed42/LDA');
    %   lda   = OceanTools.cachedFit(store, key, @() LDA(), X_train, y_train);
    %
    % See also: OceanData.DataArch.FeatureStore, OceanTools.ExperimentContext.cachedFit

    arguments
        store          % OceanData.DataArch.FeatureStore
        key            (1,1) string
        constructorFn  (1,1) function_handle
        X              double
        y
    end

    entry = store.getOrCompute(key, @() buildAndFit_(constructorFn, X, y));
    model = entry.data;
end

function model = buildAndFit_(constructorFn, X, y)
    model = constructorFn();
    model.fit(X, y);
end
