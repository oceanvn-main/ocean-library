classdef (Abstract) IVectorizer < handle
    % IVECTORIZER  Abstract interface: raw document corpus -> numeric feature matrix.
    %
    %   PyTorch-style contract, same shape as IWeighting: fit() mutates
    %   buffers (vocabulary, any wrapped IWeighting's fitted state) on self;
    %   forward() reads them back — no separate returned state struct.
    %
    %   CONVENTION:
    %     fit(corpus)      — corpus: (1 x N) string/cellstr array of raw documents.
    %     forward(corpus)  — (N x V) sparse numeric matrix, V = vocabulary size
    %                        fixed at fit() time.
    %
    %   See also: OceanNLP.FeatureExtraction.CountVectorizer
    %             OceanNLP.FeatureExtraction.TfidfVectorizer
    methods (Abstract)
        fit(obj, corpus)
        X = forward(obj, corpus)
        tf = isFitted(obj)
    end
end
