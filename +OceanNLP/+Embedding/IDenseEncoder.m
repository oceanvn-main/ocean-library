classdef (Abstract) IDenseEncoder < handle
    % IDENSEENCODER  Abstract interface: raw document corpus -> dense (N x D) embedding matrix.
    %
    %   PyTorch-style contract, same shape as IVectorizer/ITokenizer: fit()
    %   mutates buffers on self; forward() reads them back — no separate
    %   returned state struct. Frozen by Retrieval System Master Plan §3.3;
    %   this file must not diverge from that contract.
    %
    %   CONVENTION:
    %     fit(corpus)      — corpus: (1 x N) string/cellstr array of raw
    %                        documents. For a pretrained encoder (e.g.
    %                        BertEncoder) this is a documented no-op beyond
    %                        flipping isFitted() — see that class's POLICY
    %                        note — but every implementation MUST still
    %                        require fit() before forward(), for calling-
    %                        pattern uniformity with every other class in
    %                        this library.
    %     forward(corpus)  — (N x D) dense numeric matrix, D = embeddingDim().
    %     embeddingDim()   — D, the fixed output width of this encoder.
    %     name()           — short descriptive string identifying the encoder
    %                        (and, where applicable, the specific model).
    %     isFitted()       — true iff fit() has been called.
    %
    %   See also: OceanNLP.Embedding.BertEncoder, OceanNLP.Embedding.DenseEncoderFactory,
    %             OceanNLP.FeatureExtraction.IVectorizer
    methods (Abstract)
        fit(obj, corpus)
        E = forward(obj, corpus)
        d = embeddingDim(obj)
        s = name(obj)
        tf = isFitted(obj)
    end
end
