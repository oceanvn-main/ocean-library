classdef ReNetClassifier < OceanDL.nn.Module
    % RENETCLASSIFIER  Image classifier: ReNet2D feature map -> global average
    %   over the patch grid -> linear logits (Visin et al. 2015, simplified).
    %     logits = V * mean_{i,j} ReNet(x)(i, j, :) + c

    properties (SetAccess = protected)
        ReNet
        Head
    end

    methods
        function obj = ReNetClassifier(inChannels, patchSize, hidden1, hidden2, numClasses, opts)
            arguments
                inChannels (1,1) double
                patchSize (1,1) double
                hidden1 (1,1) double
                hidden2 (1,1) double
                numClasses (1,1) double
                opts.CellType (1,1) string = "gru"
                opts.Seed = []
            end
            obj.ReNet = OceanDL.Recurrent.ReNet2D(inChannels, patchSize, hidden1, hidden2, ...
                'CellType', opts.CellType, 'Seed', opts.Seed);
            obj.Head = OceanDL.nn.Linear(2 * hidden2, numClasses, 'Seed', opts.Seed);
        end

        function logits = forward(obj, X)
            Y = obj.ReNet.forward(X);                         % Hp x Wp x C2 x N
            [~, ~, C2, N] = size(Y, 1:4);
            pooled = reshape(mean(mean(Y, 1), 2), C2, N);     % global average -> C2 x N
            logits = obj.Head.forward(pooled);
        end
    end
end
