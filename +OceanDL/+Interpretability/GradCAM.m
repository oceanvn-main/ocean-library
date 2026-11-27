classdef GradCAM
    % GRADCAM  "Which high-level spatial features support this class?"
    %   Selvaraju et al., 2017. Gradient of the class score w.r.t. a
    %   target conv layer's feature maps, global-average-pooled per
    %   channel into importance weights, then a ReLU'd weighted sum of the
    %   feature maps -- exactly the case OceanDL.Autograd.Tape
    %   was designed for: a gradient w.r.t. an INTERMEDIATE node, not the
    %   input.
    %
    %   USAGE:
    %     r = GradCAM.compute(engine, X, 3, 'conv2', 'OutputLayer', 'dense1');
    %     r.CAMNormalized   % (H, W, N), [0,1], resize to input size to overlay

    methods (Static)
        function result = compute(engine, X, targetClass, convLayerName, opts)
            arguments
                engine
                X (:,:,:,:) double
                targetClass (1,1) double {mustBePositive, mustBeInteger}
                convLayerName (1,:) char
                opts.OutputLayer (1,:) char = ''
            end
            outputLayer = opts.OutputLayer;
            if isempty(outputLayer)
                outputLayer = engine.Spec.LayerNames{end};
            end

            Y = engine.forward(X);
            A = engine.getActivations(convLayerName);     % (H, W, C, N)
            seed = zeros(size(Y));
            seed(targetClass, :) = 1;
            dA = engine.gradient(outputLayer, convLayerName, seed);   % (H, W, C, N)

            alpha = mean(dA, [1 2]);                        % (1, 1, C, N)
            cam = sum(alpha .* A, 3);                        % (H, W, 1, N)
            cam = max(cam, 0);                                % ReLU
            cam = reshape(cam, size(cam, 1), size(cam, 2), size(cam, 4));  % (H, W, N)

            camNorm = OceanDL.Interpretability.GradCAM.normalizePerSample(cam);
            result = struct('CAM', cam, 'CAMNormalized', camNorm, 'Alpha', alpha, 'FeatureMap', A);
        end
    end

    methods (Static, Access = private)
        function out = normalizePerSample(cam)
            out = zeros(size(cam));
            for n = 1:size(cam, 3)
                c = cam(:, :, n);
                lo = min(c(:)); hi = max(c(:));
                if hi > lo
                    out(:, :, n) = (c - lo) / (hi - lo);
                end
            end
        end
    end
end
